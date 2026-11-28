"""Real CPU GRPO tests, including native loss masking and interrupted resume."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from trl import GRPOConfig

from contracts import GameSpec, RewardBreakdown
from integration.artifacts import latest_checkpoint
from parser.deterministic import QuestionIDParser
from reward_design.rewards import FixtureReward
from rl_pipeline.config import RunConfig
from rl_pipeline.models import load_inference_policy, load_training_policy
from rl_pipeline.rollout import public_prompt
from rl_pipeline.train import default_trainer_options, train_worker
from rl_pipeline.trainer import GameGRPOTrainer, RunCallback


def run_config() -> RunConfig:
    return RunConfig(dataset_path=str(Path(__file__).parent / "fixtures/animals.json"), max_moves=3)


def test_native_grpo_masks_environment_tokens(
    tmp_path: Path,
    assets: Path,
    game: GameSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(1)
    run = replace(run_config(), max_moves=3)
    model, tokenizer = load_training_policy(run, assets, dtype=torch.float32)
    options = default_trainer_options(str(tmp_path))
    prompt = public_prompt(game)
    trainer = GameGRPOTrainer(
        games_by_prompt={prompt: game},
        parser=QuestionIDParser(),
        scorer=FixtureReward(),
        run=run,
        model=model,
        processing_class=tokenizer,
        args=GRPOConfig(**options),
        train_dataset=Dataset.from_dict({"prompt": [prompt]}),
    )
    model.train()
    # The native training loop sets this before calling compute_loss.
    trainer.current_gradient_accumulation_steps = 1
    batch = trainer._generate_and_score_completions([{"prompt": prompt} for _ in range(4)])
    mask = batch["completion_mask"] * batch["tool_mask"]
    assert (mask == 0).any() and (mask == 1).any()
    assert torch.isfinite(batch["advantages"]).all()
    assert batch["advantages"].abs().sum() > 0
    with torch.no_grad():
        recomputed, _ = trainer._get_per_token_logps_and_entropies(
            model,
            torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1),
            torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1),
            batch["completion_ids"].shape[1],
        )
    assert torch.allclose(
        recomputed[mask.bool()],
        batch["sampling_per_token_logps"][mask.bool()],
        atol=1e-5,
    )
    captured: list[torch.Tensor] = []
    original = trainer._get_per_token_logps_and_entropies

    def record(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if result[0].requires_grad:
            result[0].retain_grad()
            captured.append(result[0])
        return result

    monkeypatch.setattr(trainer, "_get_per_token_logps_and_entropies", record)
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert captured[0].grad is not None
    assert torch.count_nonzero(captured[0].grad[mask == 0]) == 0
    assert torch.count_nonzero(captured[0].grad[mask == 1]) > 0
    # Equal-return groups also pass through native advantage calculation.
    monkeypatch.setattr(trainer.scorer, "score", lambda trace, game: RewardBreakdown({"zero": 0.0}))
    trainer.state.global_step = 1
    equal = trainer._generate_and_score_completions([{"prompt": prompt} for _ in range(4)])
    assert torch.isfinite(equal["advantages"]).all()
    assert torch.count_nonzero(equal["advantages"]) == 0


def test_resume_matches_uninterrupted_cpu_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninterrupted = tmp_path / "uninterrupted"
    restarted = tmp_path / "restarted"
    run = run_config()
    train_worker(run, default_trainer_options(str(uninterrupted)))
    original_end = RunCallback.on_step_end

    def interrupt(self: RunCallback, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        original_end(self, args, state, control, **kwargs)
        if state.global_step == 2:
            raise RuntimeError("Simulated notebook interruption")

    monkeypatch.setattr(RunCallback, "on_step_end", interrupt)
    with pytest.raises(RuntimeError, match="Simulated"):
        train_worker(run, default_trainer_options(str(restarted)))
    monkeypatch.setattr(RunCallback, "on_step_end", original_end)
    assert latest_checkpoint(restarted) == restarted / "checkpoint-1"
    train_worker(run, default_trainer_options(str(restarted)), "latest")
    assert "reuse_batch" in (restarted / "events-0.jsonl").read_text()
    first = load_file(uninterrupted / "final/adapter_model.safetensors")
    second = load_file(restarted / "final/adapter_model.safetensors")
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(torch.count_nonzero(value) for name, value in first.items() if "lora_B" in name)
    model, tokenizer = load_inference_policy(restarted / "final")
    assert not model.training and tokenizer.pad_token_id is not None
    assert list((restarted / "tensorboard").glob("events.out.tfevents.*"))
    train_worker(run, default_trainer_options(str(restarted)), "latest")
    assert latest_checkpoint(restarted) == restarted / "checkpoint-2"


def test_incomplete_checkpoint_is_ignored(tmp_path: Path) -> None:
    from integration.artifacts import mark_checkpoint, write_json

    good = tmp_path / "checkpoint-1"
    write_json(good / "state.json", {"step": 1})
    mark_checkpoint(good)
    write_json(tmp_path / "checkpoint-2/state.json", {"step": 2})
    assert latest_checkpoint(tmp_path) == good
    write_json(good / "state.json", {"step": 99})
    assert latest_checkpoint(tmp_path) is None


def test_backup_has_verified_checkpoint_and_private_metadata(tmp_path: Path) -> None:
    backup = tmp_path / "durable"
    root = tmp_path / "local"
    run = replace(run_config(), backup_dir=str(backup))
    train_worker(run, default_trainer_options(str(root)))
    assert latest_checkpoint(backup) is not None
    assert json.loads((backup / "run.json").read_text())["world_size"] == 1
    from rl_pipeline.profiling import summarize_run

    summary = summarize_run(root, warmup_updates=1)
    assert summary["measured_updates"] == 1
    assert summary["action_tokens_per_second"] > 0
    metrics_text = (root / "metrics-0.jsonl").read_text()
    assert '"loss"' in metrics_text and '"grad_norm"' in metrics_text


def test_gradient_accumulation_keeps_complete_groups(tmp_path: Path) -> None:
    options = default_trainer_options(str(tmp_path))
    options.update(max_steps=1, gradient_accumulation_steps=2)
    train_worker(run_config(), options)
    records = list((tmp_path / "batches/step-00000000/rank-0").glob("*.json"))
    assert len(records) == 8
    groups: dict[str, set[str]] = {}
    for path in records:
        trace = json.loads(path.read_text())["trajectory"]
        groups.setdefault(trace["group_id"], set()).add(trace["game_id"])
    assert len(groups) == 2 and all(len(ids) == 1 for ids in groups.values())
