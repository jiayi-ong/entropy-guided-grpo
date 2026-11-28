"""A narrow TRL 0.29.0 adapter for local multi-turn generation.

TRL keeps ownership of GRPO advantages, loss, accumulation, optimization, and
distributed gradient synchronization. This file overrides only generation.
The version pin and mask integration tests protect that internal boundary.
"""

import hashlib
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from accelerate.utils import broadcast_object_list, gather_object
from transformers import TrainerCallback
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation

from contracts import GameSpec
from integration.artifacts import (
    append_json,
    fingerprint,
    mark_checkpoint,
    publish_run,
    write_json,
)
from parser.base import QuestionParser
from reward_design.rewards import RewardCalculator
from rl_pipeline.config import RunConfig
from rl_pipeline.models import TransformersPolicy
from rl_pipeline.rollout import play_games


def episode_reward(
    completions: list[Any], episode_rewards: list[float], **kwargs: Any
) -> list[float]:
    """Pass the driver's complete-game returns to native TRL reward handling."""
    if len(completions) != len(episode_rewards):
        raise ValueError("Every complete trajectory needs one reward.")
    return episode_rewards


def rng_state() -> dict[str, Any]:
    """Store all sampling streams after a cached batch has been generated."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Cache reuse must advance randomness exactly as generation did."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_hash(model: Any) -> str:
    """Bind cached rollouts to their exact LoRA weights and base-model identity."""
    digest = hashlib.sha256(model.config.to_json_string().encode())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class GameGRPOTrainer(GRPOTrainer):
    """Replace single-turn generation with the shared game-playing function."""

    def __init__(
        self,
        *,
        games_by_prompt: dict[str, GameSpec],
        parser: QuestionParser,
        scorer: RewardCalculator,
        run: RunConfig,
        **kwargs: Any,
    ) -> None:
        self.games_by_prompt = games_by_prompt
        self.parser = parser
        self.scorer = scorer
        self.run_config = run
        super().__init__(reward_funcs=episode_reward, **kwargs)

    def publish(self) -> None:
        """All workers observe a backup failure, rather than leaving peers waiting."""
        errors: list[Any] = [None]
        if self.accelerator.is_main_process:
            try:
                for callback in self.callback_handler.callbacks:
                    writer = getattr(callback, "tb_writer", None)
                    if writer is not None:
                        writer.flush()
                publish_run(
                    Path(self.args.output_dir),
                    backup_dir=self.run_config.backup_dir,
                    hub_repo=self.run_config.hub_repo,
                )
            except Exception as error:
                errors[0] = f"Durable backup failed: {error}"
        broadcast_object_list(errors, from_process=0)
        if errors[0]:
            raise RuntimeError(errors[0])

    def _generate_single_turn(self, prompts: list[str]) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Return TRL's native generation tuple with its supported env_mask field."""
        root = Path(self.args.output_dir)
        rank = self.accelerator.process_index
        step = self.state.global_step
        batch_key = fingerprint(
            {
                "prompts": prompts,
                "weights": trainable_hash(self.model),
                "run": asdict(self.run_config),
                "temperature": self.temperature,
                "completion_limit": self.max_completion_length,
            }
        )
        folder = root / "batches" / f"step-{step:08d}" / f"rank-{rank}"
        cache = folder / f"{batch_key}.pt"
        output = None
        error_message = None
        started = time.perf_counter()
        try:
            if cache.exists():
                stored = torch.load(cache, map_location="cpu", weights_only=False)
                restore_rng(stored["rng_after"])
                output = stored["output"]
                append_json(
                    root / f"events-{rank}.jsonl",
                    {
                        "event": "reuse_batch",
                        "step": step,
                        "batch": str(cache.relative_to(root)),
                    },
                )
            else:
                output = self._play_batch(prompts, folder)
                folder.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".tmp")
                torch.save({"output": output, "rng_after": rng_state()}, temporary)
                temporary.replace(cache)
        except Exception as error:
            error_message = f"Worker {rank} could not complete a rollout group: {error}"
        errors = gather_object([error_message])
        if any(errors):
            append_json(
                root / f"events-{rank}.jsonl",
                {
                    "event": "incomplete_batch",
                    "step": step,
                    "errors": errors,
                },
            )
            raise RuntimeError("; ".join(str(error) for error in errors if error))
        assert output is not None
        if self.run_config.backup_rollouts:
            self.publish()
        seconds = time.perf_counter() - started
        self.log({"perf/rollout_seconds": seconds})
        return output

    def _play_batch(self, prompts: list[str], folder: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
        group_size = self.num_generations
        assert group_size is not None
        if len(prompts) % group_size:
            raise ValueError("Local generation batches must contain whole GRPO groups.")
        for offset in range(0, len(prompts), group_size):
            if len(set(prompts[offset : offset + group_size])) != 1:
                raise ValueError("Group members must have the same initial game.")
        rank = self.accelerator.process_index
        step = self.state.global_step
        groups = [f"step-{step}-rank-{rank}-group-{i // group_size}" for i in range(len(prompts))]
        rollouts = [f"{group}-sample-{i % group_size}" for i, group in enumerate(groups)]
        games = [self.games_by_prompt[prompt] for prompt in prompts]
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as model:
            policy = TransformersPolicy(model, self.processing_class, temperature=self.temperature)
            traces = play_games(
                games,
                policy,
                self.parser,
                self.processing_class,
                group_ids=groups,
                rollout_ids=rollouts,
                max_action_tokens=self.run_config.max_action_tokens,
                max_context_tokens=self.run_config.max_context_tokens,
                max_completion_tokens=self.max_completion_length,
                turn_log=folder / "turns.jsonl",
                free_form=self.run_config.free_form_questions,
            )
        rewards = [
            self.scorer.score(trace, game) for trace, game in zip(traces, games, strict=True)
        ]
        for trace, reward in zip(traces, rewards, strict=True):
            write_json(
                folder / f"{trace.rollout_id}.json",
                {
                    "policy_step": step,
                    "trajectory": trace,
                    "reward": reward,
                    "outcome_status": "complete",
                    "success": trace.success,
                    "actual_moves": len(trace.transitions),
                    "training_action_tokens": sum(trace.policy_mask),
                },
            )
        count = len(traces)
        metrics = {
            "rollout/success": sum(trace.success for trace in traces) / count,
            "rollout/moves": sum(len(trace.transitions) for trace in traces) / count,
            "rollout/action_tokens": float(sum(sum(trace.policy_mask) for trace in traces)),
            "parser/cache_hits": float(
                sum(
                    bool(cast(dict[str, Any], turn["parse"])["cache_hit"])
                    for trace in traces
                    for turn in trace.turns
                )
            ),
        }
        for name in rewards[0].components:
            metrics[f"reward_components/{name}"] = (
                sum(reward.components.get(name, 0.0) for reward in rewards) / count
            )
        self.log(metrics)
        return (
            [trace.prompt_ids for trace in traces],
            [trace.completion_ids for trace in traces],
            [trace.logprobs for trace in traces],
            {
                "env_mask": [trace.policy_mask for trace in traces],
                "episode_rewards": [reward.total for reward in rewards],
            },
        )


class RunCallback(TrainerCallback):
    """Write native metrics and publish only complete native checkpoints."""

    def __init__(self, trainer: GameGRPOTrainer) -> None:
        self.trainer = trainer
        self.started = 0.0

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        metrics = {"perf/update_seconds": time.perf_counter() - self.started}
        if self.trainer.accelerator.device.type == "cuda":
            metrics["resources/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            metrics["resources/peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        self._write_custom_metrics(args, state, control, metrics)

    def _write_custom_metrics(
        self, args: Any, state: Any, control: Any, metrics: dict[str, float]
    ) -> None:
        # Calling trainer.log here would clear the native pending loss/gradient log flag.
        self.on_log(args, state, control, logs=metrics)
        if self.trainer.accelerator.is_main_process:
            for callback in self.trainer.callback_handler.callbacks:
                writer = getattr(callback, "tb_writer", None)
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(f"train/{name}", value, state.global_step)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        rank = self.trainer.accelerator.process_index
        append_json(
            Path(args.output_dir) / f"metrics-{rank}.jsonl",
            {
                "step": state.global_step,
                "metrics": logs or {},
            },
        )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        trainer = self.trainer
        started = time.perf_counter()
        trainer.accelerator.wait_for_everyone()
        errors: list[Any] = [None]
        if trainer.accelerator.is_main_process:
            try:
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                mark_checkpoint(checkpoint)
                write_json(Path(args.output_dir) / "latest.json", {"checkpoint": checkpoint.name})
            except Exception as error:
                errors[0] = f"Checkpoint completion failed: {error}"
        broadcast_object_list(errors, from_process=0)
        if errors[0]:
            raise RuntimeError(errors[0])
        trainer.publish()
        self._write_custom_metrics(
            args,
            state,
            control,
            {
                "perf/checkpoint_publish_seconds": time.perf_counter() - started,
            },
        )
