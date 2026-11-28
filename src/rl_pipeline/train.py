"""Notebook and command-line entry points. GPU initialization stays inside the worker."""

import argparse
import importlib.metadata
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from integration.artifacts import (
    configure_logging,
    fingerprint,
    is_complete_checkpoint,
    latest_checkpoint,
    load_json,
    write_json,
)
from rl_pipeline.config import RunConfig


def default_trainer_options(output_dir: str) -> dict[str, Any]:
    """Explicit defaults for a short CPU run; notebooks may override native options."""
    return {
        "output_dir": output_dir,
        "max_steps": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "num_generations": 4,
        "num_iterations": 1,
        "learning_rate": 1e-3,
        "lr_scheduler_type": "constant",
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "beta": 0.0,
        "epsilon": 0.2,
        "loss_type": "grpo",
        "scale_rewards": "group",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_completion_length": 128,
        "mask_truncated_completions": False,
        "logging_steps": 1,
        "save_steps": 1,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "report_to": ["tensorboard"],
        "logging_dir": str(Path(output_dir) / "tensorboard"),
        "use_cpu": True,
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "seed": 42,
        "data_seed": 42,
        "disable_tqdm": True,
        "ddp_find_unused_parameters": False,
    }


def validate_options(options: dict[str, Any]) -> None:
    """Reject unsupported combinations rather than quietly changing the experiment."""
    group_size = options["num_generations"]
    if group_size < 2 or options["per_device_train_batch_size"] % group_size:
        raise ValueError("Per-worker microbatches must be a multiple of group size, at least two.")
    if options.get("num_iterations", 1) != 1:
        raise ValueError("The starter supports one update pass per fresh rollout batch.")
    if options.get("use_vllm", False) or options.get("use_transformers_paged", False):
        raise ValueError("This adapter uses ordinary Transformers generation.")
    if options.get("top_p", 1.0) != 1.0 or options.get("top_k", 0) not in (0, None):
        raise ValueError("Use untruncated temperature sampling so stored log probabilities agree.")
    if options.get("mask_truncated_completions", False):
        raise ValueError(
            "Whole games need their explicit policy mask, including move-limit endings."
        )
    if options.get("save_total_limit", 2) is not None and options["save_total_limit"] < 2:
        raise ValueError("Keep at least two checkpoints so an interrupted save has a fallback.")
    if options.get("save_only_model", False):
        raise ValueError("Full optimizer, scheduler, and RNG state is required for resume.")
    if options.get("fsdp") or options.get("deepspeed"):
        raise ValueError("The initial supported distributed configuration is DDP.")


def train_worker(
    run: RunConfig,
    trainer_options: dict[str, Any],
    resume_from_checkpoint: str | None = None,
) -> str:
    """Run one seed, optionally under a notebook launcher or torchrun.

    Resume requires the same experiment settings and worker count. Set resume
    to 'latest' to verify saved checksums before loading native training state.
    """
    import torch
    from accelerate import PartialState
    from accelerate.utils import set_seed
    from datasets import Dataset
    from trl import GRPOConfig

    from game_environment.fixtures import load_games
    from parser.api import ApiParserConfig, ApiQuestionParser
    from parser.deterministic import QuestionIDParser
    from reward_design.rewards import FixtureReward, TaskReward
    from rl_pipeline.models import create_tiny_assets, load_training_policy
    from rl_pipeline.rollout import TRANSCRIPT_VERSION, public_prompt
    from rl_pipeline.trainer import GameGRPOTrainer, RunCallback

    options = dict(trainer_options)
    validate_options(options)
    options["steps_per_generation"] = options["gradient_accumulation_steps"]
    state = PartialState(cpu=options.get("use_cpu", False))
    root = Path(options["output_dir"]).expanduser().resolve()
    options["output_dir"] = str(root)
    configure_logging(root, rank=state.process_index)
    # Avoid many CPU threads competing over the very small fixture model.
    if options.get("use_cpu", False):
        torch.set_num_threads(1)
    set_seed(options["seed"])
    args = GRPOConfig(**options)
    games = load_games(
        Path(run.dataset_path),
        seed=run.game_seed,
        count=run.game_count,
        max_moves=run.max_moves,
    )
    identity_options = {k: v for k, v in options.items() if k not in {"output_dir", "logging_dir"}}
    identity_run = {k: v for k, v in asdict(run).items() if k not in {"backup_dir", "hub_repo"}}
    source_root = Path(__file__).resolve().parents[1]
    source_version = fingerprint(
        {
            str(path.relative_to(source_root)): path.read_text(encoding="utf-8")
            for path in source_root.rglob("*.py")
        }
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "trl",
            "peft",
            "accelerate",
        )
    }
    identity = fingerprint(
        {
            "run": identity_run,
            "trainer": identity_options,
            "games": games,
            "world_size": state.num_processes,
            "contract": TRANSCRIPT_VERSION,
            "source_version": source_version,
            "versions": versions,
        }
    )
    metadata_path = root / "run.json"
    if metadata_path.exists():
        if not resume_from_checkpoint:
            raise ValueError("Use a new output directory or explicitly request resume.")
        if load_json(metadata_path)["identity"] != identity:
            raise ValueError("Resume settings, data, or worker count differ from the saved run.")
    elif resume_from_checkpoint:
        raise ValueError("Restore the whole run directory, including run.json, before resuming.")
    assets = root / "assets" / "tiny_base"
    if state.is_main_process and run.model_name is None and not assets.exists():
        create_tiny_assets(assets, context_length=run.max_context_tokens)
    state.wait_for_everyone()
    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
    loading_run = run
    if metadata_path.exists() and run.model_name:
        saved_revision = load_json(metadata_path)["resolved_model_revision"]
        loading_run = replace(run, model_revision=saved_revision or run.model_revision)
    model, tokenizer = load_training_policy(loading_run, assets, dtype=dtype)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if (
        metadata_path.exists()
        and load_json(metadata_path)["resolved_model_revision"] != resolved_revision
    ):
        raise ValueError(
            "The remote model revision changed. Pin the original commit before resuming."
        )
    if state.is_main_process and not metadata_path.exists():
        write_json(
            metadata_path,
            {
                "identity": identity,
                "contract_version": TRANSCRIPT_VERSION,
                "run": run,
                "trainer_options": options,
                "world_size": state.num_processes,
                "resolved_model_revision": resolved_revision,
                "reference": "initial base policy with the new RL adapter disabled",
                "source_version": source_version,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "trl",
                        "peft",
                        "accelerate",
                    )
                },
            },
        )
        write_json(root / "games.json", games)
    state.wait_for_everyone()
    parser: Any = QuestionIDParser()
    if run.parser_model:
        if run.parser_concurrency < state.num_processes:
            raise ValueError("Shared API concurrency must allow at least one request per worker.")
        parser = ApiQuestionParser(
            ApiParserConfig(
                model=run.parser_model,
                cache_dir=str(root / "parser_cache" / f"rank-{state.process_index}"),
                max_concurrency=run.parser_concurrency // state.num_processes,
                requests_per_minute=run.parser_requests_per_minute / state.num_processes,
                max_attempts=run.parser_max_attempts,
            )
        )
    mapping = {public_prompt(game, free_form=run.free_form_questions): game for game in games}
    trainer = GameGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=Dataset.from_dict({"prompt": list(mapping)}),
        games_by_prompt=mapping,
        parser=parser,
        scorer=FixtureReward() if run.reward == "fixture" else TaskReward(move_cost=run.move_cost),
        run=run,
    )
    trainer.add_callback(RunCallback(trainer))
    # Native trainer construction seeds model setup; offset only the later sampling stream.
    set_seed(options["seed"] + state.process_index)
    resume = resume_from_checkpoint
    if resume == "latest":
        checkpoint = latest_checkpoint(root)
        if checkpoint is None:
            raise ValueError("No complete checkpoint exists. Use a new run directory.")
        resume = str(checkpoint)
    if resume:
        checkpoint = Path(resume)
        if not is_complete_checkpoint(checkpoint):
            raise ValueError("Choose a complete checkpoint whose file checksums match.")
        if load_json(checkpoint / "trainer_state.json")["global_step"] >= args.max_steps:
            # A finished run must not accidentally perform one additional optimizer step.
            trainer.remove_callback(RunCallback)
            return str(root)
    if state.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        from transformers.integrations import TensorBoardCallback

        resumed_step = (
            load_json(Path(resume) / "trainer_state.json")["global_step"] if resume else 0
        )
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                callback.tb_writer = SummaryWriter(
                    log_dir=args.logging_dir,
                    flush_secs=run.tensorboard_flush_seconds,
                    purge_step=resumed_step + 1 if resume else None,
                )
    try:
        trainer.train(resume_from_checkpoint=resume)
    finally:
        for callback in trainer.callback_handler.callbacks:
            writer = getattr(callback, "tb_writer", None)
            if writer is not None:
                writer.flush()
                writer.close()
                callback.tb_writer = None
        trainer.remove_callback(RunCallback)
    trainer.save_model(str(root / "final"))
    state.wait_for_everyone()
    if state.is_main_process:
        write_json(
            root / "finished.json",
            {
                "step": trainer.state.global_step,
                "final_policy": "final",
                "identity": identity,
            },
        )
    state.wait_for_everyone()
    trainer.publish()
    # Callers in notebooks use saved files; do not return a GPU model from workers.
    return str(root)


def main() -> None:
    """Run a smoke test or a saved notebook configuration."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output-dir", default="runs/smoke")
    command.add_argument("--steps", type=int, default=2)
    command.add_argument("--resume", action="store_true")
    command.add_argument(
        "--config", type=Path, help="JSON containing run and trainer_options objects"
    )
    arguments = command.parse_args()
    if arguments.config:
        values = json.loads(arguments.config.read_text(encoding="utf-8"))
        run, options = RunConfig(**values["run"]), values["trainer_options"]
    else:
        run = RunConfig()
        options = default_trainer_options(arguments.output_dir)
        options["max_steps"] = arguments.steps
    train_worker(run, options, "latest" if arguments.resume else None)


if __name__ == "__main__":
    main()
