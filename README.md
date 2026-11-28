# entropy-guided-grpo

An academic project investigating whether entropy-reduction rewards help a small language model ask informative questions and identify hidden animals more efficiently in Twenty Questions.

## Current state

This is a working multi-turn GRPO training slice. The local smoke run uses a real
two-layer Transformer, ordinary LoRA, a deterministic 32-animal game, an ID parser,
and an artificial fixture reward. It downloads no pretrained weights and makes
no API calls. It checks integration and gradients, not reasoning quality.

## Repository structure

The repository is organized by work area. Each folder under `src/` initially contains only a `CHECKLIST.md`; implementation files will be added as development progresses.

| Path | Main purpose |
|---|---|
| `README.md` | Project overview, approach, and current status. |
| `.gitignore` | Exclude local environments, caches, logs, and generated artifacts from Git. |
| `src/game_environment/` | Dataset preparation, game rules, candidate filtering, and game-state tracking. |
| `src/parser/` | Map natural-language questions to validated attribute tests; handle ambiguity and parsing failures. |
| `src/reward_design/` | Define terminal rewards, action costs, entropy bonuses, and reward ablations. |
| `src/baselines_sft/` | Build controller baselines, generate demonstrations, and establish the SFT starting policy. |
| `src/rl_pipeline/` | Connect the policy, parser, and environment; configure training, rollouts, checkpoints, and compute. |
| `src/experiment_design/` | Specify experiment conditions, evaluation datasets, metrics, and comparison rules. |
| `src/statistical_analysis/` | Estimate improvements, confidence intervals, sample-size needs, and statistical uncertainty. |
| `src/visualization/` | Plot learning curves, entropy reduction, performance comparisons, errors, and resource use. |
| `src/integration/` | Define shared interfaces, verify end-to-end behavior, and maintain reproducibility and basic CI. |


## First run on macOS, Windows, or Linux

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Open a
terminal at this directory. These commands work in PowerShell and Unix shells;
no manual virtual-environment activation is needed.

```sh
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
uv run --locked --extra cpu mypy
uv run --locked --extra cpu pytest
uv run --locked --extra cpu python -m rl_pipeline.train --output-dir runs/smoke
```

Use Python 3.12. Direct dependencies and the complete resolution are pinned in
pyproject.toml and uv.lock. The CPU extra uses the ordinary macOS Torch wheel on
Apple Silicon. CUDA training is intended for Linux hosts such as Colab/Kaggle;
local development does not require CUDA or MPS.

The locked native macOS training environment targets Apple Silicon. An Intel
Mac can use a Linux environment or the hosted training runtime for Torch work.

CI runs the same CPU checks on Linux, Windows, and macOS. The first dependency
installation takes longer than later cached runs. Remove extra operating systems
from the workflow if your private-repository allowance is tight.


## Notebook orchestration

```sh
uv sync --locked --extra cpu --group dev --group notebook
uv run --locked --extra cpu python -m ipykernel install --user --name entropy-guided-grpo
```

Choose that kernel in your notebook editor and open notebooks/03_train.ipynb.
For hosted notebooks, follow docs/NOTEBOOK_RUNTIME_AND_RESUME.md to install into
the active kernel and configure durable saves. Training settings are exposed in
the notebook. Notebook cells import supporting Python functions.


## What to inspect

| File | Purpose |
| --- | --- |
| src/contracts.py | Shared records and statuses; no Torch dependency |
| src/game_environment/environment.py | Environment ABC and deterministic fixture |
| src/parser/base.py, deterministic.py, api.py | Parser ABC, ID parser, one frontier API adapter |
| src/reward_design/rewards.py | Complete-game scoring interface and tiny scorers |
| src/rl_pipeline/rollout.py | Shared batched game-playing function and public prompt |
| src/rl_pipeline/models.py | Policy interface, fixture model, generation, inference loading |
| src/rl_pipeline/trainer.py | Narrow TRL generation adapter and checkpoint callbacks |
| src/rl_pipeline/train.py | Seeded notebook/CLI entry point with native GRPO options |
| src/integration/artifacts.py | Small persistence, checksum, and backup helpers |
| notebooks/04_evaluation.ipynb | Loading setup and deliberately unfinished evaluation sections |

Read docs/CONTRACTS.md before changing shared behavior. The remaining developers'
folders stay checklist-led. No scientific evaluation or statistical analysis is
implemented. There is no SFT trainer, environment service, or registration system.


## Outputs and limits

Runs contain metadata, a private game manifest, turn events, complete trajectories,
recoverable token batches, metrics, TensorBoard events, native checkpoints, and a
final adapter. See docs/NOTEBOOK_RUNTIME_AND_RESUME.md.

```sh
uv run --locked --extra cpu tensorboard --logdir runs
uv run --locked --extra cpu python -m rl_pipeline.train --output-dir runs/smoke --resume
```

Resume uses the same planned total steps. A completed two-step run has no further
work; use a longer planned run to test interruption. Resume refuses incompatible
settings, data, code, dependencies, or worker count. Use a new directory for a
different experiment.

The information reward is initially zero. FixtureReward deliberately provides
action-dependent variation even for invalid output; never report it as an entropy
experiment. TaskReward implements the success/full-failure-charge objective.
All failed games can have identical returns, which TRL logs explicitly.

DDP and gradient accumulation are supported. The initial implementation allows
one optimization pass per fresh generation batch, temperature sampling without
top-k/top-p truncation, and ordinary Transformers generation. These limits keep
recovery and log-probability alignment inspectable. Unsupported settings raise
validation errors instead of being silently ignored.

The tiny vocabulary and model are fixtures. Configure a real initialization,
appropriate LoRA targets, longer action/context limits, representative data, and
an API parser when needed before estimating full compute. A merged SFT checkpoint
can be the initialization; disabling the new RL adapter gives its fixed reference.

Read docs/VALIDATION.md for checks actually run and the remaining GPU/API checks.