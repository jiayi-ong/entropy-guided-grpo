# entropy-guided-grpo

An academic project investigating whether entropy-reduction rewards help a small language model ask informative questions and identify hidden animals more efficiently in Twenty Questions.

## Planned approach

- Build a deterministic game environment using UCI Zoo animal attributes.
- Use a frozen parser to map natural-language questions to attribute tests.
- Warm-start the policy with supervised fine-tuning, then compare GRPO with and without an entropy bonus, giving both conditions an incentive to finish sooner.
- Evaluate identification success, failure-penalized moves, learning efficiency, and compute cost using controlled baselines, ablations, and multiple training seeds.

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