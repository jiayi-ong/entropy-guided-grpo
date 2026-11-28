# Delivery validation

Checked in the delivery environment on 2026-11-28: Linux x86-64, Python 3.12.14,
Torch 2.8.0+cpu, Transformers 4.57.1, TRL 0.29.0, PEFT 0.17.1, Accelerate 1.11.0.

| Check | Result |
| --- | --- |
| Ruff lint and formatting | Passed |
| mypy for source and tests | Passed |
| Locked CPU environment installation | Passed |
| pytest | 16 tests passed |
| Native GRPO policy/environment masks | Checked actual log-probability gradients; supplied tokens have zero direct loss |
| Behavior log probabilities | Matched native recomputed policy probabilities on sampled token positions |
| Mixed-return update | Changed intended LoRA weights |
| Equal-return group | Zero finite native advantages |
| Gradient accumulation | One update consumed two whole local groups |
| Interrupted resume | Rolled back an uncheckpointed update, reused its compatible saved batch, and reproduced uninterrupted adapter weights exactly |
| Completed-run resume | Skipped additional optimization |
| Checkpoint integrity | Incomplete/corrupted checkpoints rejected |
| Mounted-directory backup | Copied and verified usable checkpoint and metadata |
| Training notebook | Code cells executed with the default CPU configuration |
| Partial evaluation notebook | Setup/loading cells executed; four requested TODO sections remain empty |
| Notebook structure/syntax | Validated in CI tests |
| API adapter | Tested through mocked HTTP; no live request or charge |
| Original checklists | Preserved from the original archive |

The CPU test suite takes a few seconds after imports on this machine. That is
not a full GitHub job duration or a GPU compute estimate.

## Checks requiring your environment

No GPU was available here. A two-worker CPU DDP check was attempted but blocked
before training by the environment's Gloo socket restrictions. Distributed GPU
execution, GPU crash recovery, live frontier-model parsing, and Hub uploads are
implemented but not claimed as validated here. Run the notebook integration
checks with your selected GPU allocation and API/storage account before a full run.

The workflow includes Windows and Apple Silicon macOS jobs, but those operating
systems and the workflow on GitHub itself were not executed in this environment.

No entropy calculation or scientific evaluation metrics are implemented. The
fixture reward and tiny vocabulary are explicitly diagnostic. Existing folder
checklists retain the larger project responsibilities and future requirements.
