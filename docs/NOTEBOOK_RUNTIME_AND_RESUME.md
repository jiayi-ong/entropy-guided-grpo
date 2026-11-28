# Notebook setup, persistence, and resume

## Local notebook environment

Run the uv commands in README_STARTER.md and select the registered Python 3.12
kernel. Confirm the kernel version before importing project modules. macOS and
Windows are supported for CPU development; the GPU/DDP target is a single Linux
machine with one worker per GPU.

## Colab or Kaggle setup

1. Place this repository in the runtime, with pyproject.toml at the project root.
2. Use a Python 3.12 kernel and a clean runtime. If your hosted image uses another
   Python version, select a compatible runtime/kernel rather than installing
   incompatible wheels into it. The package intentionally targets 3.12.
3. Run the bootstrap cell below before importing Torch. It uses uv.lock as the
   source of truth and installs into the actual current kernel, not an unused
   .venv. Choose cpu when no NVIDIA GPU is available.

```python
import os
import subprocess
import sys
from pathlib import Path

assert sys.version_info[:2] == (3, 12), "Select a Python 3.12 kernel."
PROJECT = Path("/path/to/entropy-guided-grpo").resolve()
os.chdir(PROJECT)
EXTRA = "cuda"  # Use "cpu" for a CPU runtime.
subprocess.check_call([sys.executable, "-m", "pip", "install", "uv==0.13.9"])
requirements = Path("/tmp/grpo-locked-requirements.txt")
subprocess.check_call([
    sys.executable, "-m", "uv", "export", "--locked", "--extra", EXTRA,
    "--no-dev", "--no-emit-project", "--no-hashes", "-o", str(requirements),
])
wheel_index = "https://download.pytorch.org/whl/cu126" if EXTRA == "cuda" else (
    "https://download.pytorch.org/whl/cpu"
)
subprocess.check_call([
    sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable,
    "--index-url", "https://pypi.org/simple", "--extra-index-url", wheel_index,
    "--index-strategy", "unsafe-best-match", "-r", str(requirements),
])
subprocess.check_call([
    sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable,
    "--no-deps", "--editable", str(PROJECT),
])
```

4. Restart the kernel after installing. Do not repeat setup on every cell run.
   Verify the locked Torch/Transformers/TRL versions after restarting. Preinstalled
   optional vision/audio packages can conflict with a changed Torch version;
   a clean project kernel avoids those unrelated dependencies.
5. Configure persistent output before the expensive run. Open 03_train.ipynb,
   set USE_CPU=False, NUM_PROCESSES to the allocated GPU count, and choose fp16 or
   bf16 only when supported by the GPU. T4 profiling should use fp16.
6. Run configuration cells before launching. GPU model loading happens inside
   train_worker. For multiple GPUs, avoid initializing CUDA in the parent kernel
   first; restart it if you have already run interactive GPU code.

The default tiny model also runs on a GPU, but its timings are not estimates for
Qwen. Set MODEL_NAME and an exact revision, select suitable LoRA targets, and use
representative turn/context lengths before profiling. The first real run must
also validate the selected GPU/CUDA environment.

## Durable storage: implemented choices

Set at least one of the following in RunConfig. With neither, saving is local
only, which does not protect against losing the notebook VM.

| Setting | Use |
| --- | --- |
| backup_dir | A mounted persistent directory; files are copied and verified |
| hub_repo | An existing private Hugging Face model repository; upload_folder publishes run state |

For Colab, mount Google Drive through the normal notebook UI/API and choose a
dedicated backup_dir outside the local run directory. Copying many files can be
slow; checkpoint and backup time is included in profiling.

For Kaggle, hub_repo provides a concrete upload route without relying on session
memory or a final notebook save. Create a private model repository in your own
Hub account, set HF_TOKEN through Kaggle Secrets, and supply its owner/name.
The code does not create or change repository permissions. Uploads include the
private game manifest and audit traces, so use a private destination.

No keys are put in configuration files or logs. For frontier parsing, read
ANTHROPIC_API_KEY from Colab/Kaggle Secrets into the environment and set a model
ID. Shared parser concurrency and requests/minute are divided across workers;
divide the remaining account budget across independently running seeds yourself.
The starter does not enforce a provider-wide token-per-minute budget. Provider
limits still apply and are handled through bounded errors/retries.

## What is saved

| Path within a run | Content |
| --- | --- |
| run.json | Resolved configuration, source/version identity, reference identity, worker count |
| games.json | Private fixed game manifest |
| parser_cache/rank-N/ | Validated parser results; completed API calls need not be repeated |
| batches/step-S/rank-N/turns.jsonl | Flushed wave timing and per-action events, including failed attempts |
| batches/step-S/rank-N/*.json | Complete episode records, private transitions, token masks, and reward components |
| batches/step-S/rank-N/*.pt | Complete generation batch and post-generation RNG states |
| metrics-N.jsonl / events-N.jsonl / worker-N.log | Structured metrics, recovery events, readable logs |
| tensorboard/ | Native and custom scalar curves |
| checkpoint-S/ | Native model, optimizer, scheduler, RNG, trainer/sampler progress, and completion hashes |
| final/ | Inference adapter and tokenizer |

The active generation batch and optimization tensors remain in memory. Completed
history is not accumulated for the whole run. Turn JSONL is a debugging stream;
complete episode JSON is the authoritative group output. JSONL may end with an
incomplete line after abrupt termination, so an analysis reader should ignore a
trailing incomplete line rather than inventing a result.

## Resume

1. Restore the entire run directory to the same local output path when possible.
   Include run.json, games.json, assets/, checkpoints/, and cached batches.
2. For a mounted backup, copy its contents back to the local run directory. For
   a Hub backup, use the normal Hub snapshot download; copy the folder named
   after your local run directory into the intended local output path.
3. Restore the same code, uv.lock, configuration, planned max_steps, and worker
   count. Do not substitute a different model revision or seed.
4. Set RESUME=True in the training notebook, or use the CLI --resume flag.
5. The driver selects the latest checksum-verified complete checkpoint. A partial
   newer checkpoint is ignored. Native Trainer restores core training state and
   sampler progress; cached batches restore their post-generation RNG position.

If the checkpoint already meets the planned step budget, no additional update is
run. If a crash happened before final/ was exported, that completed checkpoint
itself is a valid inference adapter; select it in the evaluation notebook.

A batch is reused only if its prompts, configuration, and exact trainable policy
weights match. Cache files from other policy states remain diagnostic evidence.
An unfinished batch is regenerated; completed successful API parses stay cached.
We do not implement arbitrary mid-action continuation or off-policy replay.

Native saves happen after whole generation/update cycles, including all gradient
accumulation microbatches. Checkpoints keep at least one earlier fallback. A
completion marker is published last. Mounted copies use checksums; Hub commits
publish snapshots. TensorBoard flushes before backups, and resumed writers hide
uncommitted later-step events. Raw metrics remain available for audit.

Set save_steps for your loss tolerance. backup_rollouts=True also publishes a
completed expensive generation batch before optimization; turning it off reduces
I/O overhead but can lose the current batch on VM loss. In-flight requests or
work after the latest durable recovery point can still be lost.

## Profiling and GPU checks

Use summarize_run with a warm-up exclusion. It counts new policy tokens, sums
workers' tokens, takes the slowest worker's update timing, and includes checkpoint
publishing. Startup/final export are separate one-time costs. Inspect wave records
for generation versus API waiting. CPU/tiny/ID-parser timing cannot establish
full-model or live-parser throughput.

Before full runs, verify a two-GPU notebook launch, a GPU checkpoint interruption,
the selected backup route, API access, and representative context lengths. Exact
CPU recovery is tested automatically; bit-identical GPU recovery is not promised.

Sources: [Colab lifecycle](https://research.google.com/colaboratory/faq.html),
[native checkpointing](https://huggingface.co/docs/accelerate/usage_guides/checkpoint),
[uv PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/),
[TensorBoard writer](https://docs.pytorch.org/docs/stable/tensorboard.html).
