"""Training throughput summaries. Scientific evaluation metrics remain undefined."""

import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize_run(root: Path, *, warmup_updates: int = 0) -> dict[str, float]:
    """Use the slowest worker per update and count newly generated policy tokens.

    Update timing includes generation, API waits, and rollout backups. Native
    checkpoint publishing happens afterward and is added separately. Startup
    and final export are not included in the steady-state estimate.
    """
    durations: dict[int, list[float]] = defaultdict(list)
    saves: dict[int, list[float]] = defaultdict(list)
    tokens: dict[int, list[float]] = defaultdict(list)
    peak_memory = 0.0
    for path in root.glob("metrics-*.jsonl"):
        # A resumed worker appends records. The latest record replaces an abandoned attempt.
        worker_durations: dict[int, float] = {}
        worker_saves: dict[int, float] = {}
        worker_tokens: dict[int, float] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            step, metrics = record["step"], record["metrics"]
            if "perf/update_seconds" in metrics:
                worker_durations[step] = metrics["perf/update_seconds"]
            if "perf/checkpoint_publish_seconds" in metrics:
                worker_saves[step] = metrics["perf/checkpoint_publish_seconds"]
            # Rollouts are generated at policy step s for optimizer update s + 1.
            if "rollout/action_tokens" in metrics:
                worker_tokens[step + 1] = metrics["rollout/action_tokens"]
            peak_memory = max(peak_memory, metrics.get("resources/peak_allocated_gib", 0.0))
        for step, seconds in worker_durations.items():
            durations[step].append(seconds)
        for step, seconds in worker_saves.items():
            saves[step].append(seconds)
        for step, count in worker_tokens.items():
            tokens[step].append(count)
    steps = sorted(step for step in durations if step > warmup_updates)
    if not steps:
        raise ValueError("No measured updates remain after warm-up.")
    measured_seconds = [max(durations[step]) + max(saves.get(step, [0.0])) for step in steps]
    total_tokens = sum(sum(tokens.get(step, [])) for step in steps)
    return {
        "measured_updates": float(len(steps)),
        "median_update_seconds": statistics.median(measured_seconds),
        "mean_update_seconds": statistics.mean(measured_seconds),
        "update_seconds_std": statistics.pstdev(measured_seconds),
        "measured_action_tokens": total_tokens,
        "action_tokens_per_second": total_tokens / sum(measured_seconds),
        "peak_allocated_gib": peak_memory,
    }
