"""Small file helpers for logs, recoverable batches, and durable copies."""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_default(value: object) -> object:
    """Convert shared records and paths without hiding unsupported objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot write {type(value).__name__} to JSON.")


def fingerprint(value: object) -> str:
    """Use stable JSON for cache and configuration identifiers."""
    data = json.dumps(value, default=json_default, sort_keys=True, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Publish one whole JSON file; readers ignore its temporary predecessor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, default=json_default, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path: Path, value: object) -> None:
    """Append and flush one event. Each worker owns its own file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, default=json_default, allow_nan=False) + "\n")
        stream.flush()


def file_hash(path: Path) -> str:
    """Read large checkpoint files in chunks."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mark_checkpoint(path: Path) -> None:
    """Publish this marker last, after every worker has finished saving."""
    files = {
        str(item.relative_to(path)): file_hash(item)
        for item in path.rglob("*")
        if item.is_file() and item.name != "complete.json" and not item.name.endswith(".tmp")
    }
    write_json(path / "complete.json", files)


def is_complete_checkpoint(path: Path) -> bool:
    """Verify a checkpoint, including when the caller selects an older one explicitly."""
    marker = path / "complete.json"
    if not marker.exists():
        return False
    try:
        expected = json.loads(marker.read_text(encoding="utf-8"))
        return bool(expected) and all(
            (path / name).is_file() and file_hash(path / name) == digest
            for name, digest in expected.items()
        )
    except (ValueError, OSError):
        return False


def latest_checkpoint(root: Path) -> Path | None:
    """Find the newest fully written, checksum-verified native checkpoint."""
    paths = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    for path in reversed(paths):
        if is_complete_checkpoint(path):
            return path
    return None


def publish_run(root: Path, *, backup_dir: str | None, hub_repo: str | None) -> None:
    """Copy to a mounted directory and/or an existing private Hub model repo.

    The Hub route is useful on Kaggle. HF_TOKEN is read by the Hub library.
    Never replace an earlier complete checkpoint with a partly copied folder.
    """
    if backup_dir:
        destination = Path(backup_dir).expanduser().resolve()
        if destination == root.resolve() or destination.is_relative_to(root.resolve()):
            raise ValueError("The durable directory must be outside the local run directory.")
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(root.rglob("*"), key=lambda p: p.name == "complete.json"):
            if not source.is_file() or source.name.endswith(".tmp"):
                continue
            relative = source.relative_to(root)
            if (
                relative.parts[0].startswith("checkpoint-")
                and not (root / relative.parts[0] / "complete.json").exists()
            ):
                continue
            target = destination / relative
            if target.exists() and file_hash(source) == file_hash(target):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            if file_hash(temporary) != file_hash(source):
                raise OSError(f"Backup verification failed for {relative}.")
            temporary.replace(target)
    if hub_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=hub_repo,
            folder_path=str(root),
            path_in_repo=root.name,
            ignore_patterns=["*.tmp"],
            commit_message="Save training recovery state",
        )


def configure_logging(root: Path, *, rank: int) -> None:
    """Keep human-readable logs separate from structured events."""
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(root / f"worker-{rank}.log")],
        force=True,
    )


def load_json(path: Path) -> Any:
    """Read persisted JSON at the untyped serialization boundary."""
    return json.loads(path.read_text(encoding="utf-8"))
