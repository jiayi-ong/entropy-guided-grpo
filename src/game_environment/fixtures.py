"""Small reproducible manifests. The training driver consumes these like other data."""

import json
import random
from pathlib import Path

from contracts import GameSpec


def load_games(path: Path, *, seed: int, count: int, max_moves: int) -> list[GameSpec]:
    """Sample targets independently of candidate IDs and public game IDs."""
    candidates = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    return [
        GameSpec(f"game-{index:04d}", candidates, rng.choice(list(candidates)), max_moves)
        for index in range(count)
    ]
