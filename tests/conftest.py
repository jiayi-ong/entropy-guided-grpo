"""Small shared inputs; no model download or API key is needed."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from contracts import GameSpec
from rl_pipeline.models import create_tiny_assets


@pytest.fixture
def game() -> GameSpec:
    rows = json.loads((Path(__file__).parent / "fixtures" / "animals.json").read_text())
    return GameSpec("audit-game", rows, "animal-31", 6)


@pytest.fixture(scope="session")
def assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    path = tmp_path_factory.mktemp("tiny_base")
    create_tiny_assets(path, context_length=2048)
    return path


@pytest.fixture
def tokenizer(assets: Path) -> Any:
    return AutoTokenizer.from_pretrained(assets)
