"""Project settings that do not duplicate the native GRPO configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    """Notebook-friendly settings. None selects the local tiny policy."""

    dataset_path: str = "tests/fixtures/animals.json"
    model_name: str | None = None
    model_revision: str = "main"
    game_count: int = 8
    game_seed: int = 123
    max_moves: int = 6
    max_action_tokens: int = 1
    max_context_tokens: int = 2048
    reward: str = "fixture"
    move_cost: float = 0.02
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_targets: tuple[str, ...] = ()
    parser_model: str | None = None
    parser_concurrency: int = 2
    parser_requests_per_minute: float = 30
    parser_max_attempts: int = 2
    free_form_questions: bool = False
    backup_dir: str | None = None
    hub_repo: str | None = None
    backup_rollouts: bool = True
    tensorboard_flush_seconds: int = 10

    def __post_init__(self) -> None:
        if min(self.game_count, self.max_moves, self.max_action_tokens, self.lora_rank) < 1:
            raise ValueError("Game counts, token limits, and LoRA rank must be positive.")
        if self.reward not in {"fixture", "task"}:
            raise ValueError("Choose fixture or task reward.")
        if self.free_form_questions and not self.parser_model:
            raise ValueError("Free-form questions require an API parser model.")
