"""Records shared by the game, parser, scorer, and rollout driver.

Keep these records independent of Torch and trainer libraries. Private game
fields belong in traces and files, never in policy or parser prompts.
"""

from dataclasses import dataclass, field
from typing import Literal

type Scalar = bool | int | float | str
type ParseStatus = Literal["valid", "unsupported", "ambiguous", "invalid", "failed"]


@dataclass(frozen=True)
class GameSpec:
    """A fixed game. The target is private environment input."""

    game_id: str
    candidates: dict[str, dict[str, Scalar]]
    target_id: str
    max_moves: int = 6


@dataclass(frozen=True)
class Observation:
    """Only information that the player and parser may see."""

    game_id: str
    candidates: dict[str, dict[str, Scalar]]
    history: tuple[str, ...]
    moves_remaining: int


@dataclass(frozen=True)
class Predicate:
    """A yes/no condition. Only binary equality is implemented initially."""

    attribute: str
    value: Scalar
    operator: str = "eq"
    negated: bool = False


@dataclass(frozen=True)
class Action:
    """Exactly one of a question predicate or candidate-ID guess."""

    predicate: Predicate | None = None
    guess_id: str | None = None

    def __post_init__(self) -> None:
        if (self.predicate is None) == (self.guess_id is None):
            raise ValueError("An action needs exactly one predicate or guess ID.")


@dataclass(frozen=True)
class ParseAttempt:
    """One API attempt, including unsuccessful and repaired requests."""

    number: int
    seconds: float
    stage: str
    error_code: str | None = None
    http_status: int | None = None
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ParseResult:
    """Separate question interpretation from failure of the parser itself.

    A failed parser operation must not be turned into a losing game. Attempts
    stay attached to the final result so consumers can count either requests
    or player actions without confusing the two.
    """

    status: ParseStatus
    action: Action | None = None
    message: str = ""
    attempts: tuple[ParseAttempt, ...] = ()
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if (self.status == "valid") != (self.action is not None):
            raise ValueError("Only valid parse results carry an executable action.")


@dataclass(frozen=True)
class Transition:
    """Authoritative private evidence from one environment step."""

    turn_index: int
    feedback: str
    surviving_before: int
    surviving_after: int
    information_eligible: bool
    done: bool
    success: bool
    end_reason: str | None


@dataclass(frozen=True)
class StepResult:
    """Public observation and private transition travel separately."""

    observation: Observation
    transition: Transition


@dataclass(frozen=True)
class GeneratedTurn:
    """Exact newly sampled policy tokens and their sampling log probabilities."""

    text: str
    token_ids: list[int]
    logprobs: list[float]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs) or not self.token_ids:
            raise ValueError("Each generated token needs one log probability.")


@dataclass
class Trajectory:
    """One complete game, retained only while its current batch is in use."""

    game_id: str
    group_id: str
    rollout_id: str
    turns: list[dict[str, object]] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    action_texts: list[str] = field(default_factory=list)
    end_reason: str | None = None
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    policy_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """A game succeeds only after the environment confirms a guess."""
        return bool(self.transitions and self.transitions[-1].success)


@dataclass(frozen=True)
class RewardBreakdown:
    """Named scalar terms. Their sum is the complete-game return."""

    components: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.components.values())
