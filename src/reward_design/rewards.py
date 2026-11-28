"""Complete-game scorers. The information term is intentionally zero for now."""

from abc import ABC, abstractmethod

from contracts import GameSpec, RewardBreakdown, Trajectory


class RewardCalculator(ABC):
    """Consume authoritative traces without changing environment state."""

    @abstractmethod
    def score(self, trajectory: Trajectory, game: GameSpec) -> RewardBreakdown:
        """Return named components for one complete game."""


class TaskReward(RewardCalculator):
    """The shared success/move-cost objective with an empty information term."""

    def __init__(self, *, move_cost: float = 0.02) -> None:
        self.move_cost = move_cost

    def score(self, trajectory: Trajectory, game: GameSpec) -> RewardBreakdown:
        charged = len(trajectory.transitions) if trajectory.success else game.max_moves
        return RewardBreakdown(
            {
                "success": float(trajectory.success),
                "move_cost": -self.move_cost * charged,
                "information": 0.0,
            }
        )


class FixtureReward(RewardCalculator):
    """Artificial action-dependent variation for testing an actual GRPO update.

    This rewards a reproducible property of generated text, including invalid
    actions. It has no scientific meaning and must not label a research arm.
    """

    def score(self, trajectory: Trajectory, game: GameSpec) -> RewardBreakdown:
        text = " ".join(trajectory.action_texts)
        return RewardBreakdown({"fixture_only": (sum(map(ord, text)) % 101) / 100.0})
