"""A deterministic table game behind the shared environment interface."""

from abc import ABC, abstractmethod

from contracts import GameSpec, Observation, ParseResult, StepResult, Transition


class GameEnvironment(ABC):
    """Own game state, answers, and candidate filtering. Do not score rewards."""

    @abstractmethod
    def reset(self, game: GameSpec) -> Observation:
        """Start one fixed game and return public information."""

    @abstractmethod
    def step(self, result: ParseResult) -> StepResult:
        """Apply one interpreted policy action, including invalid actions."""


class TableGameEnvironment(GameEnvironment):
    """Minimal fixture implementation. Extend predicate execution here for Zoo."""

    def reset(self, game: GameSpec) -> Observation:
        if game.target_id not in game.candidates or game.max_moves < 1:
            raise ValueError("The game needs a listed target and a positive move limit.")
        self.game = game
        self.remaining = set(game.candidates)
        self.history: list[str] = []
        self.asked: set[tuple[str, object, str, bool]] = set()
        self.done = False
        return self._observation()

    def _observation(self) -> Observation:
        return Observation(
            self.game.game_id,
            self.game.candidates,
            tuple(self.history),
            self.game.max_moves - len(self.history),
        )

    def step(self, result: ParseResult) -> StepResult:
        if self.done:
            raise RuntimeError("Reset before stepping a finished game.")
        if result.status == "failed":
            raise RuntimeError("A failed parser operation is not a game action.")
        before = len(self.remaining)
        eligible = False
        success = False
        feedback = {"unsupported": "unknown", "ambiguous": "rephrase", "invalid": "invalid"}.get(
            result.status, "invalid"
        )
        if result.action is not None:
            action = result.action
            if action.guess_id is not None:
                if action.guess_id not in self.game.candidates:
                    feedback = "invalid"
                else:
                    success = action.guess_id == self.game.target_id
                    feedback = "correct" if success else "incorrect"
                    if not success:
                        self.remaining.discard(action.guess_id)
            else:
                predicate = action.predicate
                assert predicate is not None
                rows = self.game.candidates
                if predicate.operator != "eq" or any(
                    predicate.attribute not in row for row in rows.values()
                ):
                    feedback = "unknown"
                else:
                    key = (
                        predicate.attribute,
                        predicate.value,
                        predicate.operator,
                        predicate.negated,
                    )
                    answer = (rows[self.game.target_id][predicate.attribute] == predicate.value) ^ (
                        predicate.negated
                    )
                    feedback = "yes" if answer else "no"
                    eligible = key not in self.asked
                    self.asked.add(key)
                    self.remaining = {
                        candidate
                        for candidate in self.remaining
                        if (
                            (rows[candidate][predicate.attribute] == predicate.value)
                            ^ predicate.negated
                        )
                        == answer
                    }
        # The parser needs the meaning of earlier questions, not just yes/no answers.
        self.history.append(f"{result.action!r}: {feedback}")
        self.done = success or len(self.history) == self.game.max_moves
        reason = "correct_guess" if success else "move_limit" if self.done else None
        if self.game.target_id not in self.remaining:
            raise RuntimeError("Candidate filtering removed the true target.")
        transition = Transition(
            len(self.history),
            feedback,
            before,
            len(self.remaining),
            eligible,
            self.done,
            success,
            reason,
        )
        return StepResult(self._observation(), transition)
