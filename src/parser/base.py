"""One batch interface for deterministic and API parsers."""

from abc import ABC, abstractmethod

from contracts import Observation, ParseResult


class QuestionParser(ABC):
    """Interpret only public information. Return results in input order."""

    @abstractmethod
    def parse_batch(self, texts: list[str], contexts: list[Observation]) -> list[ParseResult]:
        """Return one result per action; internal retries are not player moves."""
