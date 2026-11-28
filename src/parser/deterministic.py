"""The fixture language is Q0..Q4 for questions and G0..G31 for guesses."""

import re

from contracts import Action, Observation, ParseResult, Predicate
from parser.base import QuestionParser


def parse_id(text: str, context: Observation) -> ParseResult | None:
    """Return None for free-form text so the API parser can interpret it."""
    value = text.strip()
    if re.fullmatch(r"G\d+", value):
        candidates = list(context.candidates)
        index = int(value[1:])
        if index < len(candidates):
            candidate = candidates[index]
            return ParseResult("valid", Action(guess_id=candidate))
        return ParseResult("invalid", message="The guess ID is not listed.")
    if re.fullmatch(r"Q\d+", value):
        columns = list(next(iter(context.candidates.values())))
        index = int(value[1:])
        if index < len(columns):
            attribute = columns[index]
            return ParseResult("valid", Action(predicate=Predicate(attribute, True)))
        return ParseResult("unsupported", message="The attribute is not listed.")
    return None


class QuestionIDParser(QuestionParser):
    """A fully local parser for CI. Free-form language is deliberately unsupported."""

    def parse_batch(self, texts: list[str], contexts: list[Observation]) -> list[ParseResult]:
        return [
            parse_id(text, context) or ParseResult("invalid", message="Use a Q or G ID.")
            for text, context in zip(texts, contexts, strict=True)
        ]
