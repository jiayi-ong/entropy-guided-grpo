"""Exercise provider errors with an in-memory HTTP transport."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from contracts import GameSpec
from game_environment.environment import TableGameEnvironment
from parser.api import ApiParserConfig, ApiQuestionParser


@pytest.mark.parametrize("first_response", ["rate_limit", "invalid_schema"])
def test_retry_records_both_attempts_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    game: GameSpec,
    first_response: str,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    responses = json.loads((Path(__file__).parent / "fixtures/parser_responses.json").read_text())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        key = first_response if len(requests) == 1 else "valid"
        code = 429 if key == "rate_limit" else 200
        return httpx.Response(
            code, json=responses[key], headers={"request-id": f"req-{len(requests)}"}
        )

    parser = ApiQuestionParser(
        ApiParserConfig(
            "chosen-model",
            str(tmp_path),
            requests_per_minute=1e9,
            retry_seconds=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    context = TableGameEnvironment().reset(game)
    parsed = parser.parse_batch(["Does it have attribute zero?"], [context])[0]
    assert parsed.status == "valid" and len(parsed.attempts) == 2
    assert parsed.attempts[1].request_id == "req-2"
    again = parser.parse_batch(["Does it have attribute zero?"], [context])[0]
    assert again.cache_hit and not again.attempts and len(requests) == 2
    assert "target_id" not in requests[0].content.decode()


def test_404_is_not_retried_or_a_policy_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    game: GameSpec,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    parser = ApiQuestionParser(
        ApiParserConfig("missing", str(tmp_path)),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(404, json={"error": {"type": "not_found_error"}})
        ),
    )
    result = parser.parse_batch(["Can it fly?"], [TableGameEnvironment().reset(game)])[0]
    assert result.status == "failed"
    assert len(result.attempts) == 1
    assert result.attempts[0].stage == "request"
    assert result.attempts[0].http_status == 404


def test_html_404_is_still_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    game: GameSpec,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    parser = ApiQuestionParser(
        ApiParserConfig("missing", str(tmp_path)),
        transport=httpx.MockTransport(lambda request: httpx.Response(404, text="not found")),
    )
    result = parser.parse_batch(["Can it fly?"], [TableGameEnvironment().reset(game)])[0]
    assert result.status == "failed" and len(result.attempts) == 1


def test_cache_changes_with_context(tmp_path: Path, game: GameSpec) -> None:
    from integration.artifacts import fingerprint

    observation = TableGameEnvironment().reset(game)
    assert fingerprint(observation) != fingerprint(replace(observation, history=("yes",)))
