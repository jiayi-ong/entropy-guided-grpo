"""One small frontier API adapter: Anthropic Messages with local validation.

Other providers should implement QuestionParser and return the same records.
Keep provider request/response details out of the game and trainer.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from contracts import Action, Observation, ParseAttempt, ParseResult, Predicate
from integration.artifacts import fingerprint, load_json, write_json
from parser.base import QuestionParser
from parser.deterministic import parse_id

PARSER_PROMPT_VERSION = "binary-parser-v1"
SYSTEM_PROMPT = """Interpret one yes/no animal question using only the supplied public table.
Return only JSON: {"status": "valid", "predicate": {"attribute": "attribute_0",
"operator": "eq", "value": true, "negated": false}}.
The current simulator supports equality on Boolean attributes only. For concepts
outside that schema use {"status": "unsupported"}. For unclear meaning use
{"status": "ambiguous"}. For a question that is not yes/no use {"status": "invalid"}.
Do not answer the question or choose an animal. Do not invent attributes."""


@dataclass(frozen=True)
class ApiParserConfig:
    """Limits apply per worker. Divide the shared account budget across workers."""

    model: str
    cache_dir: str
    max_concurrency: int = 2
    requests_per_minute: float = 30
    timeout_seconds: float = 30
    max_attempts: int = 2
    retry_seconds: float = 1
    max_retry_seconds: float = 60
    endpoint: str = "https://api.anthropic.com/v1/messages"
    api_key_env: str = "ANTHROPIC_API_KEY"

    def __post_init__(self) -> None:
        if not self.model or min(self.max_concurrency, self.max_attempts) < 1:
            raise ValueError("Set a model ID and positive concurrency/attempt limits.")
        if min(self.requests_per_minute, self.timeout_seconds) <= 0:
            raise ValueError("Request rate and timeout must be positive.")


def validate_payload(payload: object, context: Observation) -> ParseResult:
    """Validate model output. Unsupported input differs from malformed output."""
    if not isinstance(payload, dict):
        raise ValueError("The parser response must be a JSON object.")
    status = payload.get("status")
    if status in {"unsupported", "ambiguous", "invalid"}:
        return ParseResult(status, message="The API parser could not execute this question.")
    if status != "valid" or not isinstance(payload.get("predicate"), dict):
        raise ValueError("Missing status or predicate.")
    item = payload["predicate"]
    attribute = item.get("attribute")
    if not isinstance(attribute, str) or any(
        attribute not in row for row in context.candidates.values()
    ):
        raise ValueError("The parser invented an attribute.")
    if item.get("operator") != "eq" or type(item.get("value")) is not bool:
        raise ValueError("The fixture accepts Boolean equality only.")
    if type(item.get("negated")) is not bool:
        raise ValueError("Negation must be an explicit Boolean.")
    return ParseResult(
        "valid", Action(predicate=Predicate(attribute, item["value"], "eq", item["negated"]))
    )


class ApiQuestionParser(QuestionParser):
    """Concurrent HTTP parsing with bounded retries and a persistent semantic cache."""

    def __init__(
        self, config: ApiParserConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.config = config
        self.transport = transport
        self._rate_lock = threading.Lock()
        self._next_request = 0.0

    def _wait_for_rate(self) -> None:
        with self._rate_lock:
            delay = max(0.0, self._next_request - time.monotonic())
            self._next_request = max(time.monotonic(), self._next_request) + (
                60 / self.config.requests_per_minute
            )
        if delay:
            time.sleep(delay)

    def parse_batch(self, texts: list[str], contexts: list[Observation]) -> list[ParseResult]:
        pairs = list(zip(texts, contexts, strict=True))
        with httpx.Client(timeout=self.config.timeout_seconds, transport=self.transport) as client:
            with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
                return list(executor.map(lambda pair: self._parse_one(client, *pair), pairs))

    def _parse_one(self, client: httpx.Client, text: str, context: Observation) -> ParseResult:
        direct = parse_id(text, context)
        if direct is not None:
            return direct
        key = fingerprint(
            {
                "text": text,
                "context": context,
                "model": self.config.model,
                "prompt": PARSER_PROMPT_VERSION,
                "endpoint": self.config.endpoint,
            }
        )
        cache = Path(self.config.cache_dir) / f"{key}.json"
        if cache.exists():
            result = validate_payload(load_json(cache), context)
            return replace(result, cache_hit=True)
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            return ParseResult("failed", message=f"Set {self.config.api_key_env} for API parsing.")
        attempts: list[ParseAttempt] = []
        for number in range(1, self.config.max_attempts + 1):
            self._wait_for_rate()
            started = time.monotonic()
            status = None
            request_id = None
            stage = "transport"
            code = None
            retry = False
            delay = self.config.retry_seconds * 2 ** (number - 1)
            input_tokens = output_tokens = 0
            try:
                response = client.post(
                    self.config.endpoint,
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    json={
                        "model": self.config.model,
                        "max_tokens": 192,
                        "temperature": 0,
                        "system": SYSTEM_PROMPT,
                        "messages": [
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "question": text,
                                        "public_context": {
                                            "candidates": context.candidates,
                                            "history": context.history,
                                        },
                                        "repair_schema": number > 1,
                                    }
                                ),
                            }
                        ],
                    },
                )
                status = response.status_code
                request_id = response.headers.get("request-id")
                if response.is_error:
                    stage = "service" if status >= 500 or status == 429 else "request"
                    try:
                        body = response.json()
                    except ValueError:
                        body = {}
                    code = str(body.get("error", {}).get("type", f"http_{status}"))
                    retry = status >= 500 or status == 429
                    if any(
                        term in response.text.lower() for term in ("spend cap", "credit balance")
                    ):
                        retry = False
                    try:
                        delay = max(delay, float(response.headers.get("retry-after", "0")))
                    except ValueError:
                        pass
                    if delay > self.config.max_retry_seconds:
                        retry = False
                else:
                    stage = "output_validation"
                    body = response.json()
                    usage = body.get("usage", {})
                    input_tokens = int(usage.get("input_tokens", 0))
                    output_tokens = int(usage.get("output_tokens", 0))
                    content = "".join(item.get("text", "") for item in body["content"])
                    payload = json.loads(content)
                    result = validate_payload(payload, context)
                    attempts.append(
                        ParseAttempt(
                            number,
                            time.monotonic() - started,
                            "success",
                            None,
                            status,
                            request_id,
                            input_tokens,
                            output_tokens,
                        )
                    )
                    # A cache file is complete before another process may read it.
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with self._rate_lock:
                        write_json(cache, payload)
                    return replace(result, attempts=tuple(attempts))
            except httpx.TransportError as error:
                code, retry = type(error).__name__, True
            except (ValueError, KeyError, TypeError) as error:
                code = (
                    "invalid_json"
                    if isinstance(error, json.JSONDecodeError)
                    else "schema_validation_error"
                )
                retry = True
            attempts.append(
                ParseAttempt(
                    number,
                    time.monotonic() - started,
                    stage,
                    code,
                    status,
                    request_id,
                    input_tokens,
                    output_tokens,
                )
            )
            if not retry or number == self.config.max_attempts:
                break
            time.sleep(delay)
        return ParseResult(
            "failed", message=f"Parser operation failed: {code}.", attempts=tuple(attempts)
        )
