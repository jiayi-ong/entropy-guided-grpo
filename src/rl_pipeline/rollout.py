"""Shared game-playing logic. No optimizer, GRPO grouping math, or evaluation metrics."""

import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from contracts import GameSpec, Observation, Trajectory
from game_environment.environment import GameEnvironment, TableGameEnvironment
from integration.artifacts import append_json
from parser.base import QuestionParser
from rl_pipeline.models import PolicyBackend

TRANSCRIPT_VERSION = "plain-transcript-v1"


def public_prompt(game: GameSpec, *, free_form: bool = False) -> str:
    """Render the original table. The target and surviving set are never rendered."""
    columns = list(next(iter(game.candidates.values())))
    questions = (
        "Ask one yes/no question."
        if free_form
        else (
            "Questions: " + "; ".join(f"Q{i}=is {column} true?" for i, column in enumerate(columns))
        )
    )
    rows = "\n".join(
        f"G{i} {name} " + " ".join(f"{key}={value}" for key, value in attributes.items())
        for i, (name, attributes) in enumerate(game.candidates.items())
    )
    return (
        f"Animal deduction game {game.game_id}. You have {game.max_moves} moves.\n"
        f"{questions}\nGuess using exactly G plus the listed number. A wrong guess uses a move.\n"
        "Output exactly one action, with no explanation. Feedback will follow between actions.\n"
        f"Public candidate table:\n{rows}\nNext action:\n"
    )


def play_games(
    games: list[GameSpec],
    policy: PolicyBackend,
    parser: QuestionParser,
    tokenizer: Any,
    *,
    group_ids: list[str],
    rollout_ids: list[str],
    max_action_tokens: int,
    max_context_tokens: int,
    max_completion_tokens: int,
    turn_log: Path,
    free_form: bool = False,
    environment_factory: Callable[[], GameEnvironment] = TableGameEnvironment,
) -> list[Trajectory]:
    """Play a batch of independent games; evaluation may pass one game per group.

    Exact token prefixes grow by appending generated IDs and encoded feedback.
    No text decoding/re-encoding can silently rewrite earlier policy tokens.
    Completed history is flushed to disk; only the current batch stays in RAM.
    """
    if not (len(games) == len(group_ids) == len(rollout_ids)):
        raise ValueError("Each game needs a group and rollout ID.")
    environments = [environment_factory() for _ in games]
    observations: list[Observation] = [
        env.reset(game) for env, game in zip(environments, games, strict=True)
    ]
    traces = [
        Trajectory(
            game.game_id,
            group,
            rollout,
            prompt_ids=tokenizer.encode(
                public_prompt(game, free_form=free_form), add_special_tokens=False
            ),
        )
        for game, group, rollout in zip(games, group_ids, rollout_ids, strict=True)
    ]
    if any(len(trace.prompt_ids) >= max_context_tokens for trace in traces):
        raise ValueError("The full initial table does not fit. Increase the context limit.")
    active = list(range(len(games)))
    while active:
        ready = []
        limits = []
        for index in active:
            trace = traces[index]
            available = min(
                max_context_tokens - len(trace.prompt_ids) - len(trace.completion_ids),
                max_completion_tokens - len(trace.completion_ids),
            )
            if available < 1:
                trace.end_reason = "token_limit"
            else:
                ready.append(index)
                limits.append(available)
        if not ready:
            break
        started = time.perf_counter()
        generated = policy.generate_batch(
            [traces[i].prompt_ids + traces[i].completion_ids for i in ready],
            max_new_tokens=min(max_action_tokens, min(limits)),
        )
        generated_seconds = time.perf_counter() - started
        started = time.perf_counter()
        results = parser.parse_batch(
            [turn.text for turn in generated], [observations[i] for i in ready]
        )
        append_json(
            turn_log,
            {
                "event": "rollout_wave",
                "active_games": len(ready),
                "generation_seconds": generated_seconds,
                "parser_seconds": time.perf_counter() - started,
            },
        )
        next_active = []
        for index, turn, parsed in zip(ready, generated, results, strict=True):
            trace = traces[index]
            input_tokens = len(trace.prompt_ids) + len(trace.completion_ids)
            trace.action_texts.append(turn.text)
            trace.completion_ids.extend(turn.token_ids)
            trace.policy_mask.extend([1] * len(turn.token_ids))
            trace.logprobs.extend(turn.logprobs)
            record: dict[str, object] = {
                "game_id": trace.game_id,
                "group_id": trace.group_id,
                "rollout_id": trace.rollout_id,
                "turn_index": len(trace.action_texts),
                "generated": asdict(turn),
                "input_tokens": input_tokens,
                "parse": asdict(parsed),
            }
            if parsed.status == "failed":
                record["outcome_status"] = "incomplete_parser"
                append_json(turn_log, record)
                raise RuntimeError(parsed.message)
            step = environments[index].step(parsed)
            observations[index] = step.observation
            trace.transitions.append(step.transition)
            trace.end_reason = step.transition.end_reason
            record["transition"] = asdict(step.transition)
            append_json(turn_log, record)
            trace.turns.append(record)
            if not step.transition.done:
                feedback = tokenizer.encode(
                    f"\nEnvironment: {step.transition.feedback}. "
                    f"Moves remaining: {step.observation.moves_remaining}.\nNext action:\n",
                    add_special_tokens=False,
                )
                room = min(
                    max_context_tokens - len(trace.prompt_ids) - len(trace.completion_ids),
                    max_completion_tokens - len(trace.completion_ids),
                )
                if len(feedback) >= room:
                    trace.end_reason = "token_limit"
                else:
                    trace.completion_ids.extend(feedback)
                    trace.policy_mask.extend([0] * len(feedback))
                    trace.logprobs.extend([0.0] * len(feedback))
                    next_active.append(index)
        active = next_active
    return traces
