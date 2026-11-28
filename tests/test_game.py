"""Check the game rules and the public/private boundary with exact traces."""

from contracts import Action, GameSpec, ParseResult, Trajectory
from game_environment.environment import TableGameEnvironment
from parser.deterministic import QuestionIDParser
from reward_design.rewards import TaskReward
from rl_pipeline.rollout import public_prompt


def test_five_questions_then_explicit_guess(game: GameSpec) -> None:
    env = TableGameEnvironment()
    observation = env.reset(game)
    parser = QuestionIDParser()
    counts = []
    trace = Trajectory(game.game_id, "group", "rollout")
    for action in ["Q0", "Q1", "Q2", "Q3", "Q4", "G31"]:
        parsed = parser.parse_batch([action], [observation])[0]
        result = env.step(parsed)
        observation = result.observation
        trace.transitions.append(result.transition)
        counts.append(result.transition.surviving_after)
    assert counts == [16, 8, 4, 2, 1, 1]
    assert not trace.transitions[-2].done
    assert trace.success
    assert TaskReward().score(trace, game).total == 1 - 0.02 * 6
    assert not hasattr(observation, "target_id")
    assert not hasattr(observation, "remaining")


def test_wrong_guess_repeat_and_unsupported_consume_moves(game: GameSpec) -> None:
    env = TableGameEnvironment()
    observation = env.reset(game)
    parser = QuestionIDParser()
    for number, action in enumerate(["G0", "Q0", "Q0", "Q99", "bad", "bad"], start=1):
        result = env.step(parser.parse_batch([action], [observation])[0])
        observation = result.observation
        assert result.transition.turn_index == number
        if number in {1, 3, 4, 5, 6}:
            assert not result.transition.information_eligible
    assert result.transition.done and not result.transition.success
    assert result.transition.end_reason == "move_limit"


def test_failure_charge_does_not_reward_abandonment(game: GameSpec) -> None:
    trace = Trajectory(game.game_id, "group", "rollout", end_reason="token_limit")
    assert TaskReward().score(trace, game).total == -0.02 * game.max_moves


def test_hidden_target_does_not_change_public_prompt(game: GameSpec) -> None:
    alternative = GameSpec(game.game_id, game.candidates, "animal-00", game.max_moves)
    assert public_prompt(game) == public_prompt(alternative)


def test_failed_parser_is_not_a_game_failure(game: GameSpec) -> None:
    import pytest

    env = TableGameEnvironment()
    env.reset(game)
    with pytest.raises(RuntimeError, match="not a game action"):
        env.step(ParseResult("failed"))
    assert env.step(ParseResult("valid", Action(guess_id="animal-31"))).transition.turn_index == 1
