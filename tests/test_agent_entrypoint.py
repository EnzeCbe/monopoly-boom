"""
Edge-case tests for the submission entry point (agent.py).

Covers what we learned matters most today, from studying other teams'
public commit histories (not ASU): calling-convention ambiguity (different
argument orders/names a harness might use), seat aliasing, and sys.path
safety. Own tests, own reasoning — no code copied.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent, choose_action  # noqa: E402
from monopoly_game_engine.actions import ActionType  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402


def _fresh_env():
    env = MonopolyEnv(agent_ids=[0])
    state = env.reset()
    return env, state


class TestCallingConventions(unittest.TestCase):
    """A harness might call choose_action with the arguments in any order,
    as keywords, or with some omitted. Every shape must return a legal
    action, never raise, and never silently return something illegal."""

    def test_positional_state_allowed_env(self):
        env, state = _fresh_env()
        allowed = env.get_allowed_actions(0)
        action = Agent(0).choose_action(state, allowed, env)
        self.assertIn(action, allowed)

    def test_positional_state_env_allowed(self):
        env, state = _fresh_env()
        allowed = env.get_allowed_actions(0)
        action = Agent(0).choose_action(state, env, allowed)
        self.assertIn(action, allowed)

    def test_keyword_only_any_order(self):
        env, state = _fresh_env()
        allowed = env.get_allowed_actions(0)
        action = Agent(0).choose_action(env=env, state=state, allowed_actions=allowed)
        self.assertIn(action, allowed)

    def test_env_and_allowed_only_no_explicit_state(self):
        env, state = _fresh_env()
        allowed = env.get_allowed_actions(0)
        action = Agent(0).choose_action(env, allowed)
        self.assertIn(action, allowed)

    def test_single_legal_action_short_circuits(self):
        env, state = _fresh_env()
        action = Agent(0).choose_action(state, [int(ActionType.DO_NOTHING)], env)
        self.assertEqual(action, int(ActionType.DO_NOTHING))

    def test_module_level_function_matches_class(self):
        env, state = _fresh_env()
        allowed = env.get_allowed_actions(0)
        action = choose_action(state, allowed, env, 0)
        self.assertIn(action, allowed)


class TestSeatAliases(unittest.TestCase):
    """Different spec drafts have used player_id / pid / agent_id / seat."""

    def test_constructor_aliases(self):
        for key in ("player_id", "pid", "agent_id", "seat"):
            agent = Agent(**{key: 0})
            self.assertEqual(agent.player_id, 0, key)

    def test_module_function_seat_aliases(self):
        env, state = _fresh_env()
        allowed = env.get_allowed_actions(0)
        for key in ("player_id", "pid", "seat"):
            action = choose_action(state, allowed, env, **{key: 0})
            self.assertIn(action, allowed, key)


class TestFullGameNoIllegalActions(unittest.TestCase):
    """The real end-to-end check: a whole game, mixing calling conventions
    turn to turn, must never produce an action outside what the engine
    actually offered."""

    def test_alternating_conventions_full_game(self):
        env, state = _fresh_env()
        agent = Agent(0)
        from monopoly_game_engine.agents_fixed import TheBuilder, TheDealMaker, TheHoarder
        fp = {1: TheBuilder(1), 2: TheDealMaker(2), 3: TheHoarder(3)}
        steps = 0
        toggle = 0
        while not env.done and steps < 3000:
            steps += 1
            pid = env.whose_turn()
            if env.players[pid].bankrupt:
                env._advance_turn()
                continue
            allowed = env.get_allowed_actions(pid)
            if not allowed:
                allowed = [int(ActionType.DO_NOTHING)]
            if pid == 0:
                toggle += 1
                if toggle % 2 == 0:
                    action = agent.choose_action(env._get_state(0), allowed, env)
                else:
                    action = agent.choose_action(env, allowed)
            else:
                action = fp[pid].choose_action(env)
                if action not in allowed:
                    action = allowed[0]
            self.assertIn(action, allowed, f"step {steps}, seat {pid}")
            env.step(action)


if __name__ == "__main__":
    unittest.main()
