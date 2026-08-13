"""
Submission entry point.

Wraps our hybrid PPO agent (see ALGORITHM.md for the 9 hand-designed
heuristics + network split). Takes env because most of the heuristics need
the board state, not just the 300-float vector.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from monopoly_game_engine.agent_ppo import PPOAgent  # noqa: E402

_MODEL_PATH = _ROOT / "artifacts" / "submission" / "model.pt"


class Agent:
    def __init__(self, player_id):
        self.player_id = player_id
        self._agent = PPOAgent(player_id=player_id, hybrid=True)
        if _MODEL_PATH.exists():
            self._agent.load(str(_MODEL_PATH))
        if hasattr(self._agent, "epsilon"):
            self._agent.epsilon = 0.0

    def choose_action(self, state, allowed_actions, env):
        action, _log_prob, _value, _nn_allowed = self._agent.choose_action(
            state, env, list(allowed_actions)
        )
        return action
