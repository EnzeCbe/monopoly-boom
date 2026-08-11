"""
scout_vs_asu.py
----------------
Eval-only scouting: play our trained agent against ASU_FROZEN_TEACHER.

This is "playing against ASU" (allowed), never training/imitating it —
run_episode() is called with update_online=False, so no gradients ever
flow from ASU's play. We only *observe* its chosen actions (public API,
choose_action(env) -> int) to look for exploitable tendencies.

Usage:
    python tools/scout_vs_asu.py --algo ddqn --model artifacts/.../model.pt --games 200
    python tools/scout_vs_asu.py --algo ddqn --model model.pt --asu-seats 3 --asu-variant rollout
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1
from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.agent_ddqn import DDQNAgent
from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.agents_fixed import TheBuilder, TheDealMaker, TheHoarder
from monopoly_game_engine.constants import NUM_PLAYERS
from monopoly_game_engine.env import MonopolyEnv
from monopoly_game_engine.train import run_episode


def _action_bucket(action: int) -> str:
    if action == int(ActionType.BUY_PROPERTY):
        return "buy"
    if action == int(ActionType.ACCEPT_TRADE):
        return "trade_accept"
    if action == int(ActionType.DECLINE_TRADE):
        return "trade_decline"
    if OFFSETS["buy_trade"] <= action < OFFSETS["sell_trade"] + 252:
        return "trade_offer"
    if OFFSETS["improve_house"] <= action < OFFSETS["improve_hotel"]:
        return "build_house"
    if OFFSETS["improve_hotel"] <= action < OFFSETS["sell_house"]:
        return "build_hotel"
    if OFFSETS["mortgage"] <= action < OFFSETS["unmortgage"]:
        return "mortgage"
    if action == int(ActionType.DECLARE_BANKRUPT):
        return "bankrupt"
    return "other"


class TalliedAgent:
    """Wraps a fixed/ASU policy and counts its action-category choices.

    Pure observation — never alters the wrapped agent's decisions.
    """

    def __init__(self, inner, player_id: int):
        self.inner = inner
        self.player_id = player_id
        self.tally = {}

    def choose_action(self, env) -> int:
        action = self.inner.choose_action(env)
        bucket = _action_bucket(action)
        self.tally[bucket] = self.tally.get(bucket, 0) + 1
        return action


def build_asu(variant: str, player_id: int):
    cls = ASURolloutV1 if variant == "rollout" else ASUValueV1
    return cls(player_id)


def main():
    parser = argparse.ArgumentParser(description="Scout our agent against ASU_FROZEN_TEACHER")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--algo", choices=["ppo", "ddqn"], default="ddqn")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--asu-seats", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--asu-variant", choices=["value", "rollout"], default="value")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    agent_pid = 0
    if args.algo == "ppo":
        agent = PPOAgent(player_id=agent_pid, hybrid=True)
    else:
        agent = DDQNAgent(player_id=agent_pid, hybrid=True)
    agent.load(args.model)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    other_pids = [i for i in range(NUM_PLAYERS) if i != agent_pid]
    filler_classes = [TheBuilder, TheDealMaker, TheHoarder]

    fp_agents = []
    tallied_asu = []
    for i, pid in enumerate(other_pids):
        if i < args.asu_seats:
            wrapped = TalliedAgent(build_asu(args.asu_variant, pid), pid)
            fp_agents.append(wrapped)
            tallied_asu.append(wrapped)
        else:
            fp_agents.append(filler_classes[i](pid))

    out_path = args.out or str(
        Path(args.model).with_suffix("")
    ) + f"_vs_asu_{args.asu_variant}_{args.asu_seats}seat.csv"
    out_file = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(out_file)
    writer.writerow(
        ["game", "won", "reward", "steps", "properties_acquired",
         "trades_initiated", "trades_accepted", "trades_declined"]
    )

    env = MonopolyEnv(agent_ids=[agent_pid], max_rounds=200)
    wins = 0
    print(f"\nScouting vs ASU_{args.asu_variant} x{args.asu_seats} seat(s), "
          f"{NUM_PLAYERS - 1 - args.asu_seats} filler seat(s), {args.games} games")
    print(f"Game log: {out_path}\n")

    for game_num in range(1, args.games + 1):
        result = run_episode(
            env, agent, fp_agents, agent_pid, args.algo == "ppo", update_online=False
        )
        if result["won"]:
            wins += 1
        writer.writerow(
            [
                game_num,
                int(result["won"]),
                f"{result['reward']:.4f}",
                result["steps"],
                result["properties_acquired"],
                result["trades_initiated"],
                result["trades_accepted"],
                result["trades_declined"],
            ]
        )
        if game_num % max(1, args.games // 20) == 0:
            print(f"  Game {game_num:4d} | cumulative win%: {wins / game_num * 100:5.1f}%")

    out_file.close()

    print(f"\nFinal win rate vs this table: {wins / args.games * 100:.1f}%  "
          f"({wins}/{args.games})")

    for wrapped in tallied_asu:
        total = sum(wrapped.tally.values()) or 1
        print(f"\nASU (seat {wrapped.player_id}) action distribution over {total} decisions:")
        for bucket, count in sorted(wrapped.tally.items(), key=lambda kv: -kv[1]):
            print(f"    {bucket:15s} {count:6d}  ({count / total * 100:5.1f}%)")


if __name__ == "__main__":
    main()
