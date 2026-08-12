"""
train_vs_asu_1v1.py
--------------------
Heads-up (1v1) self-play training: our agent vs ASUValueV1, inside the same
4-seat engine (the other two seats are forced bankrupt at game start via
run_episode(active_pids=...)).

This is "playing against ASU" (explicitly allowed) — gradients only ever
come from OUR agent's own actions and the real game outcome (win/loss,
net-worth potential). ASU's weights/code are never read as a label or
imitation target, only called as a black-box opponent via choose_action(env).

Usage:
    python tools/train_vs_asu_1v1.py --algo ppo --games 2000 --out artifacts/ppo_plus/vs_asu_1v1.pt
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASUValueV1
from monopoly_game_engine.agent_ddqn import DDQNAgent
from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.env import MonopolyEnv
from monopoly_game_engine.train import run_episode


def main():
    parser = argparse.ArgumentParser(description="1v1 self-play training vs ASU")
    parser.add_argument("--algo", choices=["ppo", "ddqn"], default="ppo")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    agent_pid = 0
    asu_pid = 1
    if args.algo == "ppo":
        agent = PPOAgent(player_id=agent_pid, hybrid=True)
    else:
        agent = DDQNAgent(player_id=agent_pid, hybrid=True)
    if args.resume and Path(args.out).exists():
        agent.load(args.out)

    asu = ASUValueV1(asu_pid)
    env = MonopolyEnv(agent_ids=[agent_pid], max_rounds=200)

    log_path = str(Path(args.out).with_suffix("")) + "_games.csv"
    is_new = not (args.resume and Path(log_path).exists())
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if is_new:
        writer.writerow(["game", "won", "reward", "steps", "decision_seconds"])

    wins_window = 0
    started_at = time.time()
    starting_games = int(getattr(agent, "games_trained", 0))

    print(f"1v1 vs ASUValueV1 — {args.algo.upper()} hybrid, {args.games} games, out={args.out}")

    for game_num in range(1, args.games + 1):
        absolute_game = starting_games + game_num
        t0 = time.time()
        result = run_episode(
            env, agent, [asu], agent_pid, args.algo == "ppo", active_pids=[agent_pid, asu_pid]
        )
        elapsed = time.time() - t0
        agent.games_trained = absolute_game

        writer.writerow(
            [absolute_game, int(result["won"]), f"{result['reward']:.4f}", result["steps"], f"{elapsed:.2f}"]
        )
        log_file.flush()

        if result["won"]:
            wins_window += 1

        if args.checkpoint_every and absolute_game % args.checkpoint_every == 0:
            agent.save(args.out)

        if game_num % args.log_every == 0:
            win_rate = wins_window / args.log_every * 100
            total_elapsed = time.time() - started_at
            print(
                f"  Game {absolute_game:5d} | Win% (last {args.log_every}): {win_rate:5.1f}% | "
                f"{elapsed:.1f}s/game (last)  avg={total_elapsed / game_num:.1f}s/game"
            )
            wins_window = 0

    agent.save(args.out)
    log_file.close()
    print(f"Done. Model saved to {args.out}")


if __name__ == "__main__":
    main()
