"""
train_vs_3asu.py
-----------------
Trains directly against 3x ASUValueV1 — the exact table used for the "38%
winrate with RL" competitor benchmark, and matching our reference CHAMPION.pt
logs (asu_3seats), where CHAMPION only scored 17.7% over 130 games.

Self-play RL only — ASU is called strictly as a black-box opponent via
choose_action(env), never read for labels/weights (rule 2 allowed).
This table is much slower than fixed-only tables (3 ASU decisions/round
instead of 0-1), so expect fewer games/hour.

Usage:
    python tools/train_vs_3asu.py --algo ppo --games 500 --out artifacts/ppo_plus/vs_3asu.pt
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASUValueV1
from monopoly_game_engine.agent_ddqn import DDQNAgent
from monopoly_game_engine.agent_ppo import PPOAgent
from monopoly_game_engine.constants import NUM_PLAYERS
from monopoly_game_engine.env import MonopolyEnv
from monopoly_game_engine.train import run_episode


def main():
    parser = argparse.ArgumentParser(description="Train directly vs 3x ASUValueV1")
    parser.add_argument("--algo", choices=["ppo", "ddqn"], default="ppo")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    agent_pid = 0
    other_pids = [i for i in range(NUM_PLAYERS) if i != agent_pid]

    if args.algo == "ppo":
        agent = PPOAgent(player_id=agent_pid, hybrid=True)
    else:
        agent = DDQNAgent(player_id=agent_pid, hybrid=True)
    if args.resume and Path(args.out).exists():
        agent.load(args.out)

    fp_agents = [ASUValueV1(pid) for pid in other_pids]
    env = MonopolyEnv(agent_ids=[agent_pid], max_rounds=200)

    log_path = str(Path(args.out).with_suffix("")) + "_games.csv"
    is_new = not (args.resume and Path(log_path).exists())
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if is_new:
        writer.writerow(["game", "won", "reward", "steps", "properties_acquired",
                          "trades_initiated", "trades_accepted", "trades_declined"])

    wins_window = 0
    started_at = time.time()
    starting_games = int(getattr(agent, "games_trained", 0))

    print(f"vs 3x ASUValueV1 — {args.algo.upper()} hybrid, {args.games} games, out={args.out}")

    for game_num in range(1, args.games + 1):
        absolute_game = starting_games + game_num
        episode_seed = args.seed + absolute_game - 1
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)

        t0 = time.time()
        result = run_episode(env, agent, fp_agents, agent_pid, args.algo == "ppo")
        elapsed = time.time() - t0
        agent.games_trained = absolute_game

        writer.writerow([
            absolute_game, int(result["won"]), f"{result['reward']:.4f}", result["steps"],
            result["properties_acquired"], result["trades_initiated"],
            result["trades_accepted"], result["trades_declined"],
        ])
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
