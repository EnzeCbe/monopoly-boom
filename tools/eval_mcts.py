"""
eval_mcts.py
------------
Boost a trained PPO checkpoint with Max-N PUCT search (monopoly_bench/search.py)
at DECISION TIME ONLY — no extra training, no ASU involvement anywhere in this
path. Compares raw-policy win rate vs search-boosted win rate against the same
opponent table, so we know whether search is actually worth the extra latency.

The PPO checkpoint supplies the trunk+policy prior (MonopolyZeroNet.load_ppo_actor);
its value head is untrained (random init) since PPO checkpoints don't carry a
4-player value head — search still benefits from the trained policy priors even
so, and gets extra signal from terminal_value() at won/lost leaf nodes.

Usage:
    python tools/eval_mcts.py --model artifacts/ppo_plus/ppo_hybrid_mix_v1.pt --games 30
    python tools/eval_mcts.py --model model.pt --games 30 --simulations 64
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_bench.adapters import FixedAdapter, PPOAdapter, SearchAdapter
from monopoly_bench.arena import play_game
from monopoly_bench.config import SearchConfig
from monopoly_bench.model import MonopolyZeroNet
from monopoly_game_engine.agents_fixed import TheBuilder, TheDealMaker, TheHoarder


def run_match(policies, games: int, seed: int, label: str):
    wins = 0
    crashes = 0
    total_latency = 0.0
    started = time.time()
    for i in range(games):
        result = play_game(game_id=i, seed=seed + i, policies=policies)
        if result.crashes:
            crashes += 1
            print(f"  [{label}] game {i} crashed: {result.error}")
            continue
        if result.winner == 0:
            wins += 1
        total_latency += sum(result.search_latencies)
    elapsed = time.time() - started
    rate = wins / games * 100
    print(
        f"[{label}] win rate: {rate:.1f}% ({wins}/{games})  "
        f"crashes={crashes}  wall={elapsed:.1f}s  "
        f"search_latency_total={total_latency:.1f}s"
    )
    return rate


def main():
    parser = argparse.ArgumentParser(description="Compare raw PPO vs PPO+MCTS win rate")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-raw", action="store_true", help="Only run the search-boosted match")
    args = parser.parse_args()

    other_seats = {
        1: FixedAdapter(TheBuilder),
        2: FixedAdapter(TheDealMaker),
        3: FixedAdapter(TheHoarder),
    }

    if not args.skip_raw:
        raw_policies = {0: PPOAdapter(args.model), **other_seats}
        raw_rate = run_match(raw_policies, args.games, args.seed, "raw PPO")
    else:
        raw_rate = None

    model = MonopolyZeroNet()
    loaded = model.load_ppo_actor(args.model)
    print(f"Bootstrapped MonopolyZeroNet from PPO actor: {loaded}")
    model.eval()

    config = SearchConfig(simulations=args.simulations)
    search_policies = {0: SearchAdapter(model, config, self_play=False), **other_seats}
    search_rate = run_match(search_policies, args.games, args.seed, f"PPO+MCTS(sims={args.simulations})")

    if raw_rate is not None:
        print(f"\nDelta: {search_rate - raw_rate:+.1f} percentage points from search")


if __name__ == "__main__":
    main()
