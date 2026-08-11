"""
analyze_run.py
---------------
Reads a per-game training CSV (produced by train.py's log_path / written to
<out>_games.csv by tools/train_and_save.py) and prints rolling trends so we
don't have to eyeball a wall of pasted log lines.

Usage:
    python tools/analyze_run.py artifacts/ddqn_builder_dealmaker_games.csv
    python tools/analyze_run.py path/to/games.csv --window 100
"""

import argparse
import csv
import statistics
from pathlib import Path


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def windowed(rows: list[dict], window: int):
    for i in range(0, len(rows), window):
        yield rows[i : i + window]


def main():
    parser = argparse.ArgumentParser(description="Analyze a per-game training CSV")
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        raise SystemExit(f"No such file: {args.csv_path}")

    rows = load_rows(args.csv_path)
    if not rows:
        raise SystemExit("CSV is empty")

    total = len(rows)
    total_wins = sum(int(r["won"]) for r in rows)
    print(f"\n{args.csv_path}")
    print(f"Total games: {total}   Overall win rate: {total_wins / total * 100:.1f}%")
    first_win_at = next((r["game"] for r in rows if int(r["won"])), None)
    print(f"First win at game: {first_win_at or 'never'}")

    print(f"\n{'Games':>15} | {'Win%':>6} | {'AvgReward':>10} | {'AvgSteps':>8} | {'AvgProps':>8} | {'Eps':>6}")
    print("-" * 70)
    for chunk in windowed(rows, args.window):
        lo, hi = chunk[0]["game"], chunk[-1]["game"]
        win_rate = sum(int(r["won"]) for r in chunk) / len(chunk) * 100
        avg_reward = statistics.mean(float(r["reward"]) for r in chunk)
        avg_steps = statistics.mean(int(r["steps"]) for r in chunk)
        avg_props = statistics.mean(int(r["properties_acquired"]) for r in chunk)
        eps_vals = [r["epsilon"] for r in chunk if r.get("epsilon")]
        eps_str = f"{float(eps_vals[-1]):.3f}" if eps_vals else "-"
        print(
            f"{lo:>7}-{hi:<7} | {win_rate:5.1f}% | {avg_reward:10.3f} | "
            f"{avg_steps:8.0f} | {avg_props:8.1f} | {eps_str:>6}"
        )

    # Reward trend: first-quarter vs last-quarter average, cheap "is it learning" signal
    q = max(1, total // 4)
    first_q_reward = statistics.mean(float(r["reward"]) for r in rows[:q])
    last_q_reward = statistics.mean(float(r["reward"]) for r in rows[-q:])
    first_q_win = sum(int(r["won"]) for r in rows[:q]) / q * 100
    last_q_win = sum(int(r["won"]) for r in rows[-q:]) / q * 100
    print(f"\nFirst quarter: avg_reward={first_q_reward:.3f}  win%={first_q_win:.1f}%")
    print(f"Last quarter : avg_reward={last_q_reward:.3f}  win%={last_q_win:.1f}%")
    if last_q_reward > first_q_reward:
        print("-> reward trending UP: shaping signal is moving the agent, even without wins yet")
    else:
        print("-> reward NOT trending up: worth checking hyperparameters/opponent difficulty")


if __name__ == "__main__":
    main()
