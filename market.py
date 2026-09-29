import argparse
import csv
import math
import random
from pathlib import Path


def simulate_market(days=500, volatility=0.01, seed=42):
    if not isinstance(days, int) or days < 1:
        raise ValueError("duration must be a positive integer number of trading days")
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be finite and nonnegative")

    rng = random.Random(seed)
    price = 100.0
    steps_per_day = 60
    step_volatility = volatility / math.sqrt(steps_per_day)
    rows = []
    for day in range(days):
        opening = high = low = price
        for _ in range(steps_per_day):
            price *= math.exp(rng.gauss(0.0, step_volatility))
            if not math.isfinite(price) or price <= 0:
                raise ValueError("price is out of range; reduce volatility or duration")
            high = max(high, price)
            low = min(low, price)
        rows.append((day, high, low, opening, price))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=500, help="number of daily candles (default: 500 trading days)")
    parser.add_argument("--volatility", type=float, default=0.01, help="daily log-return standard deviation (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=Path("data/market.csv"), help="CSV destination (default: data/market.csv)")
    args = parser.parse_args()
    try:
        rows = simulate_market(args.days, args.volatility, args.seed)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["day", "high", "low", "open", "close"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} daily candles to {args.output}")


if __name__ == "__main__":
    main()
