import argparse
import csv
import math
import random
from pathlib import Path


def simulate_market(hours=24, volatility=0.01, seed=42):
    if hours < 1:
        raise ValueError("duration must be at least 1 hour")
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be finite and nonnegative")

    rng = random.Random(seed)
    price = 100.0
    minute_volatility = volatility / math.sqrt(60)
    rows = []
    for hour in range(hours):
        opening = high = low = price
        for _ in range(60):
            price *= math.exp(rng.gauss(0.0, minute_volatility))
            if not math.isfinite(price) or price <= 0:
                raise ValueError("price is out of range; reduce volatility or duration")
            high = max(high, price)
            low = min(low, price)
        rows.append((hour, high, low, opening, price))
    return rows


def main():
    parser = argparse.ArgumentParser()
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--hours", type=int, help="number of hourly candles (default: 24)")
    duration.add_argument("--days", type=int, help="duration in days, with 24 candles per day")
    parser.add_argument("--volatility", type=float, default=0.01, help="hourly log-return standard deviation (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=Path("data/market.csv"), help="CSV destination (default: data/market.csv)")
    args = parser.parse_args()
    hours = args.days * 24 if args.days is not None else args.hours
    if hours is None:
        hours = 24

    try:
        rows = simulate_market(hours, args.volatility, args.seed)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["hour", "high", "low", "open", "close"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} one-hour candles to {args.output}")


if __name__ == "__main__":
    main()
