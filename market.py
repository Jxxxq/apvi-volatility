import argparse
import csv
import math
import random
from pathlib import Path


def simulate_market(days=500, volatility=0.01, seed=42, high_volatility=None, regime_days=100):
    if not isinstance(days, int) or days < 1:
        raise ValueError("duration must be a positive integer number of trading days")
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be finite and nonnegative")
    if not isinstance(regime_days, int) or regime_days < 1:
        raise ValueError("regime days must be a positive integer")
    if high_volatility is not None and (not math.isfinite(high_volatility) or high_volatility <= volatility):
        raise ValueError("high volatility must be finite and greater than low volatility")

    rng = random.Random(seed)
    price = 100.0
    steps_per_day = 60
    rows = []
    for day in range(days):
        high_regime = high_volatility is not None and (day // regime_days) % 2 == 1
        daily_volatility = high_volatility if high_regime else volatility
        step_volatility = daily_volatility / math.sqrt(steps_per_day)
        opening = high = low = price
        for _ in range(steps_per_day):
            price *= math.exp(rng.gauss(0.0, step_volatility))
            if not math.isfinite(price) or price <= 0:
                raise ValueError("price is out of range; reduce volatility or duration")
            high = max(high, price)
            low = min(low, price)
        candle = (day, high, low, opening, price)
        if high_volatility is not None:
            candle += ("high" if high_regime else "low", daily_volatility)
        rows.append(candle)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=500, help="number of daily candles (default: 500 trading days)")
    parser.add_argument("--volatility", type=float, default=0.01, help="low or constant daily log-return standard deviation (default: 0.01)")
    parser.add_argument("--high-volatility", type=float, help="enable alternating low/high periods at this daily volatility")
    parser.add_argument("--regime-days", type=int, default=100, help="trading days per low or high period (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=Path("data/market.csv"), help="CSV destination (default: data/market.csv)")
    args = parser.parse_args()
    try:
        rows = simulate_market(args.days, args.volatility, args.seed, args.high_volatility, args.regime_days)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        columns = ["day", "high", "low", "open", "close"]
        if args.high_volatility is not None:
            columns += ["regime", "true_volatility"]
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"Saved {len(rows)} daily candles to {args.output}")


if __name__ == "__main__":
    main()
