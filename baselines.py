import argparse
import csv
import json
import math
import statistics
from collections import deque
from pathlib import Path


PRICE_COLUMNS = ("hour", "high", "low", "open", "close")
BASELINE_COLUMNS = ("log_return", "rolling_std", "ewma_volatility", "parkinson_volatility")


def calculate_baselines(rows, window=20, decay=0.94):
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2 candles")
    if not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("decay must be finite and strictly between 0 and 1")

    returns = deque(maxlen=window)
    range_variances = deque(maxlen=window)
    previous_hour = previous_close = ewma_variance = None
    results = []

    for hour, high, low, opening, close in rows:
        if any(not math.isfinite(price) or price <= 0 for price in (high, low, opening, close)):
            raise ValueError(f"hour {hour}: prices must be finite and positive")
        if not low <= min(opening, close) <= max(opening, close) <= high:
            raise ValueError(f"hour {hour}: high and low must contain the open and close")
        if previous_hour is not None and hour != previous_hour + 1:
            raise ValueError("candles must be ordered and exactly one hour apart")

        # subtract logs so large price ratios don't overflow
        log_range = math.log(high) - math.log(low)
        range_variances.append(log_range**2 / (4 * math.log(2)))
        parkinson = None
        if len(range_variances) == window:
            parkinson = math.sqrt(math.fsum(range_variances) / window)

        log_return = rolling_std = ewma = None
        if previous_close is not None:
            log_return = math.log(close) - math.log(previous_close)
            returns.append(log_return)
            if len(returns) == window:
                rolling_std = statistics.stdev(returns)
                if ewma_variance is None:
                    ewma_variance = math.fsum(value**2 for value in returns) / window
                else:
                    ewma_variance = decay * ewma_variance + (1 - decay) * log_return**2
                ewma = math.sqrt(ewma_variance)

        results.append(dict(zip(PRICE_COLUMNS + BASELINE_COLUMNS, (
            hour, high, low, opening, close, log_return, rolling_std, ewma, parkinson,
        ))))
        previous_hour, previous_close = hour, close

    if not results:
        raise ValueError("the CSV needs at least one candle")
    return results


def main():
    parser = argparse.ArgumentParser(description="calculate hourly volatility baselines")
    parser.add_argument("input", nargs="?", type=Path, default=Path("data/market.csv"),
                        help="hourly candle CSV (default: data/market.csv)")
    parser.add_argument("--output", type=Path, default=Path("data/baselines.csv"),
                        help="output CSV (default: data/baselines.csv)")
    parser.add_argument("--window", type=int, default=20,
                        help="rolling SD/Parkinson window and EWMA seed length (default: 20 candles)")
    parser.add_argument("--decay", type=float, default=0.94,
                        help="EWMA weight on previous variance, between 0 and 1 (default: 0.94)")
    args = parser.parse_args()
    settings_path = args.output.with_suffix(".settings.json")
    if len({path.resolve() for path in (args.input, args.output, settings_path)}) != 3:
        parser.error("input, output, and settings must be different files")

    try:
        with args.input.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not set(PRICE_COLUMNS).issubset(reader.fieldnames or []):
                raise ValueError("expected hour, high, low, open, close columns from market.py")
            rows = [(int(row["hour"]), *(float(row[key]) for key in PRICE_COLUMNS[1:]))
                    for row in reader]
        results = calculate_baselines(rows, args.window, args.decay)
        settings = {
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "rows": len(results),
            "candle_interval": "1 hour",
            "units": "decimal log-return volatility per hour",
            "annualized": False,
            "return_definition": "ln(close_t / close_previous)",
            "rolling_window": args.window,
            "rolling_std_ddof": 1,
            "parkinson_window": args.window,
            "parkinson_definition": "sqrt(mean(ln(high / low)^2 / (4 * ln(2))))",
            "ewma_decay": args.decay,
            "ewma_mean_assumption": "zero",
            "ewma_initialization": "mean squared return of first full window",
            "ewma_initialization_returns": args.window,
            "ewma_update": "variance_t = decay * variance_previous + (1 - decay) * return_t^2",
            "alignment": "at candle close, including the current candle; no future observations",
            "missing_values": "blank until enough history is available",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=PRICE_COLUMNS + BASELINE_COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.error(str(error))

    complete = sum(all(row[column] is not None for column in BASELINE_COLUMNS[1:]) for row in results)
    print(f"Saved {len(results)} candles ({complete} with all three baselines) to {args.output}")
    print(f"Settings: {settings_path}")


if __name__ == "__main__":
    main()
