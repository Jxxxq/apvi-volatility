import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from baselines import PRICE_COLUMNS, calculate_baselines


SCORE_COLUMNS = ("apvi", "arithmetic", "return_only", "range_only", "unrectified_average")


def calculate_paired(rows, train_days=200, window=20, decay=0.94):
    if not isinstance(train_days, int) or not 3 <= train_days <= len(rows):
        raise ValueError("training days must be an integer between 3 and the number of candles")
    records = calculate_baselines(rows, window, decay)
    for record in records:
        value = record["log_return"]
        record["absolute_return"] = abs(value) if value is not None else None
        record["range_signal"] = (math.log(record["high"]) - math.log(record["low"])) / math.sqrt(4 * math.log(2))

    reference = {"train_days": train_days, "paired_observations": train_days - 1,
                 "first_day": records[1]["day"], "last_day": records[train_days - 1]["day"]}
    for name, column in (("return", "absolute_return"), ("range", "range_signal")):
        values = [record[column] for record in records[1:train_days]]
        reference[name + "_mean"] = statistics.mean(values)
        reference[name + "_std"] = statistics.stdev(values)
        if reference[name + "_std"] <= 0:
            raise ValueError(f"the training {name} signal needs a positive standard deviation")

    smoothed = dict.fromkeys(SCORE_COLUMNS, 0.0)
    range_variance = None
    for index, record in enumerate(records):
        record.update(dict.fromkeys(("return_z", "range_z", *SCORE_COLUMNS)))
        if index == window - 1:
            range_variance = statistics.mean(row["range_signal"] ** 2 for row in records[:window])
        elif index >= window:
            range_variance = decay * range_variance + (1 - decay) * record["range_signal"] ** 2
        record["ewma_parkinson"] = math.sqrt(range_variance) if range_variance is not None else None
        if record["absolute_return"] is None:
            continue
        za = (record["absolute_return"] - reference["return_mean"]) / reference["return_std"]
        zb = (record["range_signal"] - reference["range_mean"]) / reference["range_std"]
        u, v = max(za, 0), max(zb, 0)
        contributions = {"apvi": math.sqrt(u * v), "arithmetic": (u + v) / 2,
                         "return_only": u, "range_only": v, "unrectified_average": (za + zb) / 2}
        for name, contribution in contributions.items():
            smoothed[name] = decay * smoothed[name] + (1 - decay) * contribution
            record[name] = smoothed[name]
        record.update(return_z=za, range_z=zb)
    return records, reference


def main():
    parser = argparse.ArgumentParser(description="calculate the paired index and comparison measures")
    parser.add_argument("input", type=Path, help="daily OHLC CSV")
    parser.add_argument("--train-days", type=int, default=200, help="initial candles used to fit reference scales (default: 200)")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.94)
    parser.add_argument("--output", type=Path, default=Path("data/paired.csv"))
    args = parser.parse_args()
    settings_path = args.output.with_suffix(".settings.json")
    if len({path.resolve() for path in (args.input, args.output, settings_path)}) != 3:
        parser.error("input, output, and settings must be different files")
    try:
        with args.input.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not set(PRICE_COLUMNS).issubset(reader.fieldnames or []):
                raise ValueError("expected day, high, low, open, close columns")
            inputs = list(reader)
        rows = [(int(row["day"]), *(float(row[key]) for key in PRICE_COLUMNS[1:])) for row in inputs]
        records, reference = calculate_paired(rows, args.train_days, args.window, args.decay)
        for index, (record, original) in enumerate(zip(records, inputs)):
            for column in ("regime", "true_volatility"):
                if column in original:
                    record[column] = original[column]
            record["split"] = "train" if index < args.train_days else "post_training"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        settings = {"input": str(args.input.resolve()), "window": args.window, "decay": args.decay,
                    "reference": reference, "score_units": "dimensionless",
                    "initialization": "zero before first paired contribution; carry state across splits",
                    "training_scores": "in-sample burn-in only; not valid for evaluation",
                    "apvi": "EWMA(sqrt(max(return_z, 0) * max(range_z, 0)))",
                    "arithmetic": "EWMA((max(return_z, 0) + max(range_z, 0)) / 2)"}
        settings_path.write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.error(str(error))
    print(f"Saved {len(records)} candles with baselines, paired index, and controls to {args.output}")
    print(f"Reference scales: {settings_path}")


if __name__ == "__main__":
    main()
