import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

from market import simulate_market
from paired import calculate_paired


METHODS = {
    "apvi": "Paired index (APVI)",
    "arithmetic": "Matched simple average",
    "rolling_std": "Rolling standard deviation",
    "ewma_volatility": "EWMA volatility",
    "parkinson_volatility": "Parkinson volatility",
    "ewma_parkinson": "EWMA Parkinson",
    "return_only": "Processed return only",
    "range_only": "Processed range only",
    "unrectified_average": "Unrectified simple average",
}
SUMMARY_METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1",
                   "false_positive_rate", "mean_delay_days", "missed_episode_rate")


def validate_scores(scores, labels):
    if not scores or len(scores) != len(labels):
        raise ValueError("scores and labels must be nonempty and have the same length")
    if any(not math.isfinite(score) for score in scores) or any(label not in (0, 1) for label in labels):
        raise ValueError("scores must be finite and labels must be 0 or 1")


def fit_threshold(scores, labels, false_positive_rate=0.05):
    validate_scores(scores, labels)
    if not math.isfinite(false_positive_rate) or not 0 <= false_positive_rate < 1:
        raise ValueError("false-positive rate must be in [0, 1)")
    low_scores = sorted(score for score, label in zip(scores, labels) if not label)
    if not low_scores:
        raise ValueError("validation needs low-volatility days to set a threshold")
    allowed = math.floor(false_positive_rate * len(low_scores))
    return low_scores[len(low_scores) - allowed - 1]


def detection_metrics(scores, labels, threshold):
    validate_scores(scores, labels)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    seen_positive = seen_negative = 0
    ap = wins = 0.0
    for score, group in itertools.groupby(sorted(zip(scores, labels), reverse=True), key=lambda pair: pair[0]):
        group_labels = [label for _, label in group]
        p, n = sum(group_labels), len(group_labels) - sum(group_labels)
        seen_positive += p
        seen_negative += n
        if positives:
            ap += p / positives * seen_positive / (seen_positive + seen_negative)
        wins += p * (negatives - seen_negative + n / 2)
    alerts = [score > threshold for score in scores]
    tp = sum(alert and label == 1 for alert, label in zip(alerts, labels))
    fp = sum(alert and label == 0 for alert, label in zip(alerts, labels))
    fn, tn = positives - tp, negatives - fp
    return {
        "days": len(labels), "high_days": positives, "prevalence": positives / len(labels),
        "average_precision": ap if positives else None,
        "roc_auc": wins / (positives * negatives) if positives and negatives else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
    }


def episode_metrics(labels, alerts, previous_high=False):
    delays = []
    episodes = missed = 0
    for label, group in itertools.groupby(enumerate(labels), key=lambda pair: pair[1]):
        indices = [index for index, _ in group]
        if not label or (indices[0] == 0 and previous_high):
            continue
        episodes += 1
        delay = next((index - indices[0] for index in indices if alerts[index]), None)
        if delay is None:
            missed += 1
        else:
            delays.append(delay)
    return {"episodes": episodes, "missed_episodes": missed,
            "missed_episode_rate": missed / episodes if episodes else None,
            "mean_delay_days": statistics.mean(delays) if delays else None}


def compare_run(candles, train_days=400, validation_days=400, window=20, decay=0.94,
                false_positive_rate=0.05):
    if (not isinstance(train_days, int) or not isinstance(validation_days, int)
            or train_days <= window or validation_days < 1 or train_days + validation_days >= len(candles)):
        raise ValueError("need training longer than the baseline window, plus separate validation and test days")
    if any(len(row) != 7 or row[5] not in ("low", "high")
           or not math.isfinite(row[6]) or row[6] < 0 for row in candles):
        raise ValueError("comparison needs candles with low/high regime labels and finite nonnegative true volatility")
    records, reference = calculate_paired([row[:5] for row in candles], train_days, window, decay)
    test_start = train_days + validation_days
    for index, (record, candle) in enumerate(zip(records, candles)):
        record.update(regime=candle[5], true_volatility=candle[6],
                      split="train" if index < train_days else "validation" if index < test_start else "test")
    validation, test = records[train_days:test_start], records[test_start:]
    validation_labels = [int(row["regime"] == "high") for row in validation]
    test_labels = [int(row["regime"] == "high") for row in test]
    if set(validation_labels) != {0, 1} or set(test_labels) != {0, 1}:
        raise ValueError("validation and test periods must each contain low and high regimes")
    results, thresholds = [], {}
    for method in METHODS:
        validation_scores = [row[method] for row in validation]
        scores = [row[method] for row in test]
        threshold = fit_threshold(validation_scores, validation_labels, false_positive_rate)
        thresholds[method] = threshold
        result = {"method": method, "threshold": threshold, **detection_metrics(scores, test_labels, threshold)}
        result["validation_fpr"] = detection_metrics(validation_scores, validation_labels, threshold)["false_positive_rate"]
        result.update(episode_metrics(test_labels, [score > threshold for score in scores],
                                      records[test_start - 1]["regime"] == "high"))
        results.append(result)
        for record in records:
            record[method + "_alert"] = None if record["split"] == "train" else int(record[method] > threshold)
    return records, results, {"reference": reference, "thresholds": thresholds}


def summarize(results):
    summary = []
    for method in METHODS:
        runs = [row for row in results if row["method"] == method]
        row = {"method": method, "runs": len(runs)}
        for metric in SUMMARY_METRICS:
            values = [run[metric] for run in runs if run[metric] is not None]
            row[metric + "_mean"] = statistics.mean(values) if values else None
            row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
            row[metric + "_runs"] = len(values)
        summary.append(row)
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(records, results, summary, output, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    test = [row for row in records if row["split"] == "test"]
    days = [row["day"] for row in test]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [1, 1, 1.3]})
    for ax in axes[:2]:
        for regime, group in itertools.groupby(test, key=lambda row: row["regime"]):
            block = list(group)
            if regime == "high":
                ax.axvspan(block[0]["day"] - 0.5, block[-1]["day"] + 0.5, color="#deebf7", zorder=0)
        ax.set_xlim(days[0], days[-1])
        ax.set_xlabel("Trading day")
        ax.grid(alpha=0.2)
    axes[0].plot(days, [row["true_volatility"] for row in test], color="black", linewidth=2,
                 drawstyle="steps-mid", label="True daily volatility")
    for method, color in (("rolling_std", "#0072b2"), ("ewma_volatility", "#d55e00"),
                          ("parkinson_volatility", "#009e73")):
        axes[0].plot(days, [row[method] for row in test], color=color, linewidth=1.2, label=METHODS[method])
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Daily volatility")
    axes[0].set_title(f"Held-out test days, seed {seed} | shaded periods have high volatility")
    axes[0].legend(fontsize=9, ncol=2, loc="upper left")
    for method, color in (("apvi", "#0072b2"), ("arithmetic", "#d55e00")):
        threshold = next(row["threshold"] for row in results if row["method"] == method)
        axes[1].plot(days, [row[method] for row in test], color=color, label=METHODS[method])
        axes[1].axhline(threshold, color=color, linestyle="--", linewidth=1, label=f"{METHODS[method]} threshold")
    axes[1].set_ylabel("Dimensionless score")
    axes[1].set_title("Same training scales and smoothing; thresholds fixed using validation days")
    axes[1].legend(fontsize=9, ncol=2, loc="upper left")
    ordered = sorted(summary, key=lambda row: row["average_precision_mean"])
    means = [row["average_precision_mean"] for row in ordered]
    axes[2].barh([METHODS[row["method"]] for row in ordered], means,
                 xerr=[row["average_precision_sd"] or 0 for row in ordered],
                 color=["#0072b2" if row["method"] == "apvi" else "#8aabbf" for row in ordered], capsize=3)
    for index, value in enumerate(means):
        axes[2].text(0.02, index, f"{value:.3f}", va="center", color="black")
    axes[2].set_xlim(0, 1.02)
    axes[2].set_xlabel("Average precision (higher is better)")
    axes[2].set_title(f"Test results across {summary[0]['runs']} seeds | bars: means, whiskers: standard deviations")
    fig.tight_layout(pad=2)
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="compare detection of simulated low/high volatility")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--train-days", type=int, default=400)
    parser.add_argument("--validation-days", type=int, default=400)
    parser.add_argument("--volatility", type=float, default=0.01)
    parser.add_argument("--high-volatility", type=float, default=0.04)
    parser.add_argument("--regime-days", type=int, default=100)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.94)
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42, help="first random seed (default: 42)")
    parser.add_argument("--seeds", type=int, default=20, help="number of consecutive random seeds (default: 20)")
    parser.add_argument("--output", type=Path, default=Path("data/comparison"))
    parser.add_argument("--plot", action="store_true", help="save PNG and SVG figures (requires matplotlib)")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    try:
        all_results, fits = [], []
        first_records = first_results = None
        for seed in range(args.seed, args.seed + args.seeds):
            candles = simulate_market(args.days, args.volatility, seed, args.high_volatility, args.regime_days)
            records, results, fit = compare_run(candles, args.train_days, args.validation_days,
                                                args.window, args.decay, args.false_positive_rate)
            args.output.mkdir(parents=True, exist_ok=True)
            write_csv(args.output / f"scores_seed_{seed}.csv", records)
            all_results.extend({"seed": seed, **row} for row in results)
            fits.append({"seed": seed, **fit})
            if first_records is None:
                first_records, first_results = records, results
        summary = summarize(all_results)
        write_csv(args.output / "metrics.csv", all_results)
        write_csv(args.output / "summary.csv", summary)
        differences = [next(row["average_precision"] for row in all_results if row["seed"] == fit["seed"] and row["method"] == "apvi")
                       - next(row["average_precision"] for row in all_results if row["seed"] == fit["seed"] and row["method"] == "arithmetic")
                       for fit in fits]
        settings = {**vars(args), "output": str(args.output.resolve()), "fits": fits,
                    "target": "same-day simulated high-volatility regime at candle close",
                    "threshold_rule": "score > validation low-state order statistic; at most nominal validation FPR",
                    "training_scores": "in-sample burn-in only; excluded from evaluation",
                    "primary_comparison": "APVI minus matched rectified arithmetic, test average precision",
                    "paired_ap_difference_mean": statistics.mean(differences),
                    "paired_ap_difference_sd": statistics.stdev(differences) if len(differences) > 1 else None,
                    "paired_ap_wins": sum(value > 0 for value in differences)}
        (args.output / "settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.plot:
            plot_comparison(first_records, first_results, summary, args.output, args.seed)
    except (OSError, ValueError, TypeError, OverflowError, ImportError) as error:
        parser.error(str(error))
    print(f"Test means across {args.seeds} seeds; {args.days - args.train_days - args.validation_days} test days per seed")
    print(f"{'Method':30} {'AP':>7} {'Recall':>8} {'FPR':>8}")
    for row in summary:
        print(f"{METHODS[row['method']]:30} {row['average_precision_mean']:7.3f} {row['recall_mean']:8.1%} {row['false_positive_rate_mean']:8.1%}")
    print(f"APVI minus matched average AP: {settings['paired_ap_difference_mean']:+.4f}")
    print(f"Saved comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
