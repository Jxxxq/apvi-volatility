import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compare import METHODS, compare_run, detection_metrics, episode_metrics, fit_threshold
from market import simulate_market


class ComparisonTests(unittest.TestCase):
    def test_hand_calculated_metrics_and_tied_scores(self):
        result = detection_metrics([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0], 0.5)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertEqual(result["roc_auc"], 0.875)
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        tied = detection_metrics([1, 1, 1, 1], [1, 1, 0, 0], 1)
        self.assertEqual(tied["average_precision"], 0.5)
        self.assertEqual(tied["roc_auc"], 0.5)
        self.assertIsNone(tied["precision"])
        self.assertEqual(tied["recall"], 0)
        one_class = detection_metrics([0, 1], [0, 0], 0)
        self.assertIsNone(one_class["roc_auc"])
        self.assertIsNone(one_class["average_precision"])
        self.assertIsNone(one_class["recall"])

    def test_threshold_limits_validation_false_alarms_including_ties(self):
        self.assertEqual(fit_threshold(list(range(20)), [0] * 20), 18)
        scores = [0] * 18 + [1, 1]
        threshold = fit_threshold(scores, [0] * 20)
        self.assertEqual(threshold, 1)
        self.assertLessEqual(sum(value > threshold for value in scores), 1)
        self.assertEqual(fit_threshold(list(range(20)) + [1000], [0] * 20 + [1]), 18)
        with self.assertRaises(ValueError):
            fit_threshold([1, 2], [1, 1])
        with self.assertRaises(ValueError):
            fit_threshold([1, 2], [0, 1], 1)
        with self.assertRaises(ValueError):
            detection_metrics([math.nan], [0], 1)

    def test_detection_delay_counts_missed_episodes_and_ignores_partial_start(self):
        labels = [1, 1, 0, 1, 1, 1, 0, 1, 1]
        result = episode_metrics(labels, [0, 1, 0, 0, 0, 1, 0, 0, 0], previous_high=True)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["missed_episodes"], 1)
        self.assertEqual(result["mean_delay_days"], 2)
        self.assertEqual(result["missed_episode_rate"], 0.5)

    def test_test_data_cannot_change_fitted_scales_thresholds_or_earlier_scores(self):
        candles = simulate_market(days=200, high_volatility=0.04, regime_days=20)
        records, results, fit = compare_run(candles, train_days=60, validation_days=60, window=10)
        changed = [row if index < 160 else (row[0], *(price * 2 for price in row[1:5]), *row[5:])
                   for index, row in enumerate(candles)]
        altered, _, changed_fit = compare_run(changed, train_days=60, validation_days=60, window=10)
        self.assertEqual(fit, changed_fit)
        self.assertEqual(records[:160], altered[:160])
        self.assertEqual(len(results), len(METHODS))
        for result in results:
            self.assertEqual(result["days"], 80)
            self.assertLessEqual(result["validation_fpr"], 0.05)
        with self.assertRaisesRegex(ValueError, "each contain low and high"):
            compare_run(candles, train_days=60, validation_days=10, window=10)

    def test_commands_save_reproducible_scores_labels_and_test_results(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            market = output / "market.csv"
            paired = output / "paired.csv"
            commands = [
                ["market.py", "--days", "120", "--high-volatility", "0.04", "--regime-days", "10", "--output", str(market)],
                ["paired.py", str(market), "--train-days", "40", "--window", "5", "--output", str(paired)],
                ["compare.py", "--days", "120", "--train-days", "40", "--validation-days", "40",
                 "--window", "5", "--regime-days", "10", "--seeds", "2", "--output", str(output / "comparison")],
                ["visualize.py", "--comparison", str(output / "comparison"), "--output", str(output / "visual.html")],
            ]
            for command in commands:
                completed = subprocess.run([sys.executable, "-B", *command], cwd=root, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            with paired.open(newline="", encoding="utf-8") as source:
                paired_rows = list(csv.DictReader(source))
            self.assertEqual(len(paired_rows), 120)
            self.assertEqual(paired_rows[40]["split"], "post_training")
            self.assertEqual(paired_rows[50]["regime"], "high")
            self.assertEqual(float(paired_rows[50]["true_volatility"]), 0.04)
            with (output / "comparison" / "metrics.csv").open(newline="", encoding="utf-8") as source:
                metrics = list(csv.DictReader(source))
            self.assertEqual(len(metrics), 2 * len(METHODS))
            self.assertTrue(all(row["days"] == "40" for row in metrics))
            with (output / "comparison" / "scores_seed_42.csv").open(newline="", encoding="utf-8") as source:
                scores = list(csv.DictReader(source))
            self.assertEqual([row["split"] for row in scores], ["train"] * 40 + ["validation"] * 40 + ["test"] * 40)
            for row in scores[40:]:
                for method in METHODS:
                    self.assertAlmostEqual(float(row[method]), float(paired_rows[int(row["day"])][method]))
                    self.assertIn(row[method + "_alert"], ("0", "1"))
            settings = json.loads((output / "comparison" / "settings.json").read_text())
            self.assertEqual(settings["fits"][0]["reference"]["last_day"], 39)
            self.assertEqual(settings["seeds"], 2)
            html = (output / "visual.html").read_text(encoding="utf-8")
            self.assertIn("2 seeds", html)
            self.assertIn("120 trading days", html)
            self.assertIn("80&ndash;119", html)
            self.assertNotIn("within &plusmn;", html)
            self.assertNotIn('<script defer src=', html)
            with (output / "comparison" / "summary.csv").open(newline="", encoding="utf-8") as source:
                summary = list(csv.DictReader(source))
            for row in summary:
                self.assertIn(METHODS[row["method"]], html)
                self.assertIn(f"{float(row['average_precision_mean']):.3f}", html)


if __name__ == "__main__":
    unittest.main()
