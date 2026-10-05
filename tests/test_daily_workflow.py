import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from market import simulate_market


class DailyWorkflowTests(unittest.TestCase):
    def test_one_candle_per_trading_day_with_valid_prices(self):
        rows = simulate_market(days=5, volatility=0.04, seed=42)
        self.assertEqual([row[0] for row in rows], list(range(5)))
        self.assertEqual(rows, simulate_market(days=5, volatility=0.04, seed=42))
        previous_close = 100.0
        for day, high, low, opening, close in rows:
            self.assertEqual(opening, previous_close)
            self.assertTrue(0 < low <= min(opening, close) <= max(opening, close) <= high)
            previous_close = close

    def test_simulated_daily_returns_match_requested_volatility(self):
        rows = simulate_market(days=2500, volatility=0.04, seed=42)
        returns = [math.log(current[4] / previous[4])
                   for previous, current in zip(rows, rows[1:])]
        self.assertAlmostEqual(statistics.stdev(returns), 0.04, delta=0.002)

    def test_zero_volatility_and_invalid_duration(self):
        self.assertEqual(simulate_market(days=3, volatility=0),
                         [(day, 100.0, 100.0, 100.0, 100.0) for day in range(3)])
        for days in (0, -1, 2.5):
            with self.subTest(days=days), self.assertRaises(ValueError):
                simulate_market(days=days)

    def test_alternating_regimes_have_known_volatility_and_continuous_prices(self):
        rows = simulate_market(days=12, volatility=0.01, high_volatility=0.04,
                               regime_days=3, seed=42)
        self.assertEqual([row[5] for row in rows], ["low"] * 3 + ["high"] * 3 +
                         ["low"] * 3 + ["high"] * 3)
        self.assertEqual([row[6] for row in rows], [0.01] * 3 + [0.04] * 3 +
                         [0.01] * 3 + [0.04] * 3)
        self.assertEqual(rows, simulate_market(days=12, volatility=0.01,
                                                high_volatility=0.04, regime_days=3, seed=42))
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(current[3], previous[4])
        for settings in ({"high_volatility": 0.01}, {"high_volatility": float("nan")},
                         {"high_volatility": 0.04, "regime_days": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                simulate_market(days=3, volatility=0.01, **settings)

    def test_regime_csv_works_as_baseline_input(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            market = Path(directory) / "market.csv"
            baselines = Path(directory) / "baselines.csv"
            commands = [
                ["market.py", "--days", "9", "--volatility", "0.01", "--high-volatility", "0.04",
                 "--regime-days", "3", "--output", str(market)],
                ["baselines.py", str(market), "--output", str(baselines)],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, "-B", *command], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            with market.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames, ["day", "high", "low", "open", "close",
                                                     "regime", "true_volatility"])
                rows = list(reader)
            self.assertEqual([(row["regime"], row["true_volatility"]) for row in rows],
                             [("low", "0.01")] * 3 + [("high", "0.04")] * 3 +
                             [("low", "0.01")] * 3)

    def test_daily_files_work_across_all_three_commands(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            market = Path(directory) / "market.csv"
            baselines = Path(directory) / "baselines.csv"
            chart = Path(directory) / "market.html"
            commands = [
                ["market.py", "--days", "30", "--volatility", "0.04", "--output", str(market)],
                ["baselines.py", str(market), "--output", str(baselines)],
                ["visualize.py", str(baselines), "--true-volatility", "0.04", "--output", str(chart)],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, "-B", *command], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

            with market.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames, ["day", "high", "low", "open", "close"])
                prices = list(reader)
            self.assertEqual(len(prices), 30)
            self.assertEqual(prices[-1]["day"], "29")
            with baselines.open(newline="", encoding="utf-8") as source:
                estimates = list(csv.DictReader(source))
            self.assertEqual(len(estimates), len(prices))
            self.assertEqual(sum(bool(row["rolling_std"]) for row in estimates), 10)
            self.assertEqual(sum(bool(row["ewma_volatility"]) for row in estimates), 10)
            self.assertEqual(sum(bool(row["parkinson_volatility"]) for row in estimates), 11)
            settings = json.loads(baselines.with_suffix(".settings.json").read_text())
            self.assertEqual(settings["candle_interval"], "1 trading day")
            self.assertEqual(settings["units"], "decimal log-return volatility per trading day")
            self.assertFalse(settings["annualized"])
            html = chart.read_text(encoding="utf-8")
            self.assertIn("30 trading days", html)
            self.assertIn("Day 29:", html)
            self.assertIn("Correct / evaluated days", html)
            self.assertIn("&plusmn;10.0%", html)
            self.assertEqual(html.count(" / 10</td>"), 3)
            for label in ("Rolling standard deviation", "EWMA volatility", "Parkinson high-low volatility"):
                self.assertIn(label, html)


if __name__ == "__main__":
    unittest.main()
