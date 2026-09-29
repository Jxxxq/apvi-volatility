import math
import unittest

from baselines import BASELINE_COLUMNS, calculate_baselines


def candles_from_returns(returns):
    price = 100.0
    rows = [(0, price, price, price, price)]
    for day, value in enumerate(returns, start=1):
        close = price * math.exp(value)
        rows.append((day, max(price, close), min(price, close), price, close))
        price = close
    return rows


class BaselineTests(unittest.TestCase):
    def test_hand_calculated_returns_and_rolling_sample_std(self):
        results = calculate_baselines(candles_from_returns([0.1, -0.1, 0.2]), window=2)
        self.assertIsNone(results[0]["log_return"])
        self.assertIsNone(results[1]["rolling_std"])
        self.assertAlmostEqual(results[1]["log_return"], 0.1)
        self.assertAlmostEqual(results[2]["log_return"], -0.1)
        self.assertAlmostEqual(results[2]["rolling_std"], math.sqrt(0.02))
        self.assertAlmostEqual(results[3]["rolling_std"], math.sqrt(0.045))

    def test_ewma_seed_and_recursive_update(self):
        results = calculate_baselines(candles_from_returns([0.1, 0.2, -0.3, 0.0]),
                                      window=2, decay=0.5)
        self.assertIsNone(results[1]["ewma_volatility"])
        self.assertAlmostEqual(results[2]["ewma_volatility"], math.sqrt(0.025))
        self.assertAlmostEqual(results[3]["ewma_volatility"], math.sqrt(0.0575))
        self.assertAlmostEqual(results[4]["ewma_volatility"], math.sqrt(0.02875))

    def test_parkinson_averages_variances_and_drops_old_candles(self):
        # change the ranges while keeping closes fixed
        rows = [(i, 100 * math.exp(width), 100, 100, 100)
                for i, width in enumerate([0.2, 0.4, 0.6])]
        results = calculate_baselines(rows, window=2)
        self.assertIsNone(results[0]["parkinson_volatility"])
        self.assertAlmostEqual(results[1]["parkinson_volatility"],
                               math.sqrt(0.1 / (4 * math.log(2))))
        self.assertAlmostEqual(results[2]["parkinson_volatility"],
                               math.sqrt(0.26 / (4 * math.log(2))))

    def test_default_warmup_and_constant_prices(self):
        results = calculate_baselines(candles_from_returns([0.0] * 20))
        self.assertIsNone(results[18]["parkinson_volatility"])
        self.assertEqual(results[19]["parkinson_volatility"], 0)
        for column in ("rolling_std", "ewma_volatility"):
            self.assertTrue(all(row[column] is None for row in results[:20]))
            self.assertEqual(results[20][column], 0)

    def test_price_scaling_does_not_change_measures(self):
        rows = candles_from_returns([0.01, -0.02, 0.03, -0.01])
        scaled = [(day, *(price * 1000 for price in prices)) for day, *prices in rows]
        original = calculate_baselines(rows, window=2)
        changed = calculate_baselines(scaled, window=2)
        for before, after in zip(original, changed):
            for column in BASELINE_COLUMNS:
                if before[column] is None:
                    self.assertIsNone(after[column])
                else:
                    self.assertAlmostEqual(before[column], after[column], places=12)

    def test_future_prices_cannot_change_past_estimates(self):
        original = calculate_baselines(candles_from_returns([0.01, -0.02, 0.03, 0.04]), window=2)
        changed = calculate_baselines(candles_from_returns([0.01, -0.02, 0.03, -0.5]), window=2)
        prefix = calculate_baselines(candles_from_returns([0.01, -0.02, 0.03]), window=2)
        self.assertEqual(original[:4], changed[:4])
        self.assertEqual(original[:4], prefix)

    def test_short_input_keeps_unavailable_estimates_blank(self):
        result = calculate_baselines(candles_from_returns([]))[0]
        self.assertTrue(all(result[column] is None for column in BASELINE_COLUMNS))

    def test_invalid_settings(self):
        rows = candles_from_returns([0.01])
        for window in (0, 1, 2.5):
            with self.subTest(window=window), self.assertRaises(ValueError):
                calculate_baselines(rows, window=window)
        for decay in (-0.1, 0, 1, float("nan"), float("inf")):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                calculate_baselines(rows, decay=decay)

    def test_invalid_candles(self):
        cases = [
            [],
            [(0, 101, 0, 100, 100)],
            [(0, 101, 99, 100, float("nan"))],
            [(0, float("inf"), 99, 100, 100)],
            [(0, 99, 98, 100, 100)],
            [(0, 101, 99, 100, 100), (0, 101, 99, 100, 100)],
            [(0, 101, 99, 100, 100), (2, 101, 99, 100, 100)],
            [(1, 101, 99, 100, 100), (0, 101, 99, 100, 100)],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                calculate_baselines(rows)


if __name__ == "__main__":
    unittest.main()
