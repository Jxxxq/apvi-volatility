import math
import unittest

from paired import SCORE_COLUMNS, calculate_paired


def example_candles():
    price = 100.0
    rows = [(0, price * math.exp(0.01 * math.sqrt(4 * math.log(2))), price, price, price)]
    for day, (value, amplitude) in enumerate(((0.01, 0.02), (0.02, 0.04), (0.005, 0.01),
                                             (0.03, 0.02), (0.03, 0.06)), start=1):
        close = price * math.exp(value)
        high = price * math.exp(amplitude * math.sqrt(4 * math.log(2)))
        rows.append((day, high, price, price, close))
        price = close
    return rows


class PairedTests(unittest.TestCase):
    def test_hand_calculated_reference_rectification_and_smoothing(self):
        records, reference = calculate_paired(example_candles(), train_days=3, window=2, decay=0.5)
        self.assertAlmostEqual(reference["return_mean"], 0.015)
        self.assertAlmostEqual(reference["return_std"], 0.01 / math.sqrt(2))
        self.assertAlmostEqual(reference["range_mean"], 0.03)
        self.assertAlmostEqual(reference["range_std"], 0.02 / math.sqrt(2))
        self.assertEqual(reference["paired_observations"], 2)
        self.assertIsNone(records[0]["apvi"])
        self.assertEqual(records[1]["apvi"], 0)
        self.assertAlmostEqual(records[2]["apvi"], 0.5 / math.sqrt(2))
        self.assertLess(records[3]["return_z"], 0)
        self.assertLess(records[3]["range_z"], 0)
        self.assertAlmostEqual(records[3]["apvi"], 0.25 / math.sqrt(2))
        self.assertAlmostEqual(records[4]["apvi"], 0.125 / math.sqrt(2))
        self.assertAlmostEqual(records[4]["arithmetic"], 0.5 * records[3]["arithmetic"] + 0.75 / math.sqrt(2))
        self.assertAlmostEqual(records[5]["apvi"], 0.5 * records[4]["apvi"] + 1.5 / math.sqrt(2))
        for record in records[1:]:
            self.assertLessEqual(record["apvi"], record["arithmetic"] + 1e-12)

    def test_ewma_parkinson_seeds_variance_and_updates(self):
        records, _ = calculate_paired(example_candles(), train_days=3, window=2, decay=0.5)
        self.assertIsNone(records[0]["ewma_parkinson"])
        self.assertAlmostEqual(records[1]["ewma_parkinson"], math.sqrt((0.01 ** 2 + 0.02 ** 2) / 2))
        self.assertAlmostEqual(records[2]["ewma_parkinson"], math.sqrt(0.5 * 0.00025 + 0.5 * 0.04 ** 2))

    def test_future_observations_do_not_change_reference_or_past_scores(self):
        rows = example_candles()
        before, reference = calculate_paired(rows, train_days=3, window=2)
        changed = rows[:-1] + [(rows[-1][0], *(price * 2 for price in rows[-1][1:]))]
        after, changed_reference = calculate_paired(changed, train_days=3, window=2)
        prefix, prefix_reference = calculate_paired(rows[:-1], train_days=3, window=2)
        self.assertEqual(reference, changed_reference)
        self.assertEqual(reference, prefix_reference)
        self.assertEqual(before[:-1], after[:-1])
        self.assertEqual(before[:-1], prefix)

    def test_price_scale_invariance_and_invalid_training(self):
        rows = example_candles()
        original, _ = calculate_paired(rows, train_days=3, window=2)
        scaled, _ = calculate_paired([(day, *(price * 1000 for price in prices)) for day, *prices in rows],
                                      train_days=3, window=2)
        for before, after in zip(original[1:], scaled[1:]):
            for column in SCORE_COLUMNS:
                self.assertAlmostEqual(before[column], after[column], places=10)
        for train_days in (0, 2, 3.5, 7):
            with self.subTest(train_days=train_days), self.assertRaises(ValueError):
                calculate_paired(rows, train_days=train_days)
        with self.assertRaisesRegex(ValueError, "positive standard deviation"):
            calculate_paired([(day, 100, 100, 100, 100) for day in range(5)], train_days=3)


if __name__ == "__main__":
    unittest.main()
