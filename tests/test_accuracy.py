import unittest

from visualize import accuracy_table


def estimates(sd, ewma, parkinson):
    return dict(zip(("rolling_std", "ewma_volatility", "parkinson_volatility"),
                    ("" if value is None else str(value) for value in (sd, ewma, parkinson))))


class AccuracyTests(unittest.TestCase):
    def test_inclusive_relative_tolerance_and_counts(self):
        rows = [
            estimates(0.036, 0.044, 0.04),
            estimates(0.044, 0.045, 0.035),
            estimates(0.035, 0.04, 0.045),
        ]
        html = accuracy_table(rows, 0.04)
        self.assertEqual(html.count("<td>2 / 3</td><td>66.7%</td>"), 2)
        self.assertIn("<td>1 / 3</td><td>33.3%</td>", html)

    def test_all_methods_use_same_days(self):
        rows = [
            estimates(None, None, 0.04),
            estimates(0.04, 0.04, 0.04),
        ]
        html = accuracy_table(rows, 0.04)
        self.assertEqual(html.count("<td>1 / 1</td><td>100.0%</td>"), 3)

    def test_empty_input_and_zero_volatility(self):
        self.assertEqual(accuracy_table([], 0.04).count("<td>N/A</td>"), 3)
        html = accuracy_table([estimates(0, 0.000001, 0)], 0)
        self.assertEqual(html.count("<td>100.0%</td>"), 2)
        self.assertIn("<td>0.0%</td>", html)

    def test_custom_tolerance_changes_hits(self):
        rows = [estimates(0.046, 0.034, 0.04)]
        self.assertEqual(accuracy_table(rows, 0.04).count("<td>100.0%</td>"), 1)
        self.assertEqual(accuracy_table(rows, 0.04, tolerance=0.20).count("<td>100.0%</td>"), 3)

    def test_invalid_scoring_inputs(self):
        rows = [estimates(0.04, 0.04, 0.04)]
        for tolerance in (-0.1, 1.1, float("nan")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                accuracy_table(rows, 0.04, tolerance=tolerance)
        for truth in (-0.01, float("nan"), float("inf")):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                accuracy_table(rows, truth)
        with self.assertRaises(ValueError):
            accuracy_table([{"close": "100"}], 0.04)
        with self.assertRaises(ValueError):
            accuracy_table([estimates("nan", 0.04, 0.04)], 0.04)


if __name__ == "__main__":
    unittest.main()
