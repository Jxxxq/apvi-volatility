import argparse
import csv
import json
import math
import webbrowser
from html import escape
from pathlib import Path

from compare import METHODS


def render_comparison(folder):
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    with (folder / "summary.csv").open(newline="", encoding="utf-8") as source:
        summary = {row["method"]: row for row in csv.DictReader(source)}
    seed = int(settings["seed"])
    with (folder / f"scores_seed_{seed}.csv").open(newline="", encoding="utf-8") as source:
        records = [row for row in csv.DictReader(source) if row["split"] == "test"]
    shown_methods = ("apvi", "arithmetic", "rolling_std", "ewma_volatility", "parkinson_volatility")
    if not records or not set(shown_methods).issubset(summary):
        raise ValueError("comparison needs test scores and a summary for the five main methods")
    test_days = int(settings["days"]) - int(settings["train_days"]) - int(settings["validation_days"])
    if len(records) != test_days:
        raise ValueError("saved test scores do not match the comparison settings")
    rows = [(int(row["day"]), *(float(row[key]) for key in ("high", "low", "open", "close")))
            for row in records]
    table_rows = []
    for method in shown_methods:
        cells = [escape(METHODS[method])]
        for metric, format_spec in (("average_precision", ".3f"), ("precision", ".1%"),
                                    ("recall", ".1%"), ("false_positive_rate", ".1%")):
            value = summary[method][metric + "_mean"]
            cells.append(format(float(value), format_spec) if value else "N/A")
        table_rows.append('<tr>' + ''.join(f'<td>{value}</td>' for value in cells) + '</tr>')
    comparison = f"""<p>Chart: seed {seed}, test days {int(records[0]['day'])}&ndash;{int(records[-1]['day'])}.
    Low/high daily volatility: {float(settings['volatility']):.1%} / {float(settings['high_volatility']):.1%}, switching every {int(settings['regime_days'])} days.</p>
    <p>Detection results: means across {int(settings['seeds'])} seeds, {int(settings['days']):,} trading days per seed
    ({int(settings['train_days'])} training, {int(settings['validation_days'])} validation, {test_days} test).
    Thresholds were set using validation data to allow at most {float(settings['false_positive_rate']):.0%} false alarms.</p>
    <table><thead><tr><th scope="col">Method</th><th scope="col">Average precision (AP)</th>
    <th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">False-alarm rate</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody></table>
    <p>Precision: how many flagged days were actually high volatility.<br>
    Recall: how many actual high-volatility days were detected.<br>
    False-alarm rate: how many actual low-volatility days were incorrectly flagged.<br>
    AP measures ranking quality across thresholds; higher is better.</p>"""
    paired_formula = r"""<p>Paired index:
    \(I_t = \lambda I_{t-1} + (1-\lambda)\sqrt{u_t v_t}\).</p>
    <p>\(u_t=\max(z^a_t,0)\), \(v_t=\max(z^b_t,0)\), where \(z^a_t\) and \(z^b_t\) standardize
    \(|r_t|\) and \(\ln(H_t/L_t)/\sqrt{4\ln 2}\) using fixed training means and sample standard deviations.
    The matched average uses \((u_t+v_t)/2\) instead of \(\sqrt{u_t v_t}\).</p>"""
    return render_chart(rows, comparison, paired_formula)


def accuracy_table(records, true_volatility, tolerance=0.10):
    if not math.isfinite(true_volatility) or true_volatility < 0:
        raise ValueError("true volatility must be finite and nonnegative")
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be between 0 and 1")
    labels = {
        "rolling_std": "Rolling standard deviation",
        "ewma_volatility": "EWMA volatility",
        "parkinson_volatility": "Parkinson high-low volatility",
    }
    if records and not set(labels).issubset(records[0]):
        raise ValueError("accuracy needs a baseline CSV with rolling_std, ewma_volatility, and parkinson_volatility columns")
    values = [tuple(float(row[column]) for column in labels) for row in records
              if all(row[column] not in (None, "") for column in labels)]
    if any(not math.isfinite(value) or value < 0 for row in values for value in row):
        raise ValueError("baseline estimates must be finite and nonnegative")
    total = len(values)
    limit = tolerance * true_volatility
    table_rows = []
    for index, label in enumerate(labels.values()):
        errors = [abs(row[index] - true_volatility) for row in values]
        correct = sum(error <= limit or math.isclose(error, limit, rel_tol=1e-12, abs_tol=0) for error in errors)
        rate = f"{correct / total:.1%}" if total else "N/A"
        table_rows.append(f"<tr><td>{label}</td><td>{correct} / {total}</td><td>{rate}</td></tr>")
    return f'''<p>Baseline accuracy: correct means within &plusmn;{tolerance:.1%} of {true_volatility:.2%} daily volatility
    ({true_volatility * (1 - tolerance):.4%} to {true_volatility * (1 + tolerance):.4%}).
    All methods use the same {total} days with complete estimates; warm-up days are excluded.</p>
  <table>
    <thead><tr><th scope="col">Baseline</th><th scope="col">Correct / evaluated days</th><th scope="col">Rate</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>'''


def render_chart(rows, accuracy="", extra_formulas=""):
    if not rows:
        raise ValueError("the CSV needs at least one candle")
    for day, high, low, opening, close in rows:
        if any(not math.isfinite(price) or price <= 0 for price in (high, low, opening, close)):
            raise ValueError("prices must be finite and positive")
        if not low <= min(opening, close) <= max(opening, close) <= high:
            raise ValueError("high and low must contain the open and close")
    if any(current[0] != previous[0] + 1 for previous, current in zip(rows, rows[1:])):
        raise ValueError("candles must have consecutive trading-day indices")

    min_price = min(row[2] for row in rows)
    max_price = max(row[1] for row in rows)
    padding = max((max_price - min_price) * 0.05, max_price * 0.001)
    min_price, max_price = min_price - padding, max_price + padding
    plot_width = max(800, len(rows) * 8)
    width = plot_width + 100
    spacing = plot_width / len(rows)
    body_width = min(12, spacing * 0.65)

    def x(index):
        return 76 + (index + 0.5) * spacing

    def y(price):
        return 24 + (max_price - price) / (max_price - min_price) * 320

    elements = []
    for tick in range(5):
        price = min_price + (max_price - min_price) * tick / 4
        position = y(price)
        elements.append(f'<line x1="76" y1="{position:.2f}" x2="{width - 24}" y2="{position:.2f}" stroke="#ddd"/>')
        elements.append(f'<text x="64" y="{position + 4:.2f}" text-anchor="end">{price:.2f}</text>')
    for index in sorted({round((len(rows) - 1) * tick / 4) for tick in range(5)}):
        elements.append(f'<text x="{x(index):.2f}" y="368" text-anchor="middle">{rows[index][0]}</text>')

    for index, (day, high, low, opening, close) in enumerate(rows):
        center = x(index)
        left, right = center - body_width / 2, center + body_width / 2
        color = "#16803c" if close >= opening else "#c62828"
        elements.append(f'<g class="candle" stroke="{color}" fill="{color}">')
        elements.append(f'<title>Day {day}: H {high:.4f}, L {low:.4f}, O {opening:.4f}, C {close:.4f}</title>')
        elements.append(f'<line x1="{center:.2f}" y1="{y(high):.2f}" x2="{center:.2f}" y2="{y(low):.2f}"/>')
        top, bottom = y(max(opening, close)), y(min(opening, close))
        if opening == close:
            elements.append(f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{right:.2f}" y2="{top:.2f}"/>')
        else:
            elements.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{body_width:.2f}" height="{bottom - top:.2f}"/>')
        elements.append('</g>')

    return rf'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daily candles</title>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script>
  <style>
    body {{ margin: 16px; font: 14px Arial, sans-serif; }}
    .chart {{ overflow-x: auto; }}
    svg {{ display: block; }}
    svg text {{ fill: #333; font: 12px Arial, sans-serif; }}
    th, td {{ padding: 4px 16px 4px 0; text-align: left; }}
  </style>
</head>
<body>
  <p>Daily candles &middot; {len(rows)} trading days &middot; green: up, red: down</p>
  <div class="chart">
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="410" viewBox="0 0 {width} 410" role="img" aria-labelledby="chart-title">
      <title id="chart-title">{len(rows)} daily candles with high, low, open, and close prices</title>
      <text x="76" y="14">Price</text>
      {''.join(elements)}
      <text x="{76 + plot_width / 2}" y="398" text-anchor="middle">Trading day</text>
    </svg>
  </div>
  {accuracy}
  <p>Baseline formulas (\(W\): window in trading days; \(\lambda\): EWMA decay).</p>
  <p>Rolling standard deviation:
    \(\hat{{\sigma}}_{{t,\mathrm{{SD}}}} = \sqrt{{\frac{{1}}{{W-1}}\sum_{{j=0}}^{{W-1}}(r_{{t-j}}-\bar{{r}}_{{t,W}})^2}}\)</p>
  <p>EWMA volatility:
    \(v_t = \lambda v_{{t-1}} + (1-\lambda)r_t^2,\quad \hat{{\sigma}}_{{t,\mathrm{{EWMA}}}} = \sqrt{{v_t}}\)</p>
  <p>Parkinson high-low volatility:
    \(\hat{{\sigma}}_{{t,\mathrm{{P}}}} = \sqrt{{\frac{{1}}{{4W\ln 2}}\sum_{{j=0}}^{{W-1}}\left[\ln\left(\frac{{H_{{t-j}}}}{{L_{{t-j}}}}\right)\right]^2}}\)</p>
  <p>\(r_t=\ln(C_t/C_{{t-1}})\) and \(\bar{{r}}_{{t,W}}=\frac{{1}}{{W}}\sum_{{j=0}}^{{W-1}}r_{{t-j}}\).
    \(C_t\), \(H_t\), and \(L_t\) are the close, high, and low prices.</p>
  {extra_formulas}
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, help="daily market or baseline CSV (default: data/market.csv)")
    parser.add_argument("--comparison", type=Path, help="folder of saved compare.py results")
    parser.add_argument("--output", type=Path, help="HTML destination (default: visual.html for comparisons, data/market.html for candles)")
    parser.add_argument("--open", action="store_true", help="open the chart in your browser")
    parser.add_argument("--true-volatility", type=float, help="known simulated daily volatility for scoring the baseline CSV")
    parser.add_argument("--tolerance", type=float, default=0.10, help="relative error allowed for a correct estimate (default: 0.10 for 10%%)")
    args = parser.parse_args()

    if args.comparison and (args.input is not None or args.true_volatility is not None):
        parser.error("--comparison uses saved results; omit the input CSV and --true-volatility")
    args.output = args.output or Path("visual.html" if args.comparison else "data/market.html")
    args.input = args.input or Path("data/market.csv")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.comparison and args.output.suffix.lower() != ".html":
        parser.error("comparison output must be an .html file")
    try:
        if args.comparison:
            html = render_comparison(args.comparison)
        else:
            with args.input.open(newline="", encoding="utf-8-sig") as source:
                records = list(csv.DictReader(source))
            rows = [(int(row["day"]), *(float(row[key]) for key in ("high", "low", "open", "close")))
                    for row in records]
            accuracy = accuracy_table(records, args.true_volatility, args.tolerance) if args.true_volatility is not None else ""
            html = render_chart(rows, accuracy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, StopIteration, csv.Error) as error:
        parser.error(str(error))

    print(f"Saved chart to {args.output.resolve()}")
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
