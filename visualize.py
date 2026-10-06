import argparse
import csv
import itertools
import json
import math
import webbrowser
from html import escape
from pathlib import Path

from compare import METHODS


def comparison_chart(records, series, chart_id, percent=False, thresholds=None):
    width, height, left, top, plot_width, plot_height = 1000, 290, 65, 18, 910, 222
    values = [float(row[column]) for row in records for column, _, _ in series]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("chart values must be finite")
    thresholds = thresholds or {}
    maximum = max(values + list(thresholds.values()) + [0.001]) * 1.1
    spacing = plot_width / len(records)

    def x(index):
        return left + (index + 0.5) * spacing

    def y(value):
        return top + plot_height * (1 - value / maximum)

    elements = []
    for regime, group in itertools.groupby(enumerate(records), key=lambda item: item[1]["regime"]):
        block = list(group)
        if regime == "high":
            elements.append(f'<rect x="{left + block[0][0] * spacing:.2f}" y="{top}" '
                            f'width="{len(block) * spacing:.2f}" height="{plot_height}" fill="#e7f0fa"/>')
    for tick in range(5):
        value = maximum * tick / 4
        label = f"{value:.1%}" if percent else f"{value:.2f}"
        elements.append(f'<line x1="{left}" y1="{y(value):.2f}" x2="{left + plot_width}" y2="{y(value):.2f}" stroke="#d9e1ea"/>')
        elements.append(f'<text x="{left - 10}" y="{y(value) + 4:.2f}" text-anchor="end">{label}</text>')
    for index in sorted({round((len(records) - 1) * tick / 4) for tick in range(5)}):
        elements.append(f'<text x="{x(index):.2f}" y="263" text-anchor="middle">{int(records[index]["day"])}</text>')
    legend = []
    for column, label, color in series:
        points = []
        for index, row in enumerate(records):
            value = float(row[column])
            if column == "true_volatility":
                points.extend((f"{left + index * spacing:.2f},{y(value):.2f}",
                               f"{left + (index + 1) * spacing:.2f},{y(value):.2f}"))
            else:
                points.append(f"{x(index):.2f},{y(value):.2f}")
        elements.append(f'<g data-series="{column}"><polyline points="{" ".join(points)}" '
                        f'fill="none" stroke="{color}" stroke-width="2"/>')
        if column in thresholds:
            position = y(thresholds[column])
            elements.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{position:.2f}" y2="{position:.2f}" '
                            f'stroke="{color}" stroke-dasharray="7 5"><title>{escape(label)} threshold: {thresholds[column]:.4f}</title></line>')
        elements.append('</g>')
        legend.append(f'<label style="--series-color:{color}"><input type="checkbox" checked '
                      f'data-chart="{chart_id}" data-toggle="{column}"> {escape(label)}</label>')
    return f'''<div class="legend">{''.join(legend)}</div>
    <svg id="{chart_id}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(', '.join(label for _, label, _ in series))}">
    {''.join(elements)}<text x="520" y="285" text-anchor="middle">Trading day</text></svg>'''


def render_comparison(folder):
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    with (folder / "summary.csv").open(newline="", encoding="utf-8") as source:
        summary = {row["method"]: row for row in csv.DictReader(source)}
    seed = int(settings["seed"])
    with (folder / f"scores_seed_{seed}.csv").open(newline="", encoding="utf-8") as source:
        records = [row for row in csv.DictReader(source) if row["split"] == "test"]
    if not records or not set(METHODS).issubset(summary):
        raise ValueError("comparison needs test scores and a summary for every method")
    fit = next(item for item in settings["fits"] if item["seed"] == seed)
    test_days = int(settings["days"]) - int(settings["train_days"]) - int(settings["validation_days"])
    if len(records) != test_days:
        raise ValueError("saved test scores do not match the comparison settings")

    def metric(method, name, format_spec=".3f"):
        value = summary[method][name + "_mean"]
        return format(float(value), format_spec) if value else "N/A"

    best = max(METHODS, key=lambda method: float(summary[method]["average_precision_mean"]))
    difference = float(settings["paired_ap_difference_mean"])
    direction = "higher than" if difference > 0 else "lower than" if difference < 0 else "equal to"
    table_rows = []
    for method, label in METHODS.items():
        spread = summary[method]["average_precision_sd"]
        spread_text = f'<span class="muted"> &plusmn; {float(spread):.3f}</span>' if spread else ""
        cells = [escape(label), metric(method, "average_precision") + spread_text,
                 metric(method, "precision", ".1%"), metric(method, "recall", ".1%"),
                 metric(method, "false_positive_rate", ".1%"), metric(method, "mean_delay_days", ".1f")]
        table_rows.append(f'<tr class="{"featured" if method == "apvi" else ""}"><th scope="row">{cells[0]}</th>' +
                          ''.join(f'<td>{value}</td>' for value in cells[1:]) + '</tr>')
    volatility = comparison_chart(records, [("true_volatility", "True daily volatility", "#172a3a"),
                                             ("rolling_std", "Rolling SD", "#2166ac"),
                                             ("ewma_volatility", "EWMA", "#b84a18"),
                                             ("parkinson_volatility", "Parkinson", "#24805e")], "volatility-chart", percent=True)
    paired = comparison_chart(records, [("apvi", "Paired index (APVI)", "#2166ac"),
                                        ("arithmetic", "Matched simple average", "#b84a18")], "paired-chart",
                              thresholds={key: fit["thresholds"][key] for key in ("apvi", "arithmetic")})
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Volatility comparison</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #f3f5f8; color: #172a3a; font: 16px/1.55 system-ui, sans-serif; }}
  main {{ max-width: 1140px; margin: auto; padding: 38px 24px; }}
  h1 {{ font-size: 36px; line-height: 1.2; margin: 8px 0 16px; }}
  h2 {{ font-size: 23px; margin: 0 0 12px; }}
  h3 {{ font-size: 18px; margin: 0 0 8px; }}
  p {{ margin: 10px 0; }} .muted, .eyebrow {{ color: #566677; }}
  .eyebrow {{ font-size: 13px; letter-spacing: .1em; text-transform: uppercase; }}
  section {{ background: white; margin: 22px 0; padding: 26px; border: 1px solid #dce3eb; border-radius: 12px; }}
  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 24px; }}
  .card {{ background: white; padding: 18px 22px; border-radius: 10px; border-top: 4px solid #2166ac; }}
  .card strong {{ display: block; font-size: 34px; }}
  .table-wrap {{ overflow-x: auto; }} table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ padding: 12px 10px; border-bottom: 1px solid #e4e9ef; text-align: right; white-space: nowrap; }}
  th:first-child {{ text-align: left; }} thead {{ background: #f3f5f8; }} .featured {{ background: #eaf2fc; }}
  svg {{ width: 100%; display: block; }} svg text {{ font: 12px system-ui, sans-serif; fill: #48586b; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 14px; }}
  .legend label {{ cursor: pointer; color: var(--series-color); }} input {{ accent-color: var(--series-color); }}
  .formula {{ overflow-wrap: anywhere; padding: 12px 16px; background: #f3f5f8; border-radius: 6px; font: 15px/1.8 ui-monospace, monospace; }}
  @media (max-width: 650px) {{ main {{ padding: 20px 12px; }} section {{ padding: 16px; }} .cards {{ grid-template-columns: 1fr; }} h1 {{ font-size: 28px; }} }}
</style></head><body><main>
<div class="eyebrow">Astronomy-inspired financial volatility</div>
<h1>Paired index: first simulation results</h1>
<p>The paired index's average precision was <strong>{abs(difference):.4f} {direction}</strong> the matched simple average.
It had higher average precision in <strong>{int(settings['paired_ap_wins'])} of {int(settings['seeds'])} seeds</strong>.
{escape(METHODS[best])} had the highest mean average precision in this experiment.</p>
<div class="cards">
  <div class="card">Paired index AP<strong>{metric('apvi', 'average_precision')}</strong></div>
  <div class="card">Matched average AP<strong>{metric('arithmetic', 'average_precision')}</strong></div>
  <div class="card">{escape(METHODS[best])} AP<strong>{metric(best, 'average_precision')}</strong></div>
</div>
<section><h2>How the comparison was run</h2>
<p><strong>{int(settings['seeds'])} seeds &middot; {int(settings['days']):,} trading days each &middot;
{float(settings['volatility']):.2%} / {float(settings['high_volatility']):.2%} daily volatility &middot;
{int(settings['regime_days'])}-day alternating periods</strong></p>
<p>{int(settings['train_days'])} initial days fit the reference scales;
{int(settings['validation_days'])} validation days set each method's threshold;
the last <strong>{test_days} days per seed</strong> are used for the results below.
Thresholds target at most {float(settings['false_positive_rate']):.0%} false alarms on validation low-volatility days.
Rolling window: {int(settings['window'])} days. EWMA decay: {float(settings['decay']):.2f}.</p>
<p class="muted">Target: identify the current high-volatility state using information available at that day's close.
{sum(row['regime'] == 'high' for row in records) / len(records):.0%} of test days are high volatility in this block design.</p></section>
<section><h2>Test results across all seeds</h2>
<p>Average precision (AP) measures how well the scores rank high-volatility days; higher is better.
Recall is the fraction of high days detected. False-positive rate is the fraction of low days that triggered an alarm.</p>
<div class="table-wrap"><table><thead><tr><th scope="col">Method</th><th scope="col">AP &plusmn; SD</th><th scope="col">Precision</th>
<th scope="col">Recall</th><th scope="col">False alarms</th><th scope="col">Delay (days)</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div>
<p class="muted">Means across seeds; AP spread is the sample standard deviation, not a confidence interval.
Delay is averaged over detected episodes. APVI's missed-episode rate: {metric('apvi', 'missed_episode_rate', '.1%')}.</p></section>
<section><h2>Baseline response &middot; seed {seed}</h2>
<p>Only test days {int(records[0]['day'])}&ndash;{int(records[-1]['day'])} are shown.
Blue shading marks the known high-volatility periods. Toggle a legend item to hide or show a line.</p>{volatility}</section>
<section><h2>Paired index and matched average &middot; seed {seed}</h2>
<p>Both use the same training scales and smoothing. Dashed lines are the thresholds fixed on validation data.
These are dimensionless scores.</p>{paired}</section>
<section><h2>Implemented formula</h2>
<p>Let <code>a = |ln(close / previous close)|</code> and
<code>b = ln(high / low) / sqrt(4 ln 2)</code>. Standardize each using its fixed training mean and sample standard deviation to get
<code>z_a</code> and <code>z_b</code>, then set <code>u = max(z_a, 0)</code> and <code>v = max(z_b, 0)</code>.</p>
<div class="formula">APVI[t] = {float(settings['decay']):.2f} APVI[t-1] + {1-float(settings['decay']):.2f} sqrt(u[t] v[t])<br>
Average[t] = {float(settings['decay']):.2f} Average[t-1] + {1-float(settings['decay']):.2f} (u[t] + v[t]) / 2</div>
<p>Both scores start at zero and carry their state across the splits. Training scores are used for initialization and excluded from evaluation.</p>
<details><summary>Baseline definitions</summary><p>Rolling SD: sample standard deviation of the last W close-to-close log returns.
EWMA: square root of exponentially smoothed squared log returns.
Parkinson: square root of the mean of ln(high / low)&sup2; / (4 ln 2) over W days.
EWMA Parkinson applies exponential smoothing to that daily range variance.</p></details></section>
<section><h2>What these results establish</h2><p>This is an initial comparison on long, regular volatility blocks.
There are no separate shocks, overnight gaps, or measurement errors. Historical SPY testing has not started.
The results do not establish future forecasting performance or performance on real markets.</p></section>
</main><script>
document.querySelectorAll('input[data-toggle]').forEach(input => {{
  input.addEventListener('change', () => {{
    document.querySelector('#' + input.dataset.chart + ' [data-series="' + input.dataset.toggle + '"]').style.display = input.checked ? '' : 'none';
  }});
}});
</script></body></html>'''


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


def render_chart(rows, accuracy=""):
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
