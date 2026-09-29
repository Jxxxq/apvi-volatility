import argparse
import csv
import math
import webbrowser
from pathlib import Path


def render_chart(rows):
    if not rows:
        raise ValueError("the CSV needs at least one candle")
    for hour, high, low, opening, close in rows:
        if any(not math.isfinite(price) or price <= 0 for price in (high, low, opening, close)):
            raise ValueError("prices must be finite and positive")
        if not low <= min(opening, close) <= max(opening, close) <= high:
            raise ValueError("high and low must contain the open and close")
    if any(current[0] != previous[0] + 1 for previous, current in zip(rows, rows[1:])):
        raise ValueError("candles must be one hour apart")

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

    for index, (hour, high, low, opening, close) in enumerate(rows):
        center = x(index)
        left, right = center - body_width / 2, center + body_width / 2
        color = "#16803c" if close >= opening else "#c62828"
        elements.append(f'<g class="candle" stroke="{color}" fill="{color}">')
        elements.append(f'<title>Hour {hour}: H {high:.4f}, L {low:.4f}, O {opening:.4f}, C {close:.4f}</title>')
        elements.append(f'<line x1="{center:.2f}" y1="{y(high):.2f}" x2="{center:.2f}" y2="{y(low):.2f}"/>')
        top, bottom = y(max(opening, close)), y(min(opening, close))
        if opening == close:
            elements.append(f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{right:.2f}" y2="{top:.2f}"/>')
        else:
            elements.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{body_width:.2f}" height="{bottom - top:.2f}"/>')
        elements.append('</g>')

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>1h candles</title>
  <style>
    body {{ margin: 16px; font: 14px Arial, sans-serif; }}
    .chart {{ overflow-x: auto; }}
    svg {{ display: block; }}
    svg text {{ fill: #333; font: 12px Arial, sans-serif; }}
  </style>
</head>
<body>
  <p>1h candles &middot; {len(rows)} hours &middot; green: up, red: down</p>
  <div class="chart">
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="410" viewBox="0 0 {width} 410" role="img" aria-labelledby="chart-title">
      <title id="chart-title">{len(rows)} hourly candles with high, low, open, and close prices</title>
      <text x="76" y="14">Price</text>
      {''.join(elements)}
      <text x="{76 + plot_width / 2}" y="398" text-anchor="middle">Hour</text>
    </svg>
  </div>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=Path("data/market.csv"), help="candle CSV (default: data/market.csv)")
    parser.add_argument("--output", type=Path, default=Path("data/market.html"), help="HTML destination (default: data/market.html)")
    parser.add_argument("--open", action="store_true", help="open the chart in your browser")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    try:
        with args.input.open(newline="", encoding="utf-8-sig") as source:
            rows = [(int(row["hour"]), *(float(row[key]) for key in ("high", "low", "open", "close")))
                    for row in csv.DictReader(source)]
        html = render_chart(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.error(f"{error}. Expected hour, high, low, open, close columns; generate a new CSV with python market.py")

    print(f"Saved chart to {args.output.resolve()}")
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
