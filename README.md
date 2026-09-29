# Astronomy-Inspired Method for Measuring Financial Volatility

This senior capstone research project explores whether an astronomy-inspired
method can help identify periods of high financial volatility. The proposed
method combines absolute close-to-close returns and high-low price ranges
and will be compared with standard volatility measures.

The approach will be tested on synthetic market data first, followed by
historical SPY data. The current code includes a basic synthetic market
generator, candlestick visualizer, and three volatility baselines.

## Generating daily data

Each candle represents one synthetic trading day. `--days 500` produces 500
candles, and `--volatility 0.04` sets a daily log-return standard deviation of
4%. Defaults are 500 trading days, volatility 0.01, and seed 42.

```powershell
python market.py --days 500 --volatility 0.04 --seed 42 --output data/market_500.csv
python baselines.py data/market_500.csv --output data/baselines_500.csv
python visualize.py data/baselines_500.csv --true-volatility 0.04 --output data/market_500.html --open
```

The generator uses 60 equally spaced intraday steps per trading day, each
with log-return standard deviation `volatility / sqrt(60)`. It derives the
daily open, high, low, and close from this path. Volatility stays constant
throughout a run, and each day opens at the previous close, with no separate
overnight jump. Day indices start at 0 and count trading observations, not
calendar dates; weekends and holidays are not simulated.

Existing CSVs without a `day` column must be regenerated before use.
The chart includes the baseline formulas; displaying the LaTeX requires an
internet connection for MathJax.

## Baseline accuracy in the chart

Give the visualizer your **baseline CSV** and the volatility you used to
generate the market:

```powershell
python visualize.py data/baselines_500.csv --true-volatility 0.04 --output data/market_500.html --open
```

The chart reads the saved estimates and counts how many are within **10%**
of that value: `0.036` through `0.044` for a target of `0.04`. Each rate is
`correct / evaluated * 100`. All methods use the same rows with complete
estimates, giving 480 evaluated days for a 500-day sample with a 20-day window.
No available rows means `N/A`; zero true volatility requires zero estimates.
Use `--tolerance 0.20` for a 20% tolerance. Without `--true-volatility`, the
visualizer shows only prices and formulas. This score checks estimation
closeness on a constant-volatility simulation.

## Running the baselines

Python 3.10+ is sufficient; no third-party packages are required.
To calculate all three measures on the existing dataset:

```powershell
python baselines.py data/market.csv
```

This creates `data/baselines.csv` with the original daily OHLC prices plus
`log_return`, `rolling_std`, `ewma_volatility`, and `parkinson_volatility`.
The corresponding `data/baselines.settings.json` records the input path,
window lengths, decay, initialization, units, and alignment for that run.

To generate a longer sample and use custom settings:

```powershell
python market.py --days 500 --seed 42 --output data/market_500.csv
python baselines.py data/market_500.csv --window 20 --decay 0.94 --output data/baselines_500.csv
```

Input must have `day,high,low,open,close` columns, positive finite prices,
valid OHLC bounds, and consecutive trading-day indices. The calculator
rejects invalid data rather than sorting or filling missing trading days.

## Baseline definitions and settings

All three measures use the same input candles and are available **at the close
of each candle**, using only that candle and earlier observations. Let
`r_t = ln(C_t / C_(t-1))` and `d_t = ln(H_t / L_t)`, using natural logarithms.

| Measure | Calculation | Default settings |
| --- | --- | --- |
| Rolling standard deviation | Sample standard deviation of the last W close-to-close log returns, including the current return | W = 20 daily returns; subtract the window mean; variance denominator W - 1 (`ddof=1`) |
| EWMA volatility | `sqrt(v_t)`, where `v_t = decay * v_(t-1) + (1-decay) * r_t^2` after initialization | Decay = 0.94; current squared-return weight = 0.06; assume zero mean |
| Parkinson volatility | `sqrt(mean(d_t^2 / (4 * ln(2))))` over the last W candles | W = 20 daily candles; average variances before taking the square root |

`--window` controls both rolling windows and the EWMA initialization length.
It must be an integer of at least 2. `--decay` must be strictly between 0 and 1.
EWMA starts with the **mean of the first W squared returns**, published only
when those returns are available. It then updates recursively using all
subsequent returns; it is not truncated to a rolling W-observation window.
There is no mean subtraction or sample-variance correction in EWMA.

The first close-to-close return is unavailable because the CSV does not
contain the preceding close; the first opening price is not substituted.
Rolling SD and EWMA therefore first appear on candle W + 1 (day 20 for a
zero-based, 20-return window). Parkinson first appears on candle W (day 19).
Earlier values are blank, not zero. Short datasets are allowed and retain
blanks for any measure without enough history. Compare the methods only on
rows where all three are available.

Outputs are **nonannualized decimal volatility per trading day**: `0.01`
means 1% daily volatility on the log-return scale. A 20-observation window
means 20 trading days. The decay of 0.94 is a fixed daily project setting,
not a value fitted to this synthetic sample. Historical SPY data will need
consistent price adjustments and a date-to-trading-index conversion.

Parkinson uses only each candle's high-low range, so it does not capture gaps
between the previous close and current open. Its Brownian-motion motivation
does not make it exact for finite simulated paths, jumps, or real markets.
These outputs are initial baselines, not evidence that any method performs
better than another.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The checks cover hand-calculated values, initialization and window alignment,
constant prices, price-scale invariance, future-data independence, and
invalid inputs, plus daily simulation scaling and the full daily CSV workflow.
Accuracy checks cover tolerance boundaries, shared evaluation days, zero
volatility, and invalid inputs.
