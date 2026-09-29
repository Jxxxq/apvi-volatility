# Astronomy-Inspired Method for Measuring Financial Volatility

This senior capstone research project explores whether an astronomy-inspired
method can help identify periods of high financial volatility. The proposed
method combines absolute close-to-close returns and high-low price ranges
and will be compared with standard volatility measures.

The approach will be tested on synthetic market data first, followed by
historical SPY data. The current code includes a basic synthetic market
generator, candlestick visualizer, and three volatility baselines.

## Running the baselines

Python 3.10+ is sufficient; no third-party packages are required.
To calculate all three measures on the existing dataset:

```powershell
python baselines.py data/market.csv
```

This creates `data/baselines.csv` with the original hourly OHLC prices plus
`log_return`, `rolling_std`, `ewma_volatility`, and `parkinson_volatility`.
The corresponding `data/baselines.settings.json` records the input path,
window lengths, decay, initialization, units, and alignment for that run.

To generate a longer sample and use custom settings:

```powershell
python market.py --hours 500 --seed 42 --output data/market_500.csv
python baselines.py data/market_500.csv --window 20 --decay 0.94 --output data/baselines_500.csv
```

Input must have `hour,high,low,open,close` columns, positive finite prices,
valid OHLC bounds, and chronological candles exactly one hour apart. The
calculator rejects invalid data rather than sorting or filling missing hours.

## Baseline definitions and settings

All three measures use the same input candles and are available **at the close
of each candle**, using only that candle and earlier observations. Let
`r_t = ln(C_t / C_(t-1))` and `d_t = ln(H_t / L_t)`, using natural logarithms.

| Measure | Calculation | Default settings |
| --- | --- | --- |
| Rolling standard deviation | Sample standard deviation of the last W close-to-close log returns, including the current return | W = 20 returns; subtract the window mean; variance denominator W - 1 (`ddof=1`) |
| EWMA volatility | `sqrt(v_t)`, where `v_t = decay * v_(t-1) + (1-decay) * r_t^2` after initialization | Decay = 0.94; current squared-return weight = 0.06; assume zero mean |
| Parkinson volatility | `sqrt(mean(d_t^2 / (4 * ln(2))))` over the last W candles | W = 20 candles; average variances before taking the square root |

`--window` controls both rolling windows and the EWMA initialization length.
It must be an integer of at least 2. `--decay` must be strictly between 0 and 1.
EWMA starts with the **mean of the first W squared returns**, published only
when those returns are available. It then updates recursively using all
subsequent returns; it is not truncated to a rolling W-observation window.
There is no mean subtraction or sample-variance correction in EWMA.

The first close-to-close return is unavailable because the CSV does not
contain the preceding close; the first opening price is not substituted.
Rolling SD and EWMA therefore first appear on candle W + 1 (hour 20 for a
zero-based, 20-return window). Parkinson first appears on candle W (hour 19).
Earlier values are blank, not zero. Short datasets are allowed and retain
blanks for any measure without enough history. Compare the methods only on
rows where all three are available.

Outputs are **nonannualized decimal volatility per hour**: `0.01` means 1%
on the log-return scale. The generator produces hourly candles even when
called with `--days`; a 20-observation window currently means 20 hours,
not 20 trading days. The decay of 0.94 is a fixed project setting, not a value
fitted to this synthetic sample. Daily SPY analysis will require daily data
handling and an explicit review of the settings.

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
invalid inputs.
