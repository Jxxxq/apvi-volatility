# Astronomy-Inspired Method for Measuring Financial Volatility

This senior capstone research project explores whether an astronomy-inspired
method can help identify periods of high financial volatility. The proposed
method combines absolute close-to-close returns and high-low price ranges
and will be compared with standard volatility measures.

The approach will be tested on synthetic market data first, followed by
historical SPY data. The current code includes a synthetic market generator,
candlestick visualizer, volatility baselines, the proposed paired index,
and a reproducible comparison on known low/high regimes.

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
with log-return standard deviation `daily volatility / sqrt(60)`. It derives
the daily open, high, low, and close from this path. Volatility stays constant
unless `--high-volatility` is given. Each day opens at the previous close,
with no separate overnight jump. Day indices start at 0 and count trading
observations, not calendar dates; weekends and holidays are not simulated.

For known low/high periods, use `--volatility` for the low daily volatility,
`--high-volatility` for the high daily volatility, and `--regime-days` for the
length of each period:

```powershell
python market.py --days 500 --volatility 0.01 --high-volatility 0.04 --regime-days 100 --seed 42 --output data/market_regimes.csv
python baselines.py data/market_regimes.csv --output data/baselines_regimes.csv
```

This starts with 100 low-volatility days, then alternates 100 high and 100 low
days. The market CSV adds `regime` (`low` or `high`) and `true_volatility` for
each day. These are simulation labels, not observed market volatility. The
baseline calculator accepts the CSV but outputs only prices and estimates.
Use `paired.py` below to retain the labels alongside all the estimates, or
`compare.py` to run a detection comparison. The candlestick chart's single
`--true-volatility` accuracy table applies only to constant-volatility runs;
omit that option for a regime run.

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

Python 3.10+ is sufficient for calculations; no third-party packages are required.
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
Their performance depends on the simulation and evaluation described below.

## Paired index

```powershell
python paired.py data/market_regimes.csv --train-days 200 --output data/paired_regimes.csv
```

This writes prices, baseline estimates, the paired index, and comparison
controls to one CSV. It preserves `regime` and `true_volatility` when supplied.
The settings JSON records the fitted reference scales. `--train-days` counts
initial candles, including the first candle whose close-to-close return is
unavailable. Only rows with both signals in this training prefix are used
to fit the means and sample standard deviations (`ddof=1`).

The two daily signals are

$$
a_t = |\ln(C_t / C_{t-1})|, \qquad
b_t = \frac{\ln(H_t / L_t)}{\sqrt{4\ln 2}}.
$$

Using fixed training means and standard deviations, define

$$
z^a_t = \frac{a_t - \mu_a}{s_a}, \qquad
z^b_t = \frac{b_t - \mu_b}{s_b}, \qquad
u_t = \max(z^a_t, 0), \quad v_t = \max(z^b_t, 0).
$$

The proposed astronomy-inspired paired-variability index (APVI) is

$$
\mathrm{APVI}_t = \lambda\mathrm{APVI}_{t-1}
                  + (1-\lambda)\sqrt{u_t v_t}, \qquad \lambda=0.94.
$$

Clipping each standardized signal before multiplication means that two
below-average signals do not produce a positive daily contribution. This is
the project's proposed adaptation, not the original astronomical Stetson J.
APVI is a dimensionless detection score, not an estimate of daily volatility.

The main control replaces `sqrt(u * v)` with `(u + v) / 2`. Both methods use
identical training scales, clipping, and smoothing. Additional controls smooth
`u` alone, `v` alone, and the unrectified average `(return_z + range_z) / 2`.
Every processed score starts at zero before the first paired observation and
updates through training, validation, and testing without resetting. Training
scores use fitted training statistics and are for initialization only; they
must not be interpreted as out-of-sample results. A zero training standard
deviation is rejected.

The output also includes EWMA Parkinson volatility: initialize variance with
the mean of the first W daily `b_t^2` values, then update it with
`variance_t = decay * variance_previous + (1-decay) * b_t^2` and take its square
root. It uses the same decay as APVI. The original three baselines retain the
definitions above.

## Comparing high-volatility detection

```powershell
python compare.py
```

The default experiment runs 20 independent random seeds (42 through 61).
Each run has 1,200 daily candles alternating between 1% and 4% daily volatility
every 100 days. It uses these chronological splits:

| Days | Purpose |
| --- | --- |
| 0–399 | Fit the paired signals' reference scales and initialize the measures |
| 400–799 | Choose each method's alert threshold |
| 800–1199 | Evaluate every method on the same 400 untouched test days |

The target is the known high-volatility state on the current day, using prices
available at that day's close. This evaluates detection, not future volatility
forecasting. Validation and test periods must each contain both regimes.
The default test prevalence is 50% high-volatility days, with two high episodes
per seed. All methods use a 20-day baseline window and/or EWMA decay 0.94 as
applicable. There is no parameter search on the test results.

Each method gets its own threshold because their scales differ. On validation
low-state days, sort its scores, allow `floor(0.05 * number_of_low_days)` false
alarms, and choose the smallest observed threshold that meets that limit with
the strict alarm rule `score > threshold`. Ties can make the actual validation
false-positive rate smaller than 5%. Freeze the thresholds before testing and
report the actual test false-positive rates.

The main ranking metric is **average precision (AP)**, computed from continuous
scores by summing precision weighted by increases in recall, with equal scores
grouped together. Higher is better. The primary comparison is APVI minus the
matched clipped simple average on each seed. The output also records ROC-AUC
(ties count as half a win), precision, recall, F1, false-positive rate, and
confusion counts. These are detection metrics; the constant-volatility
within-10% score is not used for this experiment.

Detection delay counts trading days from a high episode's start to its first
alert; an alert on the first day has delay zero. Episodes already underway at
the test boundary are excluded from delay calculations. Missed episodes are
counted separately, and mean delay includes only detected episodes. Undefined
metrics are blank in CSVs. Summaries average available per-seed metrics and
record their counts and sample standard deviations; days are not treated as
independent repetitions.

Outputs go to `data/comparison/`:

- `scores_seed_*.csv`: prices, true labels, all scores, split assignments, and alerts.
- `metrics.csv`: test results and validation thresholds for each method and seed.
- `summary.csv`: means and standard deviations across seeds.
- `settings.json`: simulation settings, fitted scales, thresholds, and paired AP differences.

For a standalone comparison figure, install matplotlib and add `--plot`:

```powershell
python -m pip install matplotlib
python compare.py --plot
```

This adds `comparison.png` and `comparison.svg`, showing the first seed's test
responses and average precision across all seeds. Matplotlib is optional;
calculation and CSV output use only the standard library. Use `--help` for the
simulation, split, seed, window, and threshold settings.

To add the saved results to the candlestick page:

```powershell
python visualize.py --comparison data/comparison --output visual.html
```

The page keeps the original candlestick chart and baseline formulas, adding a
small table for the five main methods and the paired formula. Candles show the
first seed's test days; the table shows means across all seeds. Precision is
the fraction of alerts that were correct, recall is the fraction of high days
detected, and the false-alarm rate is the fraction of low days incorrectly
flagged. Formatting the formulas uses MathJax and requires internet access,
as in the original chart. Use `--output data/market.html` to refresh that page.

### Initial synthetic results

With the defaults above, the 20-seed means are:

| Method | Average precision | Recall | Test false-positive rate |
| --- | ---: | ---: | ---: |
| Paired index (APVI) | 0.935 | 71.8% | 5.5% |
| Matched clipped simple average | 0.940 | 75.0% | 5.3% |
| Rolling standard deviation | 0.931 | 71.7% | 6.8% |
| EWMA volatility | 0.926 | 69.9% | 6.2% |
| Parkinson volatility | 0.958 | 83.0% | 5.5% |
| EWMA Parkinson | 0.947 | 78.4% | 5.5% |
| Processed return only | 0.932 | 70.9% | 5.8% |
| Processed range only | 0.945 | 77.0% | 5.4% |
| Unrectified simple average | 0.951 | 79.0% | 5.5% |

APVI minus the matched average's AP was -0.00514 on average, with a sample
standard deviation of 0.00461 across seeds. APVI had higher AP in 1 of 20 runs.
The paired index therefore shows no advantage over its matched arithmetic
control in this experiment. Its mean detection delay was 20.3 days, versus
19.9 for the matched average and 14.3 for Parkinson; none missed a full test
episode in these runs.

These results cover one deterministic block design with long regimes, no
overnight gaps, and no separate shocks or measurement contamination. They do
not establish performance on real markets. The discrete 60-step intraday
simulation also limits how closely observed ranges capture continuous-path
extremes. Shorter regimes, other scenarios, and historical evaluation remain
future work; these initial results do not justify modifying the formula based
on test performance.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The checks cover hand-calculated values, initialization and window alignment,
constant prices, price-scale invariance, future-data independence, and
invalid inputs, plus daily simulation scaling and the full daily CSV workflow.
Accuracy checks cover tolerance boundaries, shared evaluation days, zero
volatility, and invalid inputs. Paired-index checks cover hand-calculated
standardization, clipping before multiplication, smoothing, scale invariance,
and independence from future observations. Comparison checks cover tied
ranking scores, threshold selection, missed episodes, chronological splits,
and both CLI workflows.
