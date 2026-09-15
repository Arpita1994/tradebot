# MCX Order-Block Backtester

A Streamlit backtesting tool for Indian MCX commodities (Gold, Crude Oil, etc.)
built around a volume-based order block strategy, with all the inputs you asked for:

1. **Time of day for algo trade** - session start/end time window
2. **Volume going up** - current bar's volume vs. its recent rolling average, by a spike multiplier
3. **Order block** - volume-based order block detection (see definition below)
4. **Date range** - backtest period
5. **Lookback time** - bars used for the volume average and order-block detection window
6. **% of order block used as SL** - stop loss placed this % of the OB's range beyond the block
7. **Trend analysis with N lookback candles** - trades only taken with the prevailing trend

A **reward:risk multiple** (default 2:1) is also exposed since your list defines
entries and stops but not a target - pick whatever the strategy needs.

## Strategy definition

**Order block (volume-based):**
- *Bullish OB*: the last down-candle (close < open) with volume > `ob_spike_multiplier`
  x its trailing average volume, immediately followed within `displacement_window`
  bars by a close above that candle's high (an impulsive move away).
- *Bearish OB*: mirror image (up-candle + spike + impulsive move below its low).
- An OB becomes tradeable the first time price later re-enters its high-low range
  ("mitigation").

**Entry:** when an order block is mitigated, AND that bar falls inside your time
window, AND volume is "going up" (above its recent average by the volume
multiplier), AND (optionally) the trend from the lookback-candle trend filter
agrees with the OB's direction.

**Stop loss:** placed beyond the order block's far edge by `sl_pct_of_ob`% of the
block's high-low range.

**Target:** `reward_risk` x the initial risk (entry price - SL).

This is one reasonable, explicit interpretation of "order block" - it's coded in
`core/order_block.py` if you want to adjust the exact rules (e.g. require a fair
value gap, change how displacement is confirmed, etc.).

## Quick start (no API needed)

```bash
pip install -r requirements.txt
python data/make_sample_data.py     # only needed once, regenerates the demo CSV
streamlit run app.py
```

In the app, pick **"Use sample data"** (or **"Upload CSV"** with your own file:
columns `datetime, open, high, low, close, volume`) and click **Run backtest**.

## Getting real MCX data (free, via Fyers)

Credentials live in a local `.env` file only -- never pasted into the app UI,
never committed to git (it's in `.gitignore`), never sent anywhere by this code
except to Fyers itself during login.

**One-time setup:**
1. Copy `.env.example` to `.env`.
2. Fill in `FYERS_APP_ID` and `FYERS_SECRET_KEY` from your Fyers API app
   (myapi.fyers.in/dashboard), and `FYERS_REDIRECT_URI` with whatever redirect
   URL you registered for that app (e.g. `https://127.0.0.1`).
3. `pip install -r requirements.txt` (includes `fyers-apiv3` and `python-dotenv`).

**Every trading day (tokens expire ~24h):**
```bash
python generate_token.py
```
This opens a Fyers login URL for you to visit in your browser, then asks you to
paste back the redirected URL. It saves the resulting `FYERS_ACCESS_TOKEN`
straight into `.env` -- nothing to copy into the app itself.

**Then:**
```bash
streamlit run app.py
```
In the sidebar, choose **"Fyers API"** as the data source (it auto-detects the
token from `.env`), enter your symbol (e.g. `MCX:GOLD26FEBFUT`,
`MCX:CRUDEOIL26FEBFUT` -- check the current active contract in Fyers' MCX
symbol master, since these roll monthly), a resolution, and a fetch date range,
then click **Fetch from Fyers**.

`fetchers/fyers_fetcher.py` does the actual API call; `core/data_loader.py`
wraps it; `generate_token.py` handles the login. Angel One SmartAPI is a
comparable free alternative if you'd rather use that (you'd add a sibling
fetcher module and one more radio option in `app.py`).

**A gotcha to expect:** Fyers' historical intraday endpoint returns a limited
window per call (check current docs for the exact day cap per request). For
backtests spanning many months you'll need to fetch in chunks and concatenate,
then cache the result locally (CSV/Parquet) so you're not re-fetching every
run -- not yet automated here; ask if you want that built in.

Note: Fyers' historical intraday endpoint returns a limited window per call
(check current docs for the exact day cap per request); for backtests spanning
many months you may need to fetch in chunks and concatenate, then cache the
result locally (e.g. to CSV/Parquet) so you're not re-fetching every run.

## Project layout

```
app.py                     Streamlit UI - wires all inputs together
core/
  data_loader.py           CSV / Fyers loading, date-range & time-of-day filters
  indicators.py            volume_going_up(), trend_direction()
  order_block.py           order block detection + mitigation
  strategy.py               combines everything into entry/SL/target signals
  backtest.py               bar-by-bar trade simulator
  metrics.py                win rate, profit factor, drawdown, equity curve
fetchers/
  fyers_fetcher.py          Fyers API v3 historical-candle wrapper
data/
  make_sample_data.py       generates synthetic demo data (with engineered OBs)
  sample_gold_5min.csv      generated demo data (created by the script above)
```

## Known simplifications (worth knowing before you trust the numbers)

- Each signal is backtested independently - no cap on concurrent open positions
  or margin/lot-size modeling. Add position sizing before using this for real
  capital decisions.
- If a single bar's range contains both the SL and target, the SL is assumed
  hit first (conservative but not always accurate intrabar).
- No brokerage, slippage, or MCX contract-expiry rollover handling yet.
- Trend filter is a simple SMA-level + SMA-slope check, not a full swing
  high/low structure analysis - swap in your own logic in `indicators.py` if
  you want something closer to classic Dow Theory / market-structure trend ID.
