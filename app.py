"""
MCX Order-Block Backtester -- Streamlit app.

Run:
    streamlit run app.py

All seven requested inputs are exposed in the sidebar:
  1. Time of day for algo trade      -> start_time / end_time
  2. Volume going up                 -> volume spike multiplier
  3. Order block                     -> OB volume-spike multiplier + displacement window
  4. Date range                      -> start_date / end_date
  5. Lookback time                   -> lookback_time
  6. % of order block used as SL     -> sl_pct_of_ob
  7. Trend analysis lookback candles -> trend_lookback_candles
"""

import datetime as dt
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.data_loader import load_from_csv, load_from_fyers, filter_date_range

load_dotenv()  # reads .env if present, for local runs (APP_ID / SECRET_KEY / ACCESS_TOKEN)

# When deployed to Streamlit Community Cloud, secrets come from st.secrets
# (set in the app's "Secrets" panel) instead of a local .env file. Pull any
# of these into os.environ so the rest of the app (which reads via
# os.getenv) works unchanged whether running locally or deployed.
for _key in ("FYERS_APP_ID", "FYERS_SECRET_KEY", "FYERS_REDIRECT_URI", "FYERS_ACCESS_TOKEN"):
    if _key in st.secrets:
        os.environ[_key] = str(st.secrets[_key])

from core.strategy import StrategyParams, generate_signals
from core.backtest import run_backtest
from core.metrics import summarize, equity_curve, format_minutes

st.set_page_config(page_title="MCX Order-Block Backtester", layout="wide")
st.title("MCX Commodity Order-Block Backtester")
st.caption(
    "Volume-based order block strategy: a fresh order block, revisited with rising "
    "volume, inside your trading window, aligned with the prevailing trend."
)

# ---------------------------------------------------------------- Data source
st.sidebar.header("1. Data")
source = st.sidebar.radio("Data source", ["Upload CSV", "Use sample data", "Fyers API"])

df = None
if source == "Upload CSV":
    file = st.sidebar.file_uploader(
        "OHLCV CSV (columns: datetime, open, high, low, close, volume)", type=["csv"]
    )
    if file is not None:
        df = load_from_csv(file)

elif source == "Use sample data":
    st.sidebar.caption("Synthetic 5-min MCX-Gold-like data, Jan-Mar 2025 (for demo/testing).")
    df = load_from_csv("data/sample_gold_5min.csv")

else:
    env_app_id = os.getenv("FYERS_APP_ID", "")
    env_token = os.getenv("FYERS_ACCESS_TOKEN", "")

    if env_token:
        st.sidebar.success("Access token loaded from .env")
    else:
        st.sidebar.warning(
            "No token found in .env. Run `python generate_token.py` in your "
            "terminal first (see README.md), then reload this page."
        )

    symbol = st.sidebar.text_input("Symbol", "MCX:GOLD26FEBFUT")
    resolution = st.sidebar.selectbox("Resolution", ["5", "15"], index=0)
    api_start = st.sidebar.date_input("API fetch: start", dt.date(2025, 1, 1))
    api_end = st.sidebar.date_input("API fetch: end", dt.date(2025, 3, 31))
    if st.sidebar.button("Fetch from Fyers", disabled=not env_token):
        try:
            df = load_from_fyers(symbol, resolution, str(api_start), str(api_end),
                                  access_token=env_token, app_id=env_app_id)
            st.sidebar.success(f"Fetched {len(df)} candles.")
        except Exception as e:
            st.sidebar.error(f"Fetch failed: {e}")

if df is None:
    st.info("Choose a data source in the sidebar to begin (try **Use sample data** first).")
    st.stop()

data_min, data_max = df["datetime"].dt.date.min(), df["datetime"].dt.date.max()

# ---------------------------------------------------------------- Strategy inputs
st.sidebar.header("2. Date range")
start_date, end_date = st.sidebar.date_input(
    "Backtest period", value=(data_min, data_max), min_value=data_min, max_value=data_max
)

st.sidebar.header("3. Time of day")
start_time = st.sidebar.time_input("Session start", dt.time(9, 15))
end_time = st.sidebar.time_input("Session end", dt.time(15, 0))

st.sidebar.header("4. Lookback & trend")
lookback_time = st.sidebar.number_input("Lookback time (bars, for volume avg & OB window)", 5, 200, 20)
trend_lookback_candles = st.sidebar.number_input("Trend analysis lookback (candles)", 5, 200, 20)
require_trend = st.sidebar.checkbox("Only trade in direction of trend", value=True)

st.sidebar.header("5. Volume")
volume_spike_multiplier = st.sidebar.slider("Volume going up: spike multiplier (x avg)", 1.0, 3.0, 1.2, 0.1)

st.sidebar.header("6. Order block")
ob_spike_multiplier = st.sidebar.slider("Order block volume spike multiplier (x avg)", 1.0, 4.0, 1.5, 0.1)
displacement_window = st.sidebar.number_input("Displacement confirmation window (bars)", 1, 10, 3)
min_touches = st.sidebar.number_input(
    "Minimum touches before entry", 1, 5, 1,
    help="1 = enter on the first touch of the order block zone (classic ICT "
         "'first mitigation' entry). >1 = require the zone to be tested and "
         "hold that many times first (support/resistance-style confirmation) "
         "before entering -- fewer, later, more-confirmed signals."
)

st.sidebar.header("7. Stop loss")
sl_pct_of_ob = st.sidebar.slider("SL: % of order block range beyond the OB", 0, 200, 50, 5)

st.sidebar.header("8. Target (not in your list -- default provided)")
reward_risk = st.sidebar.slider("Reward:Risk multiple", 0.5, 5.0, 2.0, 0.5)

# ---------------------------------------------------------------- Run
if st.sidebar.button("Run backtest", type="primary"):
    bt_df = filter_date_range(df, start_date, end_date)
    if bt_df.empty:
        st.error("No data in the selected date range.")
        st.stop()

    params = StrategyParams(
        start_time=start_time,
        end_time=end_time,
        lookback_time=int(lookback_time),
        volume_spike_multiplier=float(volume_spike_multiplier),
        ob_spike_multiplier=float(ob_spike_multiplier),
        displacement_window=int(displacement_window),
        sl_pct_of_ob=float(sl_pct_of_ob),
        trend_lookback_candles=int(trend_lookback_candles),
        reward_risk=float(reward_risk),
        require_trend_alignment=require_trend,
        min_touches=int(min_touches),
    )

    signals = generate_signals(bt_df, params)
    trades = run_backtest(bt_df, signals)
    stats = summarize(trades)

    st.subheader("Summary")
    cols = st.columns(4)
    cols[0].metric("Total trades", stats["total_trades"])
    cols[1].metric("Win rate", f"{stats['win_rate_pct']}%")
    cols[2].metric("Total P&L (points)", stats["total_pnl_points"])
    cols[3].metric("Profit factor", stats["profit_factor"])

    cols2 = st.columns(4)
    cols2[0].metric("Avg P&L / trade", stats.get("avg_pnl_points_per_trade"))
    cols2[1].metric("Avg R multiple", stats["avg_r_multiple"])
    cols2[2].metric("Max drawdown (points)", stats["max_drawdown_points"])
    cols2[3].metric("Longs / Shorts", f"{stats.get('longs', 0)} / {stats.get('shorts', 0)}")

    cols3 = st.columns(4)
    cols3[0].metric("Avg trade duration", format_minutes(stats.get("avg_holding_time_minutes")))
    cols3[1].metric("Median trade duration", format_minutes(stats.get("median_holding_time_minutes")))
    cols3[2].metric("Longest trade", format_minutes(stats.get("max_holding_time_minutes")))

    if not trades.empty:
        st.subheader("Equity curve (cumulative points)")
        ec = equity_curve(trades).set_index("exit_datetime")
        st.line_chart(ec["cum_pnl_points"])

        st.subheader("Trade duration distribution (minutes)")
        duration_minutes = (trades["exit_datetime"] - trades["entry_datetime"]).dt.total_seconds() / 60
        st.bar_chart(duration_minutes.value_counts(bins=15).sort_index())

        st.subheader("Trade log")
        st.dataframe(trades, use_container_width=True)

        csv = trades.to_csv(index=False).encode("utf-8")
        st.download_button("Download trade log (CSV)", csv, "trade_log.csv", "text/csv")
    else:
        st.warning(
            "No trades were generated with these settings. Try widening the time "
            "window, lowering the volume/OB spike multipliers, or turning off "
            "trend alignment."
        )
else:
    st.info("Set your parameters in the sidebar, then click **Run backtest**.")