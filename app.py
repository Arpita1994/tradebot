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
# Locally, with no .streamlit/secrets.toml file at all, newer Streamlit
# versions raise just from checking `st.secrets` -- that's expected when
# running off .env, so swallow it and move on.
try:
    for _key in ("FYERS_APP_ID", "FYERS_SECRET_KEY", "FYERS_REDIRECT_URI", "FYERS_ACCESS_TOKEN"):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from core.strategy import StrategyParams, generate_signals
from core.backtest import run_backtest
from core.metrics import (
    summarize, equity_curve, format_minutes, add_net_pnl, daily_summary, exit_reason_breakdown,
)
from core.atm_options import (
    compute_current_option_month, month_code, fetch_underlying, build_legs,
    run_legs, run_legs_candle_breakout,
)

st.set_page_config(page_title="MCX Order-Block Backtester", layout="wide")
st.title("MCX Commodity Order-Block Backtester")
st.caption(
    "Volume-based order block strategy: a fresh order block, revisited with rising "
    "volume, inside your trading window, aligned with the prevailing trend."
)

# ---------------------------------------------------------------- Data source
st.sidebar.header("1. Data")
source = st.sidebar.radio(
    "Data source",
    ["Upload CSV", "Use sample data", "Fyers API", "ATM Options (current month, live)"],
)

cost_points_per_trade = st.sidebar.number_input(
    "Costs per trade (points): brokerage/taxes", 0.0, 100.0, 4.0, 0.5,
    help="Deducted from every trade's pnl_points to get its net P&L -- shown as a "
         "'Net PnL (points)' column in every trade log, and used to build the daily "
         "summary table below each trade log. Applies everywhere in this app."
)

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

elif source == "Fyers API":
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

# ---------------------------------------------------------------- ATM Options mode
# Structurally different from the other three sources (many CE/PE legs, each
# on its own strike, instead of one continuous price series) so it runs its
# own pipeline end to end and skips the generic one below.
if source == "ATM Options (current month, live)":
    st.caption(
        "Crude Oil ATM options, current active month only (based on the actual "
        "options expiry, not the calendar month). CE and PE are tracked in "
        "parallel, and the ATM strike is re-picked from the underlying's live "
        "price every few hours -- each window is backtested on its own strike, "
        "since a strike change is not a real price move. No CSV is used; the "
        "underlying's price is fetched live from Fyers."
    )

    env_app_id = os.getenv("FYERS_APP_ID", "")
    env_token = os.getenv("FYERS_ACCESS_TOKEN", "")
    if env_token:
        st.sidebar.success("Access token loaded from .env")
    else:
        st.sidebar.warning(
            "No token found in .env. Run `python generate_token.py` in your "
            "terminal first (see README.md), then reload this page."
        )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _current_option_month(app_id, token, cache_bust_date):
        # cache_bust_date forces a fresh check once per calendar day even
        # within the 1-hour TTL, since the active month can only change once
        # a contract actually expires (once/day at most, near month-end).
        return compute_current_option_month(app_id, token)

    if env_token:
        with st.spinner("Checking with Fyers which contract month is currently live..."):
            year, month, mcode = _current_option_month(env_app_id, env_token, str(dt.date.today()))
        st.sidebar.info(f"Current active options month: **{mcode}**\n\n"
                         f"(confirmed live against Fyers directly -- not calendar-guessed)")
    else:
        mcode = None
        st.sidebar.caption("Current options month will be checked once a token is available.")

    atm_strategy = st.sidebar.radio(
        "Strategy",
        ["Consolidation box (order block)", "30-min candle breakout"],
        key="atm_strategy",
        help="Consolidation box: the box/breakout/retest order-block strategy on "
             "5- or 15-min candles. 30-min candle breakout: enters above any green "
             "30-min candle on a breakout-continuation limit order (see section below)."
    )
    is_breakout = atm_strategy == "30-min candle breakout"

    option_types = st.sidebar.multiselect("Option types", ["CE", "PE"], default=["CE", "PE"])
    window_hours = st.sidebar.number_input("Re-pick ATM strike every N hours", 1, 24, 4)
    lookback_days = st.sidebar.number_input(
        "Underlying lookback (days)", 1, 60, 15,
        help="How far back to pull the live underlying price from, to build "
             "windows/strikes for the current contract."
    )
    if is_breakout:
        resolution_atm = "30"
        st.sidebar.caption("Resolution fixed at **30 min** for the candle breakout strategy.")
    else:
        resolution_atm = st.sidebar.selectbox("Resolution", ["5", "15"], index=0, key="atm_res")

    st.sidebar.header("2. Time of day")
    atm_start_time = st.sidebar.time_input("Session start", dt.time(9, 15), key="atm_start")
    atm_end_time = st.sidebar.time_input("Session end", dt.time(15, 0), key="atm_end")

    if is_breakout:
        st.sidebar.header("3. Entry, stop loss & exit")
        atm_buy_limit_pct = st.sidebar.slider(
            "Buy-limit: % of candle body above close", 0, 100, 20, 5, key="atm_blp",
            help="On a green 30-min candle's close, a buy order is placed this many "
                 "% of the candle's body ABOVE the close (a breakout-continuation "
                 "entry). It only stays valid for the single next 30-min candle -- "
                 "if that candle's high never reaches it, the order is cancelled."
        ) / 100.0

        atm_exit_mode_label = st.sidebar.radio(
            "Exit style",
            ["Trailing stop on green candles (no target)", "Fixed target (reward:risk)"],
            index=0, key="atm_exit_mode",
            help="Trailing: no target price at all -- the stop loss ratchets up off "
                 "every new green candle while the trade is open, and you exit purely "
                 "when price drops through it. Fixed target: the original 2:1-style "
                 "behavior -- one stop and one target set at entry, neither moves."
        )
        atm_exit_mode = "trailing" if atm_exit_mode_label.startswith("Trailing") else "target"

        atm_sl_buffer_pct = st.sidebar.slider(
            "Stop loss: % of candle body below the low", 0, 100, 10, 5, key="atm_slb",
            help="Stop loss sits below a green candle's low, by this many % of that "
                 "candle's body. Used to set the initial stop at entry, and -- in "
                 "trailing mode -- reused every 30 minutes to recalculate the stop "
                 "off whichever green candle just closed (only ever moving the stop "
                 "up, never back down)."
        ) / 100.0
        atm_max_loss_points = st.sidebar.number_input(
            "Max loss (points below entry candle's close, 0 = no cap)", 0.0, 1000.0, 0.0, 1.0, key="atm_max_loss",
            help="Caps how far BELOW THE SIGNAL CANDLE'S CLOSE the INITIAL stop loss "
                 "is ever allowed to sit -- e.g. 25 means the entry stop can never be "
                 "more than 25 points below that candle's close, even if the % of body "
                 "calculation would otherwise put it further away (guards against one "
                 "wicky entry candle creating an oversized risk). Applies ONLY to the "
                 "initial entry stop -- later trailing updates are left uncapped, since "
                 "the trailing ratchet only ever moves the stop up anyway. 0 disables it."
        )

        if atm_exit_mode == "target":
            atm_rr = st.sidebar.slider("Reward:Risk multiple", 0.5, 5.0, 2.0, 0.5, key="atm_rr_bo")
            atm_trail_min_body = 0.0  # unused in target mode (no trailing at all)
        else:
            atm_rr = 2.0  # unused in trailing mode
            atm_trail_min_body = st.sidebar.number_input(
                "Trailing update: minimum body (points, 0 = any green candle)",
                0.0, 1000.0, 0.0, 1.0, key="atm_trail_min_body",
                help="The trailing stop only follows the single PREVIOUS candle -- no "
                     "averaging or lookback. This sets a minimum size (in price points) "
                     "that candle's own body must have before it's allowed to move the "
                     "stop, so a tiny/insignificant green candle can't become the new "
                     "trail anchor. 0 = every green candle qualifies (default)."
            )

        st.sidebar.header("4. Decisive candle body")
        atm_body_mode_label = st.sidebar.selectbox(
            "How to combine the two checks below",
            ["Relative only (% of avg body)", "Absolute only (min points)",
             "Both required (AND)", "Either one (OR)"],
            index=2, key="atm_body_mode",
            help="Relative only: judge the candle purely against its own recent "
                 "average. Absolute only: judge it purely against a fixed points "
                 "floor you set. AND: both must pass. OR: either passing is enough."
        )
        atm_body_mode = {
            "Relative only (% of avg body)": "relative",
            "Absolute only (min points)": "absolute",
            "Both required (AND)": "and",
            "Either one (OR)": "or",
        }[atm_body_mode_label]

        if atm_body_mode != "absolute":
            atm_body_lookback = st.sidebar.number_input(
                "Body lookback (candles)", 2, 50, 10, key="atm_body_lb",
                help="How many candles before the current one to average body size "
                     "over, to judge what a 'normal' candle looks like right now."
            )
            atm_body_pct = st.sidebar.slider(
                "Decisive body: % of average body", 10, 300, 100, 10, key="atm_body_pct",
                help="A green candle's body must be at least this % of the recent "
                     "average body to trigger a signal (100% = at least average-sized). "
                     "Skipped while there isn't enough history yet to compute the average."
            )
        else:
            atm_body_lookback, atm_body_pct = 10, 100.0  # unused in this mode

        if atm_body_mode != "relative":
            atm_min_body_pts = st.sidebar.number_input(
                "Decisive body: minimum points (manual floor)", 0.0, 1000.0, 0.0, 0.5, key="atm_min_body",
                help="A hard floor on body size in price points, applied regardless of "
                     "the average -- set this manually if you want a fixed minimum "
                     "candle size no matter what the recent average is doing. 0 = no floor."
            )
        else:
            atm_min_body_pts = 0.0  # unused in this mode
    else:
        st.sidebar.header("3. Lookback & trend")
        atm_lookback = st.sidebar.number_input("Lookback time (bars)", 5, 200, 20, key="atm_lb")
        atm_trend_lookback = st.sidebar.number_input("Trend analysis lookback (candles)", 5, 200, 20, key="atm_tl")
        atm_require_trend = st.sidebar.checkbox("Only trade in direction of trend", value=True, key="atm_rt")

        st.sidebar.header("4. Volume")
        atm_vol_mult = st.sidebar.slider("Volume going up: spike multiplier (x avg)", 1.0, 3.0, 1.2, 0.1, key="atm_vm")

        st.sidebar.header("5. Order block (consolidation box)")
        atm_consol_candles = st.sidebar.number_input(
            "Minimum consolidating candles", 3, 30, 7, key="atm_cc",
            help="How many consecutive candles must trade in a tight range before "
                 "it counts as a consolidation box (your 'at least 5-10 candles' input)."
        )
        atm_tightness = st.sidebar.slider(
            "Box tightness (x avg candle range)", 1.0, 6.0, 2.5, 0.1, key="atm_tight",
            help="The box's total height must stay within this many times the average "
                 "single-candle range to still count as a genuine compression. Lower = stricter/tighter."
        )
        atm_min_touches = st.sidebar.number_input(
            "Minimum retest touches before entry", 1, 5, 1, key="atm_mt",
            help="1 = buy/sell on the first retest of the broken box edge after breakout; "
                 ">1 = require the retest to hold that many times first."
        )

        st.sidebar.header("6. Stop loss")
        atm_sl_pct = st.sidebar.slider("SL: % of order block range beyond the OB", 0, 200, 50, 5, key="atm_sl")

        st.sidebar.header("7. Target")
        atm_rr = st.sidebar.slider("Reward:Risk multiple", 0.5, 5.0, 2.0, 0.5, key="atm_rr")

    if st.button("Run ATM options backtest", type="primary", disabled=not env_token):
        with st.spinner("Fetching live underlying price and building windows..."):
            underlying = fetch_underlying(env_app_id, env_token, mcode, int(lookback_days), resolution_atm)

        if underlying.empty:
            st.error("Fyers returned no underlying data for that lookback window.")
            st.stop()

        st.success(f"Underlying: {underlying['datetime'].min()} to "
                   f"{underlying['datetime'].max()} ({len(underlying)} rows)")

        legs = build_legs(underlying, mcode, option_types, int(window_hours))
        n_windows = len(legs) // max(len(option_types), 1)
        st.write(f"Built **{n_windows}** {window_hours}-hour window(s) -> **{len(legs)}** legs "
                 f"({n_windows} windows x {len(option_types)} option type(s))")

        progress_bar = st.progress(0.0, text="Fetching option data and running the strategy per leg...")

        def update_progress(n, total, leg, status):
            progress_bar.progress(n / total, text=f"[{n}/{total}] {leg['option_type']} {leg['symbol']}")

        if is_breakout:
            breakout_kwargs = dict(
                buy_limit_pct=float(atm_buy_limit_pct),
                sl_buffer_pct=float(atm_sl_buffer_pct),
                reward_risk=float(atm_rr),
                body_lookback_candles=int(atm_body_lookback),
                body_pct_of_avg=float(atm_body_pct),
                min_body_points=float(atm_min_body_pts),
                body_filter_mode=atm_body_mode,
                exit_mode=atm_exit_mode,
                max_loss_points=float(atm_max_loss_points),
                trail_min_body_points=float(atm_trail_min_body),
            )
            with st.spinner("Running per-window backtests..."):
                window_df, combined = run_legs_candle_breakout(
                    legs, env_app_id, env_token, resolution_atm, atm_start_time, atm_end_time,
                    breakout_kwargs, progress_callback=update_progress,
                )
        else:
            strategy_kwargs = dict(
                lookback_time=int(atm_lookback),
                volume_spike_multiplier=float(atm_vol_mult),
                consolidation_candles=int(atm_consol_candles),
                consolidation_tightness=float(atm_tightness),
                sl_pct_of_ob=float(atm_sl_pct),
                trend_lookback_candles=int(atm_trend_lookback),
                reward_risk=float(atm_rr),
                require_trend_alignment=atm_require_trend,
                min_touches=int(atm_min_touches),
            )
            with st.spinner("Running per-window backtests..."):
                window_df, combined = run_legs(
                    legs, env_app_id, env_token, resolution_atm, atm_start_time, atm_end_time,
                    strategy_kwargs, progress_callback=update_progress,
                )
        progress_bar.empty()

        st.subheader("Per-window summary")
        st.dataframe(
            window_df[["window_start", "window_end", "option_type", "symbol", "status",
                       "rows", "trades", "win_rate_pct", "pnl_points"]],
            use_container_width=True,
        )

        if not combined.empty:
            st.subheader("By option type")
            for opt in option_types:
                opt_trades = combined[combined["option_type"] == opt]
                opt_stats = summarize(opt_trades)
                st.markdown(f"**{opt}**")
                cols = st.columns(5)
                cols[0].metric("Trades", opt_stats["total_trades"])
                cols[1].metric("Win rate", f"{opt_stats['win_rate_pct']}%")
                cols[2].metric("Total P&L (pts)", opt_stats["total_pnl_points"])
                cols[3].metric("Profit factor", opt_stats["profit_factor"])
                cols[4].metric("Avg duration", format_minutes(opt_stats.get("avg_holding_time_minutes")))

            st.subheader("Overall (CE + PE combined)")
            overall = summarize(combined)
            cols = st.columns(4)
            cols[0].metric("Total trades", overall["total_trades"])
            cols[1].metric("Win rate", f"{overall['win_rate_pct']}%")
            cols[2].metric("Total P&L (points)", overall["total_pnl_points"])
            cols[3].metric("Profit factor", overall["profit_factor"])
            cols2 = st.columns(4)
            cols2[0].metric("Avg P&L / trade", overall.get("avg_pnl_points_per_trade"))
            cols2[1].metric("Avg R multiple", overall["avg_r_multiple"])
            cols2[2].metric("Max drawdown (points)", overall["max_drawdown_points"])
            cols2[3].metric("Longs / Shorts", f"{overall.get('longs', 0)} / {overall.get('shorts', 0)}")

            combined_net = add_net_pnl(combined, cost_points_per_trade)

            st.subheader("Exit reasons")
            st.caption(
                "'eod_no_exit' trades never hit their SL or target (or the max-hold cutoff) -- "
                "they were force-closed because the data ran out before the trade could resolve. "
                "Their P&L is somewhat arbitrary; treat them with a bit of skepticism, especially if "
                "they're a large share. 'day_square_off' trades hit the day's defined time_end while "
                "still open -- that's a deliberate flat-by-end-of-day rule, not a backtest artifact, "
                "so its P&L is real (just cut short of wherever the trade might have gone overnight)."
            )
            st.dataframe(exit_reason_breakdown(combined), use_container_width=True)

            st.subheader("Trade log")
            st.dataframe(combined_net, use_container_width=True)
            csv = combined_net.to_csv(index=False).encode("utf-8")
            st.download_button("Download trade log (CSV)", csv, "atm_options_trade_log.csv", "text/csv")

            st.subheader("Daily summary")
            st.caption(
                f"Grouped by exit date. Net PnL/Net realised PnL both after deducting "
                f"{cost_points_per_trade:g} points/trade for brokerage & taxes -- "
                f"'Net realised PnL' is the running cumulative total across days."
            )
            daily = daily_summary(combined, cost_points_per_trade)
            st.dataframe(daily, use_container_width=True)
            daily_csv = daily.to_csv(index=False).encode("utf-8")
            st.download_button("Download daily summary (CSV)", daily_csv,
                                "atm_options_daily_summary.csv", "text/csv")
        elif (window_df["status"] == "ok").sum() == 0:
            st.warning("No window returned any data -- every option contract was either "
                       "invalid or unavailable from Fyers. This is a data availability "
                       "issue, not a strategy/parameter issue.")
        else:
            st.warning("Data was fetched for at least one window, but no trades were "
                       "generated. A window is only a few hours of candles, so this can "
                       "happen often -- try loosening the volume/OB spike multipliers, "
                       "or widening the re-pick interval.")
    else:
        st.info("Set your parameters in the sidebar, then click **Run ATM options backtest**.")
    st.stop()

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

st.sidebar.header("6. Order block (consolidation box)")
consolidation_candles = st.sidebar.number_input(
    "Minimum consolidating candles", 3, 30, 7,
    help="How many consecutive candles must trade in a tight range before it "
         "counts as a consolidation box (a.k.a. order block) -- your 'at least "
         "5-10 candles' input."
)
consolidation_tightness = st.sidebar.slider(
    "Box tightness (x avg candle range)", 1.0, 6.0, 2.5, 0.1,
    help="The box's total height must stay within this many times the average "
         "single-candle range to still count as a genuine compression. Lower = stricter/tighter."
)
min_touches = st.sidebar.number_input(
    "Minimum retest touches before entry", 1, 5, 1,
    help="1 = buy/sell on the first retest of the broken box edge after "
         "breakout. >1 = require the retest to hold that many times first "
         "(support/resistance-style confirmation) before entering -- fewer, "
         "later, more-confirmed signals."
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
        consolidation_candles=int(consolidation_candles),
        consolidation_tightness=float(consolidation_tightness),
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

        trades_net = add_net_pnl(trades, cost_points_per_trade)

        st.subheader("Exit reasons")
        st.caption(
            "'eod_no_exit' trades never hit their SL or target (or the max-hold cutoff) -- "
            "they were force-closed because the data ran out. Their P&L is somewhat arbitrary; "
            "treat them with a bit of skepticism, especially if they're a large share."
        )
        st.dataframe(exit_reason_breakdown(trades), use_container_width=True)

        st.subheader("Trade log")
        st.dataframe(trades_net, use_container_width=True)

        csv = trades_net.to_csv(index=False).encode("utf-8")
        st.download_button("Download trade log (CSV)", csv, "trade_log.csv", "text/csv")

        st.subheader("Daily summary")
        st.caption(
            f"Grouped by exit date. Net PnL/Net realised PnL both after deducting "
            f"{cost_points_per_trade:g} points/trade for brokerage & taxes -- "
            f"'Net realised PnL' is the running cumulative total across days."
        )
        daily = daily_summary(trades, cost_points_per_trade)
        st.dataframe(daily, use_container_width=True)
        daily_csv = daily.to_csv(index=False).encode("utf-8")
        st.download_button("Download daily summary (CSV)", daily_csv,
                            "daily_summary.csv", "text/csv")
    else:
        st.warning(
            "No trades were generated with these settings. Try widening the time "
            "window, lowering the volume/OB spike multipliers, or turning off "
            "trend alignment."
        )
else:
    st.info("Set your parameters in the sidebar, then click **Run backtest**.")