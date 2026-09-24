"""
Backtests the order-block strategy on MCX Crude Oil AT-THE-MONEY OPTION
premiums -- CURRENT ACTIVE OPTIONS MONTH ONLY, tracking CALLS (CE) and PUTS
(PE) in parallel, re-picking the ATM strike every few hours from the
underlying's live price. No CSV involved -- the underlying's price is
fetched directly from Fyers each run.

See core/atm_options.py for the full design notes (why per-window strikes,
why the "current month" is based on the actual options expiry rather than
the calendar month, why no CSV).

Usage:
    python generate_token.py       # if today's token isn't fresh yet
    python backtest_atm_options_4h.py
"""

import datetime as dt
import os

import pandas as pd
from dotenv import load_dotenv

from core.atm_options import (
    compute_current_option_month, month_code, fetch_underlying, build_legs, run_legs,
)
from core.metrics import summarize, format_minutes

load_dotenv()

APP_ID = os.getenv("FYERS_APP_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

if not APP_ID or not ACCESS_TOKEN:
    raise SystemExit(
        "Missing FYERS_APP_ID or FYERS_ACCESS_TOKEN in .env. "
        "Run `python generate_token.py` first."
    )

# --- adjust these ---
OPTION_TYPES = ["CE", "PE"]     # tracked in parallel
RESOLUTION = "5"                # "5" or "15"
WINDOW_HOURS = 4                # re-pick ATM strike this often
UNDERLYING_LOOKBACK_DAYS = 15   # how far back to pull live underlying price
                                 # from, to build windows/strikes for the
                                 # current contract (approximate -- see notes
                                 # in core/atm_options.py)
TIME_START = dt.time(9, 15)     # intraday session filter used INSIDE each window
TIME_END = dt.time(15, 0)

STRATEGY_KWARGS = dict(
    lookback_time=20,
    volume_spike_multiplier=1.2,
    consolidation_candles=7,     # at least this many consecutive candles must consolidate
    consolidation_tightness=2.5, # box height must stay within this many x avg candle range
    sl_pct_of_ob=50.0,
    trend_lookback_candles=20,
    reward_risk=2.0,
    require_trend_alignment=True,
    min_touches=1,
)

OUTPUT_TRADES_CSV = "data/atm_options_4h_trade_log.csv"
OUTPUT_WINDOW_SUMMARY_CSV = "data/atm_options_4h_window_summary.csv"
# ---------------------

print("Checking with Fyers which contract month is currently live...")
year, month, mcode = compute_current_option_month(APP_ID, ACCESS_TOKEN)
print(f"Current active options month: {mcode}\n")

print(f"Fetching underlying live from Fyers (last {UNDERLYING_LOOKBACK_DAYS} days, no CSV)...")
underlying = fetch_underlying(APP_ID, ACCESS_TOKEN, mcode, UNDERLYING_LOOKBACK_DAYS, RESOLUTION)
if underlying.empty:
    raise SystemExit("Fyers returned no underlying data for the lookback window -- "
                      "check your token/symbol and try again.")
print(f"Underlying: {underlying['datetime'].min()} to {underlying['datetime'].max()} "
      f"({len(underlying)} rows)\n")

legs = build_legs(underlying, mcode, OPTION_TYPES, WINDOW_HOURS)
n_windows = len(legs) // max(len(OPTION_TYPES), 1)
print(f"Built {n_windows} {WINDOW_HOURS}-hour window(s) -> {len(legs)} legs "
      f"({n_windows} windows x {len(OPTION_TYPES)} option types):")
for leg in legs:
    print(f"  [{leg['window_start']:%Y-%m-%d %H:%M} - {leg['window_end']:%H:%M}] "
          f"{leg['option_type']}  underlying~{leg['underlying_ref_price']:.0f} -> {leg['symbol']}")
print()


def progress(n, total, leg, status):
    tag = {"ok": "done", "empty": "no candles", "no_data": "FAILED"}[status]
    print(f"  [{n}/{total}] {leg['option_type']} {leg['symbol']} "
          f"[{leg['window_start']:%m-%d %H:%M}] -- {tag}")


window_df, combined = run_legs(
    legs, APP_ID, ACCESS_TOKEN, RESOLUTION, TIME_START, TIME_END, STRATEGY_KWARGS,
    progress_callback=progress,
)

os.makedirs("data", exist_ok=True)
window_df.to_csv(OUTPUT_WINDOW_SUMMARY_CSV, index=False)

print("\n" + "=" * 80)
print("PER-WINDOW SUMMARY")
print("=" * 80)
pd.set_option("display.width", 180)
print(window_df[["window_start", "window_end", "option_type", "symbol", "status",
                  "rows", "trades", "win_rate_pct", "pnl_points"]].to_string(index=False))
print(f"\nSaved to {OUTPUT_WINDOW_SUMMARY_CSV}")

if not combined.empty:
    combined.to_csv(OUTPUT_TRADES_CSV, index=False)

    print("\n" + "=" * 80)
    print("BY OPTION TYPE")
    print("=" * 80)
    for opt in OPTION_TYPES:
        opt_trades = combined[combined["option_type"] == opt]
        opt_stats = summarize(opt_trades)
        print(f"\n{opt}:")
        for k, v in opt_stats.items():
            print(f"  {k}: {v}")
        if opt_stats["total_trades"]:
            print(f"  avg trade duration: {format_minutes(opt_stats.get('avg_holding_time_minutes'))}")

    overall = summarize(combined)
    print("\n" + "=" * 80)
    print("OVERALL (CE + PE, all windows combined)")
    print("=" * 80)
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\n  avg trade duration: {format_minutes(overall.get('avg_holding_time_minutes'))}")
    print(f"\nFull combined trade log saved to {OUTPUT_TRADES_CSV}")
elif (window_df["status"] == "ok").sum() == 0:
    print("\nNo window returned any data -- every option contract was either "
          "invalid or unavailable from Fyers (see the per-window table above). "
          "This is a data availability issue, not a strategy/parameter issue.")
else:
    print("\nData was fetched successfully for at least one window, but no "
          "trades were generated. With only a few hours of 5-min candles per "
          "window (minus the lookback/displacement bars the strategy needs "
          "just to warm up), that's expected fairly often -- consider "
          "loosening the volume/OB spike multipliers, or widening WINDOW_HOURS, "
          "if you want more signals.")