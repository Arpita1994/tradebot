"""
Backtests the order-block strategy on MCX Crude Oil AT-THE-MONEY OPTION
premiums -- not the futures price.

IMPORTANT DESIGN NOTE: an ATM option's premium is NOT a continuous price
series the way futures are. When the strike rolls (because the underlying
moved, or the month changed), the premium jumps to a completely different
price scale -- not because the market moved, but purely because it's now a
different contract. Stitching that into one continuous series and running
order-block detection across the seam would manufacture fake "big move"
signals at every roll point.

So instead, this script:
  1. Splits the backtest period into monthly LEGS (one per calendar month,
     matching how the underlying futures contract already rolls monthly).
  2. For each leg, picks the ATM strike using the underlying's price at the
     start of that month (rounded to the nearest 50, the live strike step
     confirmed from Fyers' MCX symbol master), and fetches that ONE specific
     option contract's real intraday history for that month from Fyers.
  3. Runs the SAME unmodified order-block strategy engine independently on
     each leg's own real price data -- no cross-leg contamination.
  4. Combines the resulting TRADES (not the raw price series) from every leg
     into one overall report, plus a per-leg breakdown so you can see which
     months had usable data and how each performed on its own.

Expired option contracts fall out of Fyers' historical API (same issue we
hit with expired futures) -- legs where that happens are skipped and
reported, not silently ignored.

Usage:
    python generate_token.py       # if today's token isn't fresh yet
    python backtest_atm_options.py
"""

import datetime as dt
import os

import pandas as pd
from dotenv import load_dotenv

from fetchers.fyers_fetcher import fetch_history
from core.strategy import StrategyParams, generate_signals
from core.backtest import run_backtest
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
UNDERLYING_FUTURES_CSV = "data/crudeoil_last_6_months.csv"  # from download_history.py
OPTION_TYPE = "CE"          # "CE" (call) or "PE" (put) -- one series per run
STRIKE_STEP = 50            # confirmed live strike spacing from Fyers symbol master
RESOLUTION = "5"            # "5" or "15"
TIME_START = dt.time(9, 15)
TIME_END = dt.time(15, 0)

STRATEGY_KWARGS = dict(
    lookback_time=20,
    volume_spike_multiplier=1.2,
    consolidation_candles=7,
    consolidation_tightness=2.5,
    sl_pct_of_ob=50.0,
    trend_lookback_candles=20,
    reward_risk=2.0,
    require_trend_alignment=True,
    min_touches=1,
)

OUTPUT_TRADES_CSV = "data/atm_options_trade_log.csv"
OUTPUT_LEG_SUMMARY_CSV = "data/atm_options_leg_summary.csv"
# ---------------------


def month_code(d: dt.date) -> str:
    """e.g. 2026-09-15 -> '26SEP' (matches Fyers' MCX contract naming)."""
    return f"{d.strftime('%y')}{d.strftime('%b').upper()}"


def nearest_strike(price: float, step: int = STRIKE_STEP) -> int:
    return int(round(price / step) * step)


def month_starts(start: dt.date, end: dt.date):
    """Yield the first day of each calendar month touching [start, end]."""
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


underlying = pd.read_csv(UNDERLYING_FUTURES_CSV, parse_dates=["datetime"])
underlying["date"] = pd.to_datetime(underlying["datetime"]).dt.date
data_start, data_end = underlying["date"].min(), underlying["date"].max()

print(f"Underlying futures data: {data_start} to {data_end} ({len(underlying)} rows)\n")

legs = []
for m_start in month_starts(data_start, data_end):
    m_end_exclusive = (m_start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    leg_start = max(m_start, data_start)
    leg_end = min(m_end_exclusive - dt.timedelta(days=1), data_end)
    if leg_start > leg_end:
        continue

    ref_row = underlying[underlying["date"] >= leg_start].iloc[0]
    ref_price = ref_row["close"]
    strike = nearest_strike(ref_price)
    symbol = f"MCX:CRUDEOIL{month_code(m_start)}{strike}{OPTION_TYPE}"

    legs.append({
        "month": m_start.strftime("%Y-%m"),
        "leg_start": leg_start,
        "leg_end": leg_end,
        "underlying_ref_price": ref_price,
        "strike": strike,
        "symbol": symbol,
    })

print(f"Planned {len(legs)} monthly legs:")
for leg in legs:
    print(f"  {leg['month']}: {leg['symbol']}  (underlying ~{leg['underlying_ref_price']:.0f} "
          f"-> strike {leg['strike']})")
print()

all_trades = []
leg_results = []

for leg in legs:
    print(f"--- {leg['month']}  {leg['symbol']}  [{leg['leg_start']} to {leg['leg_end']}] ---")
    try:
        df = fetch_history(
            symbol=leg["symbol"],
            resolution=RESOLUTION,
            start_date=leg["leg_start"].isoformat(),
            end_date=leg["leg_end"].isoformat(),
            access_token=ACCESS_TOKEN,
            app_id=APP_ID,
        )
    except Exception as e:
        print(f"  SKIPPED -- could not fetch data: {e}\n")
        leg_results.append({**leg, "status": "no_data", "rows": 0, "trades": 0,
                             "pnl_points": None, "win_rate_pct": None})
        continue

    if df.empty:
        print("  SKIPPED -- no candles returned\n")
        leg_results.append({**leg, "status": "empty", "rows": 0, "trades": 0,
                             "pnl_points": None, "win_rate_pct": None})
        continue

    params = StrategyParams(start_time=TIME_START, end_time=TIME_END, **STRATEGY_KWARGS)
    signals = generate_signals(df, params)
    trades = run_backtest(df, signals)

    if not trades.empty:
        trades["leg_month"] = leg["month"]
        trades["leg_symbol"] = leg["symbol"]
        all_trades.append(trades)

    stats = summarize(trades)
    print(f"  rows={len(df)}  trades={stats['total_trades']}  "
          f"win_rate={stats['win_rate_pct']}%  pnl={stats['total_pnl_points']} pts\n")

    leg_results.append({
        **leg, "status": "ok", "rows": len(df),
        "trades": stats["total_trades"],
        "pnl_points": stats["total_pnl_points"],
        "win_rate_pct": stats["win_rate_pct"],
    })

leg_df = pd.DataFrame(leg_results)
os.makedirs("data", exist_ok=True)
leg_df.to_csv(OUTPUT_LEG_SUMMARY_CSV, index=False)

print("=" * 70)
print("PER-LEG SUMMARY")
print("=" * 70)
pd.set_option("display.width", 160)
print(leg_df[["month", "symbol", "status", "rows", "trades", "win_rate_pct", "pnl_points"]]
      .to_string(index=False))
print(f"\nSaved to {OUTPUT_LEG_SUMMARY_CSV}")

if all_trades:
    combined = pd.concat(all_trades, ignore_index=True)
    combined.to_csv(OUTPUT_TRADES_CSV, index=False)
    overall = summarize(combined)

    print("\n" + "=" * 70)
    print("OVERALL (all legs combined)")
    print("=" * 70)
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\n  avg trade duration: {format_minutes(overall.get('avg_holding_time_minutes'))}")
    print(f"\nFull combined trade log saved to {OUTPUT_TRADES_CSV}")
elif (leg_df["status"] == "ok").sum() == 0:
    print("\nNo leg returned any data -- every option contract was either "
          "invalid, expired, or unavailable from Fyers (see the per-leg "
          "table above for the specific error per month). This is a data "
          "availability issue, not a strategy/parameter issue: try a more "
          "recent date range where the option contracts are more likely to "
          "still be live and queryable.")
else:
    print("\nData was fetched successfully for at least one leg, but no "
          "trades were generated. This can happen if the option premium's "
          "own volume/price action rarely triggers the current thresholds "
          "-- options often trade thinner than the underlying future, so "
          "consider loosening the volume/OB spike multipliers for this run.")