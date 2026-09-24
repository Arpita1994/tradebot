"""
Runs the backtest across a grid of parameter combinations and reports which
ones hold up vs. which look like noise/overfitting.

A strategy with a real edge should show broadly similar results (win rate,
profit factor, drawdown) across nearby parameter values. If performance
swings wildly between adjacent settings (e.g. great at multiplier=1.2 but
terrible at 1.3), that's a red flag that you're fitting noise, not finding
a real pattern.

Usage:
    python parameter_sweep.py
"""

import datetime as dt
import itertools

import pandas as pd

from core.data_loader import load_from_csv, filter_date_range
from core.strategy import StrategyParams, generate_signals
from core.backtest import run_backtest
from core.metrics import summarize

# --- adjust these ---
CSV_PATH = "data/crudeoil_last_6_months.csv"
START_DATE = None   # None = use full file range
END_DATE = None
START_TIME = dt.time(9, 15)
END_TIME = dt.time(15, 0)

# The grid: every combination of these values gets backtested.
# Keep it small at first -- it grows multiplicatively (3x3x3x2x2 = 108 runs here).
GRID = {
    "lookback_time": [15, 20, 30],
    "volume_spike_multiplier": [1.1, 1.2, 1.5],
    "consolidation_candles": [5, 7, 10],
    "consolidation_tightness": [2.0, 2.5, 3.0],
    "sl_pct_of_ob": [30, 50, 75],
    "trend_lookback_candles": [20],       # add more values to sweep this too
    "reward_risk": [2.0],                 # add more values to sweep this too
    "require_trend_alignment": [True, False],
    "min_touches": [1, 2],                # 1 = first-retest entry, 2 = require one hold+retest first
}
# ---------------------

df = load_from_csv(CSV_PATH)
if START_DATE and END_DATE:
    df = filter_date_range(df, START_DATE, END_DATE)

print(f"Loaded {len(df)} candles from {CSV_PATH}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")

keys = list(GRID.keys())
combos = list(itertools.product(*GRID.values()))
print(f"Running {len(combos)} parameter combinations...\n")

results = []
for combo in combos:
    kwargs = dict(zip(keys, combo))
    params = StrategyParams(
        start_time=START_TIME,
        end_time=END_TIME,
        **kwargs,
    )
    signals = generate_signals(df, params)
    trades = run_backtest(df, signals)
    stats = summarize(trades)
    results.append({**kwargs, **stats})

results_df = pd.DataFrame(results)

# Only look at combos with a reasonable number of trades -- a "great" result
# from 3 trades is meaningless.
MIN_TRADES = 10
usable = results_df[results_df["total_trades"] >= MIN_TRADES].copy()

print(f"Combinations with >= {MIN_TRADES} trades: {len(usable)} / {len(results_df)}\n")

if usable.empty:
    print("No combination produced enough trades. Try loosening the grid "
          f"(lower multipliers) or lowering MIN_TRADES.")
else:
    usable = usable.sort_values("total_pnl_points", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("Top 10 by total P&L (points):")
    print(usable.head(10)[
        ["lookback_time", "volume_spike_multiplier", "consolidation_candles",
         "consolidation_tightness",
         "sl_pct_of_ob", "require_trend_alignment", "min_touches", "total_trades",
         "win_rate_pct", "total_pnl_points", "profit_factor", "max_drawdown_points"]
    ].to_string(index=False))

    print(f"\nWin rate across all usable combos: "
          f"min={usable['win_rate_pct'].min():.1f}%  "
          f"median={usable['win_rate_pct'].median():.1f}%  "
          f"max={usable['win_rate_pct'].max():.1f}%")
    print(f"Total P&L across all usable combos: "
          f"min={usable['total_pnl_points'].min():.1f}  "
          f"median={usable['total_pnl_points'].median():.1f}  "
          f"max={usable['total_pnl_points'].max():.1f}")
    print("\nIf these ranges are wide (e.g. P&L swinging from -200 to +400 "
          "between nearby settings), the strategy is sensitive to exact "
          "parameter choice -- a warning sign, not a feature.")

out_path = "data/parameter_sweep_results.csv"
results_df.to_csv(out_path, index=False)
print(f"\nFull results saved to {out_path}")