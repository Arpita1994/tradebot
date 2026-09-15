"""
Bar-by-bar trade simulator: walks forward from each signal's entry bar until the
stop loss or target is hit (or data runs out), one open trade per signal.
Assumes a single position at a time is NOT enforced across overlapping signals --
each signal is tested independently, which is standard for a first-pass backtest.
If a bar's range contains both SL and target, SL is assumed hit first (conservative).
"""

from __future__ import annotations

import pandas as pd


def run_backtest(df: pd.DataFrame, signals: list[dict], max_hold_bars: int = 500) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    trades = []

    for sig in signals:
        i = sig["signal_bar_index"]
        direction = sig["direction"]
        entry_price = sig["entry_price"]
        sl_price = sig["sl_price"]
        target_price = sig["target_price"]

        exit_price = None
        exit_datetime = None
        exit_reason = "eod_no_exit"

        end = min(i + max_hold_bars, len(df) - 1)
        for j in range(i, end + 1):
            bar = df.iloc[j]
            if direction == "long":
                hit_sl = bar["low"] <= sl_price
                hit_tp = bar["high"] >= target_price
            else:
                hit_sl = bar["high"] >= sl_price
                hit_tp = bar["low"] <= target_price

            if hit_sl and hit_tp:
                exit_price, exit_reason = sl_price, "sl"          # conservative: SL first
            elif hit_sl:
                exit_price, exit_reason = sl_price, "sl"
            elif hit_tp:
                exit_price, exit_reason = target_price, "target"

            if exit_price is not None:
                exit_datetime = bar["datetime"]
                break

        if exit_price is None:
            last_bar = df.iloc[end]
            exit_price = last_bar["close"]
            exit_datetime = last_bar["datetime"]

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)

        trades.append({
            **sig,
            "exit_datetime": exit_datetime,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_points": pnl_points,
            "r_multiple": pnl_points / sig["risk_points"] if sig["risk_points"] else None,
        })

    return pd.DataFrame(trades)
