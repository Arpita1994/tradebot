from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "total_trades": 0, "win_rate_pct": 0, "total_pnl_points": 0,
            "profit_factor": None, "avg_r_multiple": None, "max_drawdown_points": 0,
        }

    wins = trades[trades["pnl_points"] > 0]
    losses = trades[trades["pnl_points"] <= 0]

    gross_profit = wins["pnl_points"].sum()
    gross_loss = -losses["pnl_points"].sum()

    equity = trades["pnl_points"].cumsum()
    running_max = equity.cummax()
    drawdown = running_max - equity
    max_dd = drawdown.max() if not drawdown.empty else 0

    holding_time = trades["exit_datetime"] - trades["entry_datetime"]
    avg_minutes = holding_time.mean().total_seconds() / 60
    median_minutes = holding_time.median().total_seconds() / 60
    max_minutes = holding_time.max().total_seconds() / 60

    return {
        "total_trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2),
        "total_pnl_points": round(trades["pnl_points"].sum(), 2),
        "avg_pnl_points_per_trade": round(trades["pnl_points"].mean(), 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "avg_r_multiple": round(trades["r_multiple"].mean(), 2),
        "max_drawdown_points": round(max_dd, 2),
        "longs": int((trades["direction"] == "long").sum()),
        "shorts": int((trades["direction"] == "short").sum()),
        "avg_holding_time_minutes": round(avg_minutes, 1),
        "median_holding_time_minutes": round(median_minutes, 1),
        "max_holding_time_minutes": round(max_minutes, 1),
    }


def format_minutes(minutes: float) -> str:
    """Human-readable holding time, e.g. 135.0 -> '2h 15m'."""
    if minutes is None:
        return "-"
    total = int(round(minutes))
    hours, mins = divmod(total, 60)
    if hours == 0:
        return f"{mins}m"
    return f"{hours}h {mins}m"


def equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["exit_datetime", "cum_pnl_points"])
    t = trades.sort_values("exit_datetime").copy()
    t["cum_pnl_points"] = t["pnl_points"].cumsum()
    return t[["exit_datetime", "cum_pnl_points"]]