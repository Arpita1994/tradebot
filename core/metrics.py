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


def add_net_pnl(trades: pd.DataFrame, cost_points: float = 4.0) -> pd.DataFrame:
    """Adds a 'net_pnl_points' column: pnl_points minus a flat per-trade cost
    (brokerage/taxes, in points), so the trade log shows both the raw price
    move (pnl_points) and what actually lands in the account (net_pnl_points).
    Returns a new DataFrame; does not mutate the input."""
    out = trades.copy()
    if out.empty:
        out["net_pnl_points"] = pd.Series(dtype=float)
    else:
        out["net_pnl_points"] = out["pnl_points"] - cost_points
    return out


def daily_summary(trades: pd.DataFrame, cost_points: float = 4.0) -> pd.DataFrame:
    """One row per calendar day (grouped by exit date, since that's when a
    trade's P&L is actually realized): trade count, win/loss count (by the
    raw price move -- pnl_points > 0 counts as a win, independent of costs),
    that day's win rate, that day's GROSS P&L (before costs), that day's NET
    P&L (after cost_points per trade), and the running cumulative NET P&L
    across days up to and including that day -- i.e. the "net realised P&L"
    equity curve, in points."""
    cols = ["date", "trades", "wins", "losses", "win_rate_pct",
            "gross_pnl_points", "net_pnl_points", "net_realised_pnl_points"]
    if trades.empty:
        return pd.DataFrame(columns=cols)

    t = trades.copy()
    t["net_pnl_points"] = t["pnl_points"] - cost_points
    t["date"] = pd.to_datetime(t["exit_datetime"]).dt.date

    grouped = t.groupby("date").agg(
        trades=("pnl_points", "count"),
        wins=("pnl_points", lambda s: int((s > 0).sum())),
        losses=("pnl_points", lambda s: int((s <= 0).sum())),
        gross_pnl_points=("pnl_points", "sum"),
        net_pnl_points=("net_pnl_points", "sum"),
    ).reset_index().sort_values("date")

    grouped["win_rate_pct"] = round(100 * grouped["wins"] / grouped["trades"], 2)
    grouped["gross_pnl_points"] = grouped["gross_pnl_points"].round(2)
    grouped["net_pnl_points"] = grouped["net_pnl_points"].round(2)
    grouped["net_realised_pnl_points"] = grouped["net_pnl_points"].cumsum().round(2)

    return grouped[cols]