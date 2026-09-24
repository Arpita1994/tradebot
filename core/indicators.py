"""
Volume and trend indicators used to gate entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_avg_volume(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Average volume of the `lookback` bars BEFORE the current one (no look-ahead)."""
    return df["volume"].rolling(lookback, min_periods=max(2, lookback // 2)).mean().shift(1)


def rolling_avg_candle_range(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Average single-candle (high - low) range of the `lookback` bars BEFORE
    the current one (no look-ahead). Used as the baseline "normal" candle
    size a consolidation box's total height is compared against -- a box is
    only a genuine compression if it's meaningfully tighter than this."""
    candle_range = df["high"] - df["low"]
    return candle_range.rolling(lookback, min_periods=max(2, lookback // 2)).mean().shift(1)


def rolling_avg_body(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Average absolute candle body (|close - open|) of the `lookback` bars
    BEFORE the current one (no look-ahead). Used to judge whether a given
    candle's own body is unusually large/decisive relative to recent candles
    (as opposed to rolling_avg_candle_range, which uses high-low)."""
    body = (df["close"] - df["open"]).abs()
    return body.rolling(lookback, min_periods=max(2, lookback // 2)).mean().shift(1)


def volume_going_up(df: pd.DataFrame, lookback: int, spike_multiplier: float = 1.2) -> pd.Series:
    """
    True where current bar's volume is rising vs its own recent average, i.e.
    volume[i] > spike_multiplier * avg(volume[i-lookback : i]).
    This is the "volume going up" input.
    """
    avg_vol = rolling_avg_volume(df, lookback)
    return df["volume"] > (avg_vol * spike_multiplier)


def trend_direction(df: pd.DataFrame, lookback_candles: int) -> pd.Series:
    """
    Trend analysis using N lookback candles:
    - 'up'   : close > SMA(N) AND SMA(N) is higher than it was N/2 bars ago (rising)
    - 'down' : close < SMA(N) AND SMA(N) is lower than it was N/2 bars ago (falling)
    - 'flat' : otherwise
    """
    sma = df["close"].rolling(lookback_candles, min_periods=max(2, lookback_candles // 2)).mean()
    slope_ref = sma.shift(max(1, lookback_candles // 2))

    up = (df["close"] > sma) & (sma > slope_ref)
    down = (df["close"] < sma) & (sma < slope_ref)

    trend = pd.Series(np.where(up, "up", np.where(down, "down", "flat")), index=df.index)
    return trend