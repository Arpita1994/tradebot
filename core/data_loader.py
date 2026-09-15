"""
Load OHLCV candle data for backtesting.

Two paths are supported:
1. load_from_csv()  -> read a local CSV (exported from a broker, TradingView, etc.)
2. load_from_fyers() -> pull directly from the Fyers API (see fetchers/fyers_fetcher.py)

The rest of the engine only cares about a DataFrame with columns:
    datetime (tz-aware, Asia/Kolkata), open, high, low, close, volume
sorted ascending by datetime.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and dtypes regardless of the source's casing/format."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # common alternate names
    rename_map = {
        "date": "datetime",
        "time": "datetime",
        "timestamp": "datetime",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vol": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("Asia/Kolkata")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[REQUIRED_COLS]


def load_from_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    return _standardize(df)


def load_from_fyers(symbol: str, resolution: str, start_date: str, end_date: str,
                     access_token: str, app_id: str) -> pd.DataFrame:
    """
    Thin wrapper around fetchers.fyers_fetcher. Requires a Fyers API app + a
    daily-generated access_token (see generate_token.py and README.md).
    """
    from fetchers.fyers_fetcher import fetch_history

    raw = fetch_history(symbol=symbol, resolution=resolution,
                         start_date=start_date, end_date=end_date,
                         access_token=access_token, app_id=app_id)
    return _standardize(raw)


def filter_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    mask = (df["datetime"].dt.date >= pd.to_datetime(start_date).date()) & \
           (df["datetime"].dt.date <= pd.to_datetime(end_date).date())
    return df.loc[mask].reset_index(drop=True)


def filter_time_of_day(df: pd.DataFrame, start_time, end_time) -> pd.Series:
    """Returns a boolean mask (not a filtered df) so callers can still see prior
    bars for lookback calculations while only trading within the window."""
    t = df["datetime"].dt.time
    return (t >= start_time) & (t <= end_time)
