"""
Historical MCX candle fetcher via the Fyers API v3.

Setup (one-time):
  1. Open a free Fyers trading account: https://fyers.in
  2. Create an API app at https://myapi.fyers.in/dashboard -> note your app_id
     and secret_key.
  3. Install the SDK:   pip install fyers-apiv3
  4. Generate a daily access_token (Fyers tokens expire ~daily; automate this
     with the TOTP flow described in their docs if you want unattended runs):
     https://myapi.fyers.in/docs/

Symbol format for MCX, e.g.:
  "MCX:GOLD26FEBFUT"   (current-month Gold future)
  "MCX:CRUDEOIL26FEBFUT"
Check exact active contract symbols in the Fyers symbol master:
  https://public.fyers.in/sym_details/NSE_FO.csv / MCX_FO.csv

Resolution examples: "5" (5-min), "15" (15-min), "D" (daily).

NOTE: This module is a thin, ready-to-fill template. It is NOT called unless
you choose the "Fyers API" data source in the Streamlit app AND provide an
access token -- the app works fully on uploaded/sample CSVs without this.
"""

from __future__ import annotations

import time
import datetime as dt

import pandas as pd

# Fyers' documented cap for intraday resolutions (1/2/3/5/10/15/20/30/45/60/
# 120/180/240-min): 100 calendar days of data per single API call. Daily
# ("D") resolution has a much larger cap, but we chunk everything the same
# way for simplicity -- it just means fewer, larger chunks for "D".
MAX_DAYS_PER_CALL = 100


def fetch_history(symbol: str, resolution: str, start_date: str, end_date: str,
                   access_token: str, app_id: str) -> pd.DataFrame:
    """Single API call -- covers at most ~100 days for intraday resolutions.
    Use fetch_history_chunked() below for anything longer."""
    try:
        from fyers_apiv3 import fyersModel
    except ImportError as e:
        raise ImportError(
            "fyers-apiv3 is not installed. Run: pip install fyers-apiv3"
        ) from e

    fyers = fyersModel.FyersModel(client_id=app_id, is_async=False, token=access_token)

    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",       # 1 = yyyy-mm-dd strings for range_from/to
        "range_from": start_date,
        "range_to": end_date,
        "cont_flag": "1",
    }
    resp = fyers.history(data=data)

    if resp.get("s") != "ok":
        raise RuntimeError(f"Fyers API error: {resp}")

    candles = resp["candles"]  # [ts, open, high, low, close, volume]
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def fetch_history_chunked(symbol: str, resolution: str, start_date: str, end_date: str,
                           access_token: str, app_id: str,
                           chunk_days: int = MAX_DAYS_PER_CALL,
                           pause_seconds: float = 1.0,
                           progress_callback=None) -> pd.DataFrame:
    """
    Fetches a long date range by splitting it into <=100-day chunks (Fyers'
    documented per-call limit for intraday resolutions), calling
    fetch_history() for each, and concatenating + de-duplicating the result.

    progress_callback, if given, is called as progress_callback(chunk_start,
    chunk_end, chunk_index, total_chunks) after each chunk completes -- handy
    for printing progress in a script or updating a Streamlit progress bar.
    """
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    if start > end:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    # Build the list of (chunk_start, chunk_end) windows, oldest first.
    windows = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        windows.append((cur, chunk_end))
        cur = chunk_end + dt.timedelta(days=1)

    frames = []
    for i, (w_start, w_end) in enumerate(windows, start=1):
        chunk_df = fetch_history(
            symbol=symbol, resolution=resolution,
            start_date=w_start.isoformat(), end_date=w_end.isoformat(),
            access_token=access_token, app_id=app_id,
        )
        frames.append(chunk_df)
        if progress_callback:
            progress_callback(w_start, w_end, i, len(windows))
        if i < len(windows):
            time.sleep(pause_seconds)  # be polite to the rate limiter

    if not frames:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return combined
