"""
Simple standalone test: fetch August 2026 5-min Crude Oil (MCX) historical
candles via Fyers and print/save the result. Nothing else -- just to confirm
your .env credentials + access token actually work before running the full app.

Usage:
    python generate_token.py     # if you haven't already today
    python test_fyers_connection.py
"""

import os

import pandas as pd
from dotenv import load_dotenv

from fetchers.fyers_fetcher import fetch_history

load_dotenv()

APP_ID = os.getenv("FYERS_APP_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

if not APP_ID or not ACCESS_TOKEN:
    raise SystemExit(
        "Missing FYERS_APP_ID or FYERS_ACCESS_TOKEN in .env. "
        "Run `python generate_token.py` first."
    )

# --- adjust these if needed ---
SYMBOL = "MCX:CRUDEOIL26AUGFUT"   # MCX Crude Oil, August 2026 expiry contract
RESOLUTION = "5"                  # 5-min candles
START_DATE = "2026-08-01"
END_DATE = "2026-08-31"
# --------------------------------
# NOTE: MCX contracts roll monthly. If this symbol errors out or returns
# nothing, check the exact live contract name in Fyers' MCX symbol master:
# https://public.fyers.in/sym_details/MCX_FO.csv (search for "CRUDEOIL").

print(f"Fetching {SYMBOL} [{RESOLUTION}-min] from {START_DATE} to {END_DATE} ...")

df = fetch_history(
    symbol=SYMBOL,
    resolution=RESOLUTION,
    start_date=START_DATE,
    end_date=END_DATE,
    access_token=ACCESS_TOKEN,
    app_id=APP_ID,
)

print(f"\nRows fetched: {len(df)}")
print(df.head())
print(df.tail())

out_path = "data/crude_august_2026.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
