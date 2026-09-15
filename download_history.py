"""
Downloads several months of MCX historical candles from Fyers, handling the
100-day-per-call limit automatically (fetches in chunks, stitches them
together, saves one clean CSV ready to use in the app).

Usage:
    python generate_token.py       # if today's token isn't fresh yet
    python download_history.py

Edit the settings below to change symbol / months back / resolution.
"""

import datetime as dt
import os

from dotenv import load_dotenv

from fetchers.fyers_fetcher import fetch_history_chunked

load_dotenv()

APP_ID = os.getenv("FYERS_APP_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

if not APP_ID or not ACCESS_TOKEN:
    raise SystemExit(
        "Missing FYERS_APP_ID or FYERS_ACCESS_TOKEN in .env. "
        "Run `python generate_token.py` first."
    )

# --- adjust these ---
SYMBOL = "MCX:CRUDEOIL26SEPFUT"   # current near-month contract; cont_flag=1
                                  # (set in fyers_fetcher.py) stitches in the
                                  # correct historical front-month data for
                                  # each date, so this covers past months too
RESOLUTION = "5"                 # "5", "15", "D", etc.
MONTHS_BACK = 6                  # how many months of history to pull
OUTPUT_CSV = "data/crudeoil_last_6_months.csv"
# ---------------------

end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=MONTHS_BACK * 30)

print(f"Fetching {SYMBOL} [{RESOLUTION}-min] from {start_date} to {end_date} "
      f"(in <=100-day chunks) ...")


def show_progress(w_start, w_end, i, total):
    print(f"  chunk {i}/{total}: {w_start} to {w_end} -- done")


df = fetch_history_chunked(
    symbol=SYMBOL,
    resolution=RESOLUTION,
    start_date=start_date.isoformat(),
    end_date=end_date.isoformat(),
    access_token=ACCESS_TOKEN,
    app_id=APP_ID,
    progress_callback=show_progress,
)

print(f"\nTotal rows fetched: {len(df)}")
if not df.empty:
    print(df.head())
    print(df.tail())

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")
print("You can now load this file in the app via Data source -> Upload CSV.")
