"""
Generates synthetic-but-plausible 5-min MCX Gold intraday OHLCV data so the
app can be demoed / sanity-checked without any API access. Includes engineered
volume spikes + displacement moves so order blocks actually get detected.

Run:  python data/make_sample_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample(start="2025-01-01", end="2025-03-31", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end, tz="Asia/Kolkata")

    rows = []
    price = 62000.0  # rough MCX Gold (per 10g) ballpark level

    for day in dates:
        # MCX trading session ~ 09:00 to 23:30 IST; keep it simple: 09:00-17:00
        session = pd.date_range(
            day.replace(hour=9, minute=0), day.replace(hour=17, minute=0), freq="5min"
        )
        base_vol = rng.integers(400, 900)
        for k, ts in enumerate(session):
            drift = rng.normal(0, 6)
            price = max(price + drift, 100)
            open_ = price
            close = max(price + rng.normal(0, 8), 100)
            high = max(open_, close) + abs(rng.normal(0, 4))
            low = min(open_, close) - abs(rng.normal(0, 4))
            volume = max(int(base_vol + rng.normal(0, 120)), 10)

            # engineer occasional order-block-like events: a spike-volume down
            # candle followed by a strong impulsive up move (or mirror)
            if rng.random() < 0.03 and k < len(session) - 4:
                volume = int(volume * rng.uniform(2.0, 3.5))
                if rng.random() < 0.5:
                    close = open_ - abs(rng.normal(15, 5))  # down candle
                else:
                    close = open_ + abs(rng.normal(15, 5))  # up candle
                high = max(open_, close) + abs(rng.normal(0, 3))
                low = min(open_, close) - abs(rng.normal(0, 3))

            price = close
            rows.append([ts, round(open_, 1), round(high, 1), round(low, 1), round(close, 1), volume])

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    return df


if __name__ == "__main__":
    df = make_sample()
    out_path = __file__.replace("make_sample_data.py", "sample_gold_5min.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
