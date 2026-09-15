"""
Combines all the requested inputs into entry signals:
  - time of day window
  - volume going up
  - order block (mitigation of a fresh order block)
  - lookback time (used for volume avg + order block detection window)
  - trend analysis with N lookback candles (trade only with the trend)
  - % of order block used as stop loss

Output: a list of dicts, one per signal, ready for the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime

import pandas as pd

from core.indicators import volume_going_up, trend_direction
from core.order_block import detect_order_blocks, mark_mitigations


@dataclass
class StrategyParams:
    start_time: dtime
    end_time: dtime
    lookback_time: int              # bars used for volume-avg & OB detection window
    volume_spike_multiplier: float  # how far above average volume counts as "going up"
    ob_spike_multiplier: float      # volume spike multiplier required to qualify as an order block
    displacement_window: int        # bars allowed for the impulsive move confirming an OB
    sl_pct_of_ob: float              # % of order block range added beyond the OB as stop loss
    trend_lookback_candles: int      # N candles for trend analysis
    reward_risk: float = 2.0         # target = risk * this multiple (not one of the named
                                      # inputs, but needed to define an exit -- defaults to 2:1)
    require_trend_alignment: bool = True
    min_touches: int = 1             # 1 = enter on first touch (original behavior);
                                      # >1 = require the zone to hold N touches first


def generate_signals(df: pd.DataFrame, params: StrategyParams):
    df = df.reset_index(drop=True)

    vol_up = volume_going_up(df, params.lookback_time, params.volume_spike_multiplier)
    trend = trend_direction(df, params.trend_lookback_candles)
    time_mask = (df["datetime"].dt.time >= params.start_time) & \
                (df["datetime"].dt.time <= params.end_time)

    blocks = detect_order_blocks(
        df,
        lookback=params.lookback_time,
        spike_multiplier=params.ob_spike_multiplier,
        displacement_window=params.displacement_window,
    )
    mark_mitigations(df, blocks, min_touches=params.min_touches)

    signals = []
    used_blocks = set()

    for ob in blocks:
        if ob.mitigated_at is None:
            continue
        i = ob.mitigated_at
        if i >= len(df):
            continue
        if not time_mask.iloc[i]:
            continue
        if not bool(vol_up.iloc[i]):
            continue
        if (ob.direction, ob.index) in used_blocks:
            continue

        bar_trend = trend.iloc[i]
        if params.require_trend_alignment:
            if ob.direction == "bullish" and bar_trend != "up":
                continue
            if ob.direction == "bearish" and bar_trend != "down":
                continue

        ob_range = ob.ob_high - ob.ob_low
        if ob_range <= 0:
            continue

        if i + 1 >= len(df):
            continue
        entry_bar = df.iloc[i + 1]
        entry_price = entry_bar["open"]

        if ob.direction == "bullish":
            sl_price = ob.ob_low - (params.sl_pct_of_ob / 100.0) * ob_range
            risk = entry_price - sl_price
            target_price = entry_price + params.reward_risk * risk
            direction = "long"
        else:
            sl_price = ob.ob_high + (params.sl_pct_of_ob / 100.0) * ob_range
            risk = sl_price - entry_price
            target_price = entry_price - params.reward_risk * risk
            direction = "short"

        if risk <= 0:
            continue

        used_blocks.add((ob.direction, ob.index))
        signals.append({
            "signal_bar_index": i + 1,
            "entry_datetime": entry_bar["datetime"],
            "direction": direction,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "target_price": target_price,
            "risk_points": risk,
            "ob_datetime": ob.datetime,
            "ob_high": ob.ob_high,
            "ob_low": ob.ob_low,
            "trend_at_entry": bar_trend,
            "touch_count": ob.touch_count,
        })

    return signals