"""
Volume-based order block detection.

Definition used here (as chosen for this tool):
A BULLISH order block is the last DOWN candle (close < open) that shows a volume
spike (volume > spike_multiplier * average volume over `lookback` prior bars),
immediately followed by an impulsive UP move (displacement) that closes above
the candle's high within `displacement_window` bars.

A BEARISH order block is the mirror: last UP candle with a volume spike,
followed by an impulsive DOWN move closing below the candle's low.

An order block is "mitigated" (usable as an entry zone) the first time price
later trades back into [ob_low, ob_high]. With min_touches > 1, instead of
entering on the first touch, the zone must be tested and HOLD (price enters
the zone but doesn't close through the far side) that many times before an
entry fires -- treating repeated tests as support/resistance confirmation
rather than a single-touch "institutional order fill" entry. A close through
the far side at any point invalidates the block entirely (no more entries
from it), regardless of how many touches came before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from core.indicators import rolling_avg_volume


@dataclass
class OrderBlock:
    index: int
    datetime: pd.Timestamp
    direction: Literal["bullish", "bearish"]
    ob_high: float
    ob_low: float
    volume: float
    mitigated_at: int | None = None
    touch_count: int = 0


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int,
    spike_multiplier: float = 1.5,
    displacement_window: int = 3,
) -> list[OrderBlock]:
    avg_vol = rolling_avg_volume(df, lookback)
    blocks: list[OrderBlock] = []

    n = len(df)
    for i in range(lookback, n - displacement_window):
        row = df.iloc[i]
        if pd.isna(avg_vol.iloc[i]):
            continue
        is_volume_spike = row["volume"] > (spike_multiplier * avg_vol.iloc[i])
        if not is_volume_spike:
            continue

        future = df.iloc[i + 1: i + 1 + displacement_window]
        is_down_candle = row["close"] < row["open"]
        is_up_candle = row["close"] > row["open"]

        if is_down_candle and (future["close"] > row["high"]).any():
            blocks.append(OrderBlock(
                index=i, datetime=row["datetime"], direction="bullish",
                ob_high=row["high"], ob_low=row["low"], volume=row["volume"],
            ))
        elif is_up_candle and (future["close"] < row["low"]).any():
            blocks.append(OrderBlock(
                index=i, datetime=row["datetime"], direction="bearish",
                ob_high=row["high"], ob_low=row["low"], volume=row["volume"],
            ))

    return blocks


def mark_mitigations(df: pd.DataFrame, blocks: list[OrderBlock], min_touches: int = 1) -> None:
    """
    Mutates each OrderBlock in place, setting mitigated_at to the index of the
    bar where the Nth valid touch (min_touches) occurs, and touch_count to how
    many valid touches happened by then.

    A "touch" is any bar whose range overlaps [ob_low, ob_high]. A touch
    "holds" (counts toward min_touches) unless that same bar's close breaks
    through the far side of the block, which invalidates it permanently --
    no entry fires from an invalidated block, no matter how many touches it
    had before breaking.

    min_touches=1 (default) reproduces the original single-touch/"first
    mitigation" behavior.
    """
    for ob in blocks:
        touches = 0
        for j in range(ob.index + 1, len(df)):
            bar = df.iloc[j]
            touched = bar["low"] <= ob.ob_high and bar["high"] >= ob.ob_low
            if not touched:
                continue

            if ob.direction == "bullish":
                broke_through = bar["close"] < ob.ob_low
            else:
                broke_through = bar["close"] > ob.ob_high

            if broke_through:
                break  # invalidated -- stop looking, no entry from this block

            touches += 1
            if touches >= min_touches:
                ob.mitigated_at = j
                ob.touch_count = touches
                break