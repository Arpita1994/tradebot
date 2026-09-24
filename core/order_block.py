"""
Consolidation-box order block detection.

Definition used here (replaces the earlier single-candle volume-spike
definition, per your request):

1. Find a CONSOLIDATION BOX: a run of at least `min_candles` consecutive
   candles whose combined high-low range stays tight -- specifically, the
   box's total height (highest high to lowest low across the run) must
   stay within `tightness_multiplier` times the average single-candle
   range over the preceding `baseline_lookback` bars. That "tight relative
   to normal candle size" check is what makes it a genuine compression /
   consolidation, not just any arbitrary N-bar stretch (any N candles
   trivially fit inside their own high-low range, so a plain N-bar box
   isn't meaningful on its own -- it has to be unusually tight).
   The box is grown greedily bar-by-bar for as long as it stays within
   that tightness bound (capped at `max_candles` as a safety limit), so
   what you get is the box of MAXIMUM consolidation starting from each
   scan point, not just the minimum-required length.

2. Watch for a BREAKOUT: the first candle after the box whose CLOSE moves
   outside the box (above box_high = bullish breakout, below box_low =
   bearish breakout).

3. Watch for a RETEST: after the breakout, price coming back to touch the
   broken edge of the box. A retest "holds" (counts toward `min_touches`)
   unless price closes all the way back through the box's far side, which
   invalidates the breakout entirely (failed breakout -- no entry).
   Once the required number of holding retests has occurred, an entry
   fires on the next bar, in the breakout's direction.

So "buying the retest" means: box forms -> price breaks out -> price comes
back down (for a bullish breakout) to retest the top of the box -> if it
holds, buy on the next bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from core.indicators import rolling_avg_candle_range


@dataclass
class OrderBlock:
    index: int                      # index of the BREAKOUT candle (retest watch starts right after this)
    datetime: pd.Timestamp          # breakout candle's datetime
    direction: Literal["bullish", "bearish"]   # breakout direction
    ob_high: float                  # consolidation box's top
    ob_low: float                   # consolidation box's bottom
    volume: float                   # breakout candle's volume (informational)
    box_start_index: int = -1       # first candle of the consolidation box
    box_end_index: int = -1         # last candle of the consolidation box
    box_length: int = 0             # how many candles consolidated
    mitigated_at: int | None = None  # index where the Nth valid retest occurred
    touch_count: int = 0


def detect_consolidation_boxes(
    df: pd.DataFrame,
    min_candles: int = 7,
    tightness_multiplier: float = 2.5,
    baseline_lookback: int = 20,
    max_candles: int = 40,
) -> list[OrderBlock]:
    avg_range = rolling_avg_candle_range(df, baseline_lookback)
    n = len(df)
    blocks: list[OrderBlock] = []

    i = baseline_lookback
    while i < n:
        baseline = avg_range.iloc[i]
        if pd.isna(baseline) or baseline <= 0:
            i += 1
            continue

        # Greedily grow a box starting at i for as long as it stays tight.
        box_high = df["high"].iloc[i]
        box_low = df["low"].iloc[i]
        j = i
        while j + 1 < n and (j - i + 1) < max_candles:
            cand_high = max(box_high, df["high"].iloc[j + 1])
            cand_low = min(box_low, df["low"].iloc[j + 1])
            if (cand_high - cand_low) <= tightness_multiplier * baseline:
                box_high, box_low, j = cand_high, cand_low, j + 1
            else:
                break

        length = j - i + 1
        if length < min_candles:
            i += 1  # not enough consolidation starting here -- slide forward and retry
            continue

        # Valid box [i, j]. Look forward for the first close outside it.
        breakout_index, direction = None, None
        k = j + 1
        while k < n:
            close = df["close"].iloc[k]
            if close > box_high:
                breakout_index, direction = k, "bullish"
                break
            if close < box_low:
                breakout_index, direction = k, "bearish"
                break
            k += 1

        if breakout_index is not None:
            blocks.append(OrderBlock(
                index=breakout_index,
                datetime=df["datetime"].iloc[breakout_index],
                direction=direction,
                ob_high=box_high, ob_low=box_low,
                volume=df["volume"].iloc[breakout_index],
                box_start_index=i, box_end_index=j, box_length=length,
            ))
            i = breakout_index + 1  # keep scanning after this breakout
        else:
            i = j + 1  # no breakout yet within available data -- move past the box

    return blocks


def mark_mitigations(df: pd.DataFrame, blocks: list[OrderBlock], min_touches: int = 1) -> None:
    """
    Mutates each OrderBlock in place: finds the RETEST of the broken box
    edge after the breakout, setting mitigated_at to the bar index where the
    Nth valid (holding) retest occurs, and touch_count to how many there
    were by then.

    A "touch" is any bar whose range overlaps [ob_low, ob_high] (the box).
    A touch holds (counts toward min_touches) unless that same bar's close
    breaks all the way through the box's FAR side -- i.e. for a bullish
    breakout, closing back below ob_low means the breakout failed and price
    fully re-entered/fell through the old range, which invalidates it
    permanently (no entry from this box, no matter how many good retests
    came before). Mirrored for a bearish breakout (close above ob_high).

    min_touches=1 (default) fires on the very first retest.
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
                break  # failed breakout -- stop looking, no entry from this block

            touches += 1
            if touches >= min_touches:
                ob.mitigated_at = j
                ob.touch_count = touches
                break