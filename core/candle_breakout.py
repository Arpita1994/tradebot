"""
30-minute candle breakout-and-fill strategy.

On the CLOSE of every 30-min candle:
  - If the candle is green (close > open) AND its body is "decisive" (see
    below), a buy-stop/limit order is placed ABOVE the candle, at close +
    buy_limit_pct of the candle's body. This is a breakout-continuation
    entry (price has to keep pushing up through this level to trigger it),
    not a pullback-into-the-body entry.
  - The stop loss starts out BELOW the whole candle: low - sl_buffer_pct of
    the candle's body -- UNLESS that would put it more than max_loss_points
    below the candle's close, in which case it's capped there instead (see
    "MAX LOSS POINTS" below).
  - What happens after entry depends on exit_mode --
      "target"   : a fixed target = entry + reward_risk * (entry - stop
                   loss) is set once, at entry, and never moves. The stop
                   loss also never moves. Exit is whichever of SL/target is
                   hit first (2:1 by default).
      "trailing" : there is no target at all. Instead, every 30 minutes
                   while the trade is open, if the candle that just closed
                   is ALSO green, the stop loss is recalculated the exact
                   same way (that candle's low - sl_buffer_pct of ITS body
                   -- NOT capped by max_loss_points, see below) and
                   replaces the old stop loss IF the new level is higher
                   (the stop only ever ratchets up, protecting gains -- it
                   never moves down). The trade exits purely when price
                   drops through whatever the current stop is.

MAX LOSS POINTS: a signal candle's low - sl_buffer_pct of its body can
sometimes sit very far below its close (e.g. a candle with a long lower
wick, or a huge body), risking a much bigger loss than intended on THAT
ONE ENTRY. Setting max_loss_points (e.g. 25) caps how far BELOW THAT
CANDLE'S CLOSE the INITIAL stop (set once, at entry) is ever allowed to
sit: the entry stop actually used is
    max(low - sl_buffer_pct * body, close - max_loss_points)
i.e. whichever of the two is tighter (closer to price) wins. This cap
applies ONLY to the very first stop set at entry -- it is intentionally
NOT reapplied on later trailing updates in "trailing" mode, since those
are already only ever ratcheting the stop UP in the trade's favor, and
re-capping every step would fight that ratchet instead of just guarding
against a single bad entry candle. max_loss_points = 0 (the default)
disables the cap entirely.

WHAT MAKES A GREEN CANDLE'S BODY "DECISIVE": a green candle only produces a
signal if its body passes BOTH of these checks --
  1. Relative: body >= body_pct_of_avg% of the average body over the
     preceding body_lookback_candles candles (a candle has to be at least
     this much of "normal" size for the current market to count). While
     there isn't yet enough history to compute that average, this check is
     skipped (not treated as a fail).
  2. Absolute: body >= min_body_points, a manual floor in price points that
     applies regardless of what the recent average happens to be.
A weak/small green candle that fails either check produces no signal at all
-- not a smaller position, just no order.

The order is only valid for the SINGLE next 30-min candle. If that candle's
high never reaches the buy-limit price, the order is cancelled outright --
it does not carry forward. The candle after that gets its own fresh
evaluation (a new green candle, a new order), independent of the cancelled
one.

Only one open position at a time is allowed for whatever single price
series this is run against. The caller runs this once per CE leg and once
per PE leg (each leg is its own independent call), so CE and PE can each
have a trade open at the same time, but neither can have two trades open
against itself simultaneously.

--------------------------------------------------------------------------
A NOTE ON WHAT THIS CANNOT KNOW FROM 30-MIN CANDLES: an OHLC candle only
tells you the open/high/low/close of that 30 minutes, not the order in
which those prices were actually touched. If a single candle's range
touches BOTH the stop loss and the target (or, on the entry candle itself,
both fills the buy-limit AND later reverses through the stop within that
same 30 minutes), there is no way to tell from this data alone which came
first. This code always resolves that ambiguity conservatively -- stop
loss wins any same-candle tie -- so the backtest can only be pessimistic
about an ambiguous candle, never optimistic. Live fills can occasionally
land the other way. The only way to remove this ambiguity entirely is to
re-check the ambiguous candles against a finer resolution (e.g. 1-min)
feed; this function does not do that on its own.
"""

from __future__ import annotations

import pandas as pd

from core.indicators import rolling_avg_body


def _check_exit(open_trade: dict, bar: pd.Series, bar_index: int, max_hold_bars: int):
    """Shared SL/target/max-hold check against a single bar's range, used both
    for bars after the entry bar and for the entry bar itself (same-candle
    fill-then-reverse is checked the exact same way, so a stop breached in
    the very candle that filled the order is never missed). target_price is
    None in trailing mode, so that check is simply skipped."""
    low, high = bar["low"], bar["high"]
    sl_price = open_trade["sl_price"]
    target_price = open_trade["target_price"]

    if low <= sl_price:                 # conservative: SL wins a same-bar tie
        return sl_price, "sl"
    if target_price is not None and high >= target_price:
        return target_price, "target"
    if bar_index - open_trade["entry_bar_index"] >= max_hold_bars:
        return bar["close"], "eod_no_exit"
    return None, None


def _raw_candle_sl(bar: pd.Series, sl_buffer_pct: float) -> float:
    """The plain stop loss a given green candle implies: low - sl_buffer_pct
    of its own body. No max_loss_points cap -- see _entry_candle_sl for the
    capped version, used only at entry."""
    body = bar["close"] - bar["open"]
    return bar["low"] - sl_buffer_pct * body


def _entry_candle_sl(bar: pd.Series, sl_buffer_pct: float, max_loss_points: float) -> float:
    """The INITIAL stop loss set at entry, from the signal candle: same as
    _raw_candle_sl, but capped so it's never more than max_loss_points below
    the candle's close (max_loss_points <= 0 disables the cap). This cap
    exists purely to stop a single wild/wicky signal candle from setting an
    oversized initial risk -- it intentionally does NOT apply to later
    trailing updates (see _trail_sl), since those are already only ever
    ratcheting the stop UP in the trade's favor, and re-capping every step
    would fight that ratchet instead of just guarding the first one."""
    raw_sl = _raw_candle_sl(bar, sl_buffer_pct)
    if max_loss_points and max_loss_points > 0:
        capped_sl = bar["close"] - max_loss_points
        return max(raw_sl, capped_sl)
    return raw_sl


def _trail_sl(open_trade: dict, bar: pd.Series, sl_buffer_pct: float) -> None:
    """If `bar` is green, recompute a candidate stop loss (this candle's low
    - sl_buffer_pct of its own body -- no max_loss_points cap; see
    _entry_candle_sl's docstring for why), and raise the trade's stop to it
    if -- and only if -- that's actually higher than the current stop. The
    stop never moves down, so a pullback candle can't drag it back toward
    the entry."""
    if bar["close"] <= bar["open"]:
        return
    candidate_sl = _raw_candle_sl(bar, sl_buffer_pct)
    if candidate_sl > open_trade["sl_price"]:
        open_trade["sl_price"] = candidate_sl


def run_candle_breakout_backtest(
    df: pd.DataFrame,
    buy_limit_pct: float = 0.20,
    sl_buffer_pct: float = 0.10,
    reward_risk: float = 2.0,
    time_start=None,
    time_end=None,
    max_hold_bars: int = 200,
    body_lookback_candles: int = 10,
    body_pct_of_avg: float = 100.0,
    min_body_points: float = 0.0,
    body_filter_mode: str = "and",
    exit_mode: str = "target",
    max_loss_points: float = 0.0,
) -> pd.DataFrame:
    """
    body_filter_mode controls how the two decisive-body checks (relative %
    of recent average body, and the absolute points floor) combine:
      "relative" -- only the % of average body check applies (min_body_points ignored)
      "absolute" -- only the points floor applies (body_pct_of_avg ignored)
      "and"      -- both must pass (default)
      "or"       -- either one passing is enough

    exit_mode controls how a trade exits once it's open:
      "target"   -- fixed SL + fixed target (reward_risk : 1), set once at
                    entry, neither ever moves (default, unchanged behavior).
      "trailing" -- no target. The SL ratchets up off every subsequent green
                    candle while the trade is open (see _trail_sl above).
                    reward_risk is ignored in this mode.
    """
    if body_filter_mode not in ("relative", "absolute", "and", "or"):
        raise ValueError(f"body_filter_mode must be one of 'relative', 'absolute', "
                          f"'and', 'or' -- got {body_filter_mode!r}")
    if exit_mode not in ("target", "trailing"):
        raise ValueError(f"exit_mode must be 'target' or 'trailing' -- got {exit_mode!r}")
    df = df.reset_index(drop=True)
    n = len(df)
    trades = []
    open_trade = None
    pending = None
    avg_body = rolling_avg_body(df, body_lookback_candles)

    def _record_exit(exit_price, exit_reason, exit_bar):
        pnl = exit_price - open_trade["entry_price"]
        risk = open_trade["risk_points"]
        trades.append({
            "signal_bar_index": open_trade["entry_bar_index"],
            "candle_datetime": open_trade["candle_datetime"],
            "entry_datetime": open_trade["entry_datetime"],
            "direction": "long",
            "entry_price": open_trade["entry_price"],
            "sl_price": open_trade["sl_price"],
            "target_price": open_trade["target_price"],
            "risk_points": risk,
            "exit_datetime": exit_bar["datetime"],
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_points": pnl,
            "r_multiple": (pnl / risk) if risk else None,
        })

    for i in range(n):
        bar = df.iloc[i]

        # 1) manage an already-open trade first (entered on an earlier bar)
        if open_trade is not None:
            exit_price, exit_reason = _check_exit(open_trade, bar, i, max_hold_bars)
            if exit_price is not None:
                _record_exit(exit_price, exit_reason, bar)
                open_trade = None
            elif exit_mode == "trailing":
                _trail_sl(open_trade, bar, sl_buffer_pct)

        # 2) check whether a pending order fills on THIS bar -- it is only
        #    ever checked once, on the single bar right after the green
        #    candle that created it. If it fills, ALSO check this same
        #    bar's own range for an immediate SL/target hit -- a fast
        #    30-min candle can plausibly do both (spike up through the
        #    buy-limit, then reverse through the stop) inside one candle.
        if pending is not None:
            if i == pending["valid_bar"]:
                if open_trade is None and bar["high"] >= pending["buy_limit_price"]:
                    open_trade = {
                        "entry_price": pending["buy_limit_price"],
                        "sl_price": pending["sl_price"],
                        "target_price": pending["target_price"],
                        "risk_points": pending["risk_points"],
                        "entry_bar_index": i,
                        "entry_datetime": bar["datetime"],
                        "candle_datetime": pending["candle_datetime"],
                    }
                    exit_price, exit_reason = _check_exit(open_trade, bar, i, max_hold_bars)
                    if exit_price is not None:
                        _record_exit(exit_price, exit_reason, bar)
                        open_trade = None
                    elif exit_mode == "trailing":
                        _trail_sl(open_trade, bar, sl_buffer_pct)
                pending = None  # filled or expired -- gone either way, never carries forward
            elif i > pending["valid_bar"]:
                pending = None

        # 3) at the close of this candle, look for a fresh green-candle setup
        in_session = True
        if time_start is not None and time_end is not None:
            t = bar["datetime"].time()
            in_session = time_start <= t <= time_end
        is_green = bar["close"] > bar["open"]

        if is_green and in_session and open_trade is None and pending is None and i + 1 < n:
            body = bar["close"] - bar["open"]
            bar_avg_body = avg_body.iloc[i]
            passes_relative = pd.isna(bar_avg_body) or body >= (body_pct_of_avg / 100.0) * bar_avg_body
            passes_absolute = body >= min_body_points

            if body_filter_mode == "relative":
                is_decisive = passes_relative
            elif body_filter_mode == "absolute":
                is_decisive = passes_absolute
            elif body_filter_mode == "or":
                is_decisive = passes_relative or passes_absolute
            else:  # "and"
                is_decisive = passes_relative and passes_absolute

            if body > 0 and is_decisive:
                buy_limit_price = bar["close"] + buy_limit_pct * body
                sl_price = _entry_candle_sl(bar, sl_buffer_pct, max_loss_points)
                risk = buy_limit_price - sl_price
                if risk > 0:
                    target_price = None if exit_mode == "trailing" else buy_limit_price + reward_risk * risk
                    pending = {
                        "buy_limit_price": buy_limit_price,
                        "sl_price": sl_price,
                        "target_price": target_price,
                        "risk_points": risk,
                        "valid_bar": i + 1,
                        "candle_datetime": bar["datetime"],
                    }

    # If a trade is still open when the data simply runs out (never hit SL,
    # target, or the max-hold cutoff), close it at the last bar's close
    # instead of silently dropping it -- same "eod_no_exit" convention as
    # the consolidation-box engine's run_backtest().
    if open_trade is not None and n > 0:
        last_bar = df.iloc[n - 1]
        _record_exit(last_bar["close"], "eod_no_exit", last_bar)

    return pd.DataFrame(trades)