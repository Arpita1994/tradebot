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

import datetime as dt

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


def _peak_high_datetime(df: pd.DataFrame, entry_bar_index: int, exit_bar_index: int):
    """For require_close_above_loss_reference: finds the datetime of
    whichever bar had the highest `high` during a trade's ENTIRE lifetime,
    from its entry/fill bar through its exit bar (inclusive on both ends
    -- entry_bar_index == exit_bar_index for a same-candle fill-then-exit
    is just that one bar). This is NOT necessarily the entry bar itself --
    a trade can run for several candles before reversing, and the
    reference is wherever price actually peaked during the whole ride."""
    window = df.iloc[entry_bar_index:exit_bar_index + 1]
    peak_idx = window["high"].idxmax()
    return df.loc[peak_idx, "datetime"]


def _sl_anchor(bar: pd.Series, sl_basis: str) -> float:
    """The price the sl_buffer_pct offset is measured down FROM.
      "wick_low" (default, original behavior) -- the candle's actual low,
                 wick included.
      "body_low" -- the candle's body low, i.e. min(open, close) -- for a
                 green candle this is just `open`. Ignores any lower wick
                 entirely, so the resulting SL sits tighter (closer to
                 price) whenever the candle has a lower wick below its open."""
    if sl_basis == "body_low":
        return min(bar["open"], bar["close"])
    return bar["low"]


def _raw_candle_sl(bar: pd.Series, sl_buffer_pct: float, sl_basis: str = "wick_low") -> float:
    """The plain stop loss a given green candle implies: sl_anchor -
    sl_buffer_pct of its own body, where sl_anchor is the wick low (default)
    or the body low, per sl_basis. No max_loss_points cap -- see
    _entry_candle_sl for the capped version, used only at entry."""
    body = bar["close"] - bar["open"]
    return _sl_anchor(bar, sl_basis) - sl_buffer_pct * body


def _entry_candle_sl(bar: pd.Series, sl_buffer_pct: float, max_loss_points: float, sl_basis: str = "wick_low") -> float:
    """The INITIAL stop loss set at entry, from the signal candle: same as
    _raw_candle_sl, but capped so it's never more than max_loss_points below
    the candle's close (max_loss_points <= 0 disables the cap). This cap
    exists purely to stop a single wild/wicky signal candle from setting an
    oversized initial risk -- it intentionally does NOT apply to later
    trailing updates (see _trail_sl), since those are already only ever
    ratcheting the stop UP in the trade's favor, and re-capping every step
    would fight that ratchet instead of just guarding the first one. This
    cap behaves identically regardless of sl_basis -- it's always measured
    from the candle's close, not from whichever anchor sl_basis picked."""
    raw_sl = _raw_candle_sl(bar, sl_buffer_pct, sl_basis)
    if max_loss_points and max_loss_points > 0:
        capped_sl = bar["close"] - max_loss_points
        return max(raw_sl, capped_sl)
    return raw_sl


def _trail_sl(open_trade: dict, bar: pd.Series, sl_buffer_pct: float, trail_min_body_points: float, sl_basis: str = "wick_low") -> None:
    """If `bar` is green AND its body is at least trail_min_body_points (a
    minimum size in price points -- trail_min_body_points <= 0 disables this
    check, so every green candle qualifies, same as before), recompute a
    candidate stop loss (this candle's sl_anchor - sl_buffer_pct of its own
    body -- no max_loss_points cap; see _entry_candle_sl's docstring for
    why), and raise the trade's stop to it if -- and only if -- that's
    actually higher than the current stop. The stop never moves down, so a
    pullback candle can't drag it back toward the entry. This only looks at
    the single candle that just closed -- there is no multi-candle lookback
    here, by design: the trailing stop always follows just the previous
    candle."""
    if bar["close"] <= bar["open"]:
        return
    body = bar["close"] - bar["open"]
    if trail_min_body_points and trail_min_body_points > 0 and body < trail_min_body_points:
        return
    candidate_sl = _raw_candle_sl(bar, sl_buffer_pct, sl_basis)
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
    trail_min_body_points: float = 0.0,
    entry_cutoff=None,
    sl_basis: str = "wick_low",
    require_close_above_loss_reference: bool = False,
    entry_floor=None,
    initial_day: dt.date | None = None,
    initial_loss_active: bool = False,
    initial_loss_ref_time=None,
) -> pd.DataFrame:
    """
    sl_basis controls what the sl_buffer_pct offset is measured down from,
    for BOTH the initial entry stop and every trailing update:
      "wick_low" -- the candle's actual low, wick included (default,
                    original/unchanged behavior).
      "body_low" -- the candle's body low, i.e. min(open, close) (= open,
                    for a green candle). Ignores any lower wick, so the
                    resulting stop sits tighter (closer to price) whenever
                    the candle has a lower wick below its open. The 20-point
                    max_loss_points cap at entry is completely unaffected --
                    it's always measured from the candle's close either way.

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

    trail_min_body_points (trailing mode only) -- a green candle only moves
    the trailing stop if its OWN body is at least this many price points.
    0 (default) means every green candle qualifies, same as before this was
    added. This looks at just the single candle that just closed each time
    -- no averaging or multi-candle lookback -- it's simply a minimum size
    filter so a tiny, insignificant green candle can't become the new trail
    anchor. (Separate from body_lookback_candles/body_pct_of_avg/
    min_body_points/body_filter_mode above, which only gate NEW entries.)

    DAY SQUARE-OFF: when time_end is given, any trade still open the moment
    a candle's time reaches time_end is force-closed right there at that
    candle's close, tagged "day_square_off" -- it never carries into the
    next day's candles. An unfilled pending order at that point is dropped
    the same way rather than left to fill tomorrow. This applies on EVERY
    day time_end is crossed, however many days of data `df` spans (the
    ATM-options caller may now hand this function several days of the same
    contract's candles at once -- see entry_cutoff below); each day gets
    its own fresh square-off, and a new signal can only start a new
    position the following day once in_session is true again. Without
    time_end (or with only time_start set), no square-off happens and a
    trade can run for as long as max_hold_bars / the data allows, same as
    before this was added.

    entry_cutoff -- if given (a pandas-comparable timestamp), no NEW signal
    is generated and no NEW pending order is created for a candle whose
    datetime is >= entry_cutoff. This is for the ATM-options caller: once a
    leg's own window has ended (its strike is no longer the freshly-picked
    ATM strike), it shouldn't open fresh positions on that now-stale
    contract. It does NOT affect a trade that's already open (or already
    pending) at that point -- that trade keeps being managed (SL/target/
    trailing-stop checks) using whatever bars of `df` extend past the
    cutoff, all the way to its own natural exit, exactly as if there were
    no cutoff at all. The cutoff stops the strategy from looking for a NEW
    position on a symbol once its window has passed -- it never forces an
    existing position closed just because the window ended.

    require_close_above_loss_reference -- a "loss-recovery loop" for the
    rest of a calendar day, triggered by a losing trade and broken by a
    winning one:

      1. Whenever a trade closes at a loss (pnl_points < 0 -- any exit
         reason: sl, day_square_off, or eod_no_exit), the loop STARTS (or,
         if already running, its reference REFRESHES): find whichever bar
         had the HIGHEST high during that losing trade's ENTIRE lifetime,
         from its entry/fill bar through its exit bar (inclusive both
         ends -- see _peak_high_datetime) -- not necessarily the entry bar
         itself, since a trade can run for several candles before
         reversing, and price may have peaked partway through the ride,
         not right at entry. Remember that peak bar's datetime. Call this
         loss_ref_time.
      2. While the loop is running, every later green/decisive candle must
         pass one extra check before it's allowed to signal: look up the
         candle at loss_ref_time (on THIS SAME symbol's own price data --
         see entry_floor/initial_loss_ref_time below for what happens
         across a strike roll) and require the new candidate's CLOSE to be
         strictly above THAT ONE candle's HIGH. Fails -> negated outright,
         no order, and the loop keeps running for the next candle (it does
         not fall back to the plain rule after one skip). If no candle
         exists at exactly loss_ref_time in this data (e.g. a data gap),
         the check can't be evaluated and is treated as a pass rather than
         silently blocking forever on a data problem.
      3. The loop BREAKS the moment any trade taken under it (or, per (1),
         even the original triggering loss's very next trade) closes at a
         WIN (pnl_points >= 0) -- back to the plain rule with no
         restriction, until another loss starts a fresh loop. A trade that
         loses again while the loop is already running does NOT break it
         -- it refreshes loss_ref_time to ITS OWN peak-high bar instead
         (the most recent loss is always what later candles are measured
         against).
      4. The loop, and loss_ref_time, are cleared at every new calendar day
         found in `df` -- a new day always starts clean, loop only (re)starts
         once that day has its own loss.

    Default False: unchanged original behavior, no restriction at all.

    entry_floor -- if given (a pandas-comparable timestamp), no NEW signal
    is generated for a candle whose datetime is < entry_floor, though
    earlier bars (if present in `df`) are still used for lookups like the
    loss-reference check above and the rolling body average. This is for
    the ATM-options caller: when a strike has just rolled and the new
    leg's own reference candle (loss_ref_time from BEFORE the roll) needs
    to be read off this new symbol's data, the caller may hand this
    function extra bars from earlier that same day purely so that lookup
    has something to read -- entry_floor (the leg's real window_start)
    stops those extra early bars from being treated as tradeable
    candles in their own right.

    initial_day / initial_loss_active / initial_loss_ref_time -- seeds the
    loop's state from a PRIOR call (e.g. an earlier leg on the same
    option_type, before an ATM strike roll), so the loop can correctly
    keep running across a strike change within the same calendar day even
    though each leg/strike runs as its own separate call to this function.
    If `df`'s first bar is on a different calendar day than `initial_day`,
    the normal same-day-boundary reset logic fires immediately and clears
    everything anyway, exactly as if this were a fresh day. The caller
    reads back the ending state via the returned DataFrame's
    `.attrs["ending_day"]` / `.attrs["ending_loss_active"]` /
    `.attrs["ending_loss_ref_time"]` to pass into the NEXT chronological
    leg of the same option_type (see run_legs_candle_breakout in
    atm_options.py).
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

    # -- require_close_above_loss_reference state (see docstring) --
    current_day = initial_day            # calendar date of the day currently being tracked
    loss_active = initial_loss_active    # is the loss-recovery loop currently running?
    loss_ref_time = initial_loss_ref_time  # datetime of the PEAK-HIGH bar during the most
                                            # recent losing trade's lifetime -- later candles must
                                            # close above THAT candle's high

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
        day_cutoff_hit = time_end is not None and bar["datetime"].time() >= time_end

        if require_close_above_loss_reference:
            bar_date = bar["datetime"].date()
            if bar_date != current_day:
                current_day = bar_date
                loss_active = False
                loss_ref_time = None

        # 1) manage an already-open trade first (entered on an earlier bar)
        if open_trade is not None:
            exit_price, exit_reason = _check_exit(open_trade, bar, i, max_hold_bars)
            if exit_price is not None:
                entry_bar_idx = open_trade["entry_bar_index"]
                _record_exit(exit_price, exit_reason, bar)
                open_trade = None
                if require_close_above_loss_reference:
                    if trades[-1]["pnl_points"] < 0:
                        loss_active, loss_ref_time = True, _peak_high_datetime(df, entry_bar_idx, i)
                    else:
                        loss_active, loss_ref_time = False, None
            elif exit_mode == "trailing":
                _trail_sl(open_trade, bar, sl_buffer_pct, trail_min_body_points, sl_basis)

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
                        if require_close_above_loss_reference:
                            if trades[-1]["pnl_points"] < 0:
                                # same-bar fill-then-exit -- entry and exit are this one bar
                                loss_active, loss_ref_time = True, _peak_high_datetime(df, i, i)
                            else:
                                loss_active, loss_ref_time = False, None
                    elif exit_mode == "trailing":
                        _trail_sl(open_trade, bar, sl_buffer_pct, trail_min_body_points, sl_basis)
                pending = None  # filled or expired -- gone either way, never carries forward
            elif i > pending["valid_bar"]:
                pending = None

        # 2.5) day square-off: force-close any trade still open once this
        # candle reaches the day's defined time_end, instead of letting it
        # carry into the next day's candles -- a fresh, unfilled pending
        # order is also dropped here rather than left to fill tomorrow.
        if day_cutoff_hit:
            if open_trade is not None:
                entry_bar_idx = open_trade["entry_bar_index"]
                _record_exit(bar["close"], "day_square_off", bar)
                open_trade = None
                if require_close_above_loss_reference:
                    if trades[-1]["pnl_points"] < 0:
                        loss_active, loss_ref_time = True, _peak_high_datetime(df, entry_bar_idx, i)
                    else:
                        loss_active, loss_ref_time = False, None
            pending = None

        # 3) at the close of this candle, look for a fresh green-candle setup
        in_session = True
        if time_start is not None and time_end is not None:
            t = bar["datetime"].time()
            in_session = time_start <= t <= time_end
        before_cutoff = entry_cutoff is None or bar["datetime"] < entry_cutoff
        after_floor = entry_floor is None or bar["datetime"] >= entry_floor
        is_green = bar["close"] > bar["open"]

        if (is_green and in_session and before_cutoff and after_floor and not day_cutoff_hit
                and open_trade is None and pending is None and i + 1 < n):
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

            passes_loss_reference = True
            if require_close_above_loss_reference and loss_active and loss_ref_time is not None:
                ref_rows = df[df["datetime"] == loss_ref_time]
                if not ref_rows.empty:
                    passes_loss_reference = bar["close"] > ref_rows.iloc[0]["high"]
                # else: no candle found at loss_ref_time in this data (e.g. a gap, or
                # this symbol's fetched history doesn't reach back that far) -- can't
                # evaluate the check, so don't silently block on a data problem; leave
                # passes_loss_reference True.

            if body > 0 and is_decisive and passes_loss_reference:
                buy_limit_price = bar["close"] + buy_limit_pct * body
                sl_price = _entry_candle_sl(bar, sl_buffer_pct, max_loss_points, sl_basis)
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
        entry_bar_idx = open_trade["entry_bar_index"]
        _record_exit(last_bar["close"], "eod_no_exit", last_bar)
        if require_close_above_loss_reference:
            if trades[-1]["pnl_points"] < 0:
                loss_active, loss_ref_time = True, _peak_high_datetime(df, entry_bar_idx, n - 1)
            else:
                loss_active, loss_ref_time = False, None

    result = pd.DataFrame(trades)
    # Ending loop state, for a caller running consecutive legs (per
    # option_type, across ATM strike rolls) to carry into the NEXT leg's
    # initial_day/initial_loss_active/initial_loss_ref_time -- see this
    # function's docstring and run_legs_candle_breakout in atm_options.py.
    # Set unconditionally (cheap, harmless) so the caller doesn't need to
    # know in advance whether this particular leg had any bars at all.
    result.attrs["ending_day"] = current_day
    result.attrs["ending_loss_active"] = loss_active
    result.attrs["ending_loss_ref_time"] = loss_ref_time
    return result