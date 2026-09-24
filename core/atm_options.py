"""
Shared logic for the live ATM options backtest: current active option month
only (chosen from the actual MCX Crude Oil options expiry, not just the
calendar month), CE and PE tracked in parallel, ATM strike re-picked every
N hours -- with the underlying's price fetched live from Fyers, never from
a downloaded CSV.

Used by both backtest_atm_options_4h.py (terminal script) and app.py
(the "ATM Options" data source, click-to-run in the Streamlit sidebar).

--------------------------------------------------------------------------
WHY RE-PICK THE STRIKE PER WINDOW INSTEAD OF ONCE: an ATM option's premium
is only a continuous, meaningful price series while it's tracking the SAME
strike. The moment the ATM strike rolls (because the underlying moved by a
strike step), the premium jumps to a different contract's price scale --
not a real market move. So every window is its own independent mini-backtest
on whichever single strike was ATM at the start of that window, for CE and
PE separately. Trades from every window are combined afterwards (never the
raw prices).

WHY "CURRENT MONTH" IS DISCOVERED FROM FYERS, NOT COMPUTED FROM A CALENDAR
RULE: MCX Crude Oil's expiry calendar doesn't follow a fixed "19th of the
month" pattern (an earlier version of this code assumed that, and it broke:
the real Sept 2026 expiry turned out to be the 21st, not the 19th -- MCX
expiry dates shift around to track international crude benchmark contracts
and can land earlier OR later than a fixed day). So instead of guessing,
compute_current_option_month() just asks Fyers directly: try the current
calendar month's futures contract, and if Fyers says it's invalid (i.e. it
has actually expired and rolled off), try next month. Whichever one Fyers
accepts is the real current contract -- no calendar math, no guessing.

WHY NO CSV: the underlying's price is fetched directly from Fyers for a
trailing lookback window ending today, using the continuous-contract flag
(cont_flag=1) the same way the rest of this project already relies on --
no separate download step, no local file to keep in sync.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from fetchers.fyers_fetcher import fetch_history
from core.strategy import StrategyParams, generate_signals
from core.backtest import run_backtest
from core.candle_breakout import run_candle_breakout_backtest
from core.metrics import summarize

SYMBOL_PREFIX = "CRUDEOILM"  # Mini Crude Oil (lot size 10 bbl), per your Fyers screenshot --
                              # NOT the standard CRUDEOIL contract (lot size 100 bbl) this
                              # tool originally targeted. Different expiry calendar too
                              # (e.g. 17 Sept / 15 Oct / 17 Nov 2026), which is exactly why
                              # compute_current_option_month() below asks Fyers directly
                              # instead of assuming any fixed day-of-month.
STRIKE_STEP = 50            # ASSUMED, NOT CONFIRMED for CRUDEOILM -- carried over from the
                              # standard CRUDEOIL contract's confirmed spacing. Your screenshot
                              # only showed one strike (10000), which is consistent with either
                              # a 50 or 100 step, so this could be wrong. Please check the
                              # "Option Chain" tab on Fyers for CRUDEOILM and confirm the actual
                              # gap between adjacent listed strikes, then update this if it's
                              # not 50 -- an incorrect step means nearest_strike() will often
                              # compute a strike that isn't actually listed, which shows up as
                              # a "no_data" leg exactly like the expired-contract issue did.
MAX_MONTHS_TO_PROBE = 4     # safety cap so a persistent API problem can't loop forever


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + n
    return total // 12, total % 12 + 1


def _symbol_is_valid(app_id: str, access_token: str, symbol: str) -> bool:
    """Asks Fyers directly whether a contract symbol is currently tradable,
    by requesting a single day of daily candles for it (cheap: resolution
    'D', tiny date range). Fyers returns 'Invalid symbol provided' for a
    contract that has expired/rolled off, and a normal (possibly empty for
    a non-trading day, but not an error) response otherwise."""
    today = dt.date.today()
    try:
        fetch_history(
            symbol=symbol, resolution="D",
            start_date=(today - dt.timedelta(days=5)).isoformat(),
            end_date=today.isoformat(),
            access_token=access_token, app_id=app_id,
        )
        return True
    except Exception:
        return False


def compute_current_option_month(app_id: str, access_token: str,
                                  today: dt.date | None = None) -> tuple[int, int, str]:
    """Returns (year, month, mcode) for whichever contract month Fyers
    currently accepts as a live Crude Oil futures symbol -- tries this
    calendar month first, then rolls forward one month at a time until one
    is accepted. No expiry-date guessing: Fyers' own answer is the source
    of truth, since MCX's actual expiry calendar doesn't follow a fixed
    day-of-month rule (it tracks international crude benchmark contracts
    and can land earlier or later than any fixed assumption)."""
    today = today or dt.date.today()
    year, month = today.year, today.month
    for _ in range(MAX_MONTHS_TO_PROBE):
        mcode = month_code(year, month)
        if _symbol_is_valid(app_id, access_token, f"MCX:{SYMBOL_PREFIX}{mcode}FUT"):
            return year, month, mcode
        year, month = _add_months(year, month, 1)
    # Fell through without a hit -- most likely an auth/connectivity problem
    # rather than every month genuinely being invalid. Return this month's
    # code anyway so the caller's later fetch produces a clear error message.
    return today.year, today.month, month_code(today.year, today.month)


def month_code(year: int, month: int) -> str:
    """e.g. (2026, 9) -> '26SEP' (matches Fyers' MCX contract naming)."""
    return f"{year % 100:02d}{dt.date(year, month, 1).strftime('%b').upper()}"


def nearest_strike(price: float, step: int = STRIKE_STEP) -> int:
    return int(round(price / step) * step)


def windows_in_range(start_dt: dt.datetime, end_dt: dt.datetime, hours: int):
    """Fixed clock-hour blocks (e.g. 09:00-13:00, 13:00-17:00, ...) walked
    forward from the first day's midnight, clipped to [start_dt, end_dt]."""
    cur = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    step = dt.timedelta(hours=hours)
    windows = []
    while cur < end_dt:
        w_end = cur + step
        if w_end > start_dt:
            windows.append((max(cur, start_dt), min(w_end, end_dt)))
        cur = w_end
    return windows


def fetch_underlying(app_id: str, access_token: str, mcode: str,
                      lookback_days: int, resolution: str = "5") -> pd.DataFrame:
    """Live fetch (no CSV) of the current contract's underlying futures price,
    for the trailing `lookback_days` up to today. cont_flag=1 (set inside
    fetch_history) stitches in the correct historical front-month data even
    though the symbol names the CURRENT contract month."""
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback_days)
    symbol = f"MCX:{SYMBOL_PREFIX}{mcode}FUT"
    df = fetch_history(
        symbol=symbol, resolution=resolution,
        start_date=start_date.isoformat(), end_date=end_date.isoformat(),
        access_token=access_token, app_id=app_id,
    )
    if not df.empty and df["datetime"].dt.tz is not None:
        df = df.copy()
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df.sort_values("datetime").reset_index(drop=True)


def build_legs(underlying: pd.DataFrame, mcode: str, option_types: list[str],
               window_hours: int, strike_step: int = STRIKE_STEP) -> list[dict]:
    """One leg per (option_type, run of consecutive windows that all land on
    the SAME ATM strike), strike picked from the underlying's most recent
    close at or before each window's start.

    Windows are still checked every `window_hours` to see whether the ATM
    strike should roll, but consecutive windows that come out to the same
    strike are merged into a single leg spanning all of them (window_end
    pushed out to the last one) instead of being run as separate,
    independently-truncated legs -- a leg only ends where the strike
    actually changes (or the data does). Each leg also carries
    "fetch_until": the overall end of the available underlying data, so the
    caller can fetch this leg's symbol beyond its own window_end and let an
    already-open trade run to its natural exit instead of being cut off at
    the window boundary (see run_legs_candle_breakout's entry_cutoff use)."""
    if underlying.empty:
        return []

    data_start_dt = underlying["datetime"].min().to_pydatetime()
    data_end_dt = underlying["datetime"].max().to_pydatetime()

    all_windows = windows_in_range(data_start_dt, data_end_dt, window_hours)
    times = underlying["datetime"]
    windows = [
        (w_start, w_end) for (w_start, w_end) in all_windows
        if ((times >= w_start) & (times < w_end)).any()
    ]

    legs = []
    for opt in option_types:
        current = None
        for w_start, w_end in windows:
            ref_rows = underlying[underlying["datetime"] <= w_start]
            if ref_rows.empty:
                continue
            ref_price = float(ref_rows.iloc[-1]["close"])
            strike = nearest_strike(ref_price, strike_step)
            symbol = f"MCX:{SYMBOL_PREFIX}{mcode}{strike}{opt}"

            if current is not None and current["symbol"] == symbol:
                # Same strike still ATM -- extend the running leg instead of
                # closing it out and opening an identical new one.
                current["window_end"] = w_end
                continue

            if current is not None:
                legs.append(current)
            current = {
                "window_start": w_start,
                "window_end": w_end,
                "option_type": opt,
                "underlying_ref_price": ref_price,
                "strike": strike,
                "symbol": symbol,
                "fetch_until": data_end_dt,
            }
        if current is not None:
            legs.append(current)

    legs.sort(key=lambda leg: (leg["window_start"], leg["option_type"]))
    return legs


def _fetch_leg_df(leg: dict, app_id: str, access_token: str, resolution: str,
                   fetch_cache: dict, fetch_end: dt.datetime | None = None) -> pd.DataFrame | None:
    """Shared per-leg fetch used by every strategy runner below: pulls (and
    caches, per symbol+day-range, so a strike carried across consecutive
    windows isn't re-fetched) the option's candles, timezone-normalizes them,
    and clips to this leg's window. Returns None if nothing usable came back
    (caller decides how to record that).

    fetch_end, if given, extends the fetch (and the clip's upper bound)
    past leg["window_end"] up to fetch_end instead of stopping at the
    window -- used by run_legs_candle_breakout so a trade still open when
    the window ends can keep being tracked, on this same symbol's data,
    all the way to its own natural exit (see candle_breakout's
    entry_cutoff). Without it (the default), behavior is unchanged: clipped
    to the leg's own window."""
    clip_end = max(leg["window_end"], fetch_end) if fetch_end is not None else leg["window_end"]
    key = (leg["symbol"], leg["window_start"].date(), clip_end.date())
    if key not in fetch_cache:
        fetch_cache[key] = fetch_history(
            symbol=leg["symbol"], resolution=resolution,
            start_date=leg["window_start"].date().isoformat(),
            end_date=clip_end.date().isoformat(),
            access_token=access_token, app_id=app_id,
        )
    day_df = fetch_cache[key]
    if day_df.empty:
        return day_df

    if day_df["datetime"].dt.tz is not None:
        day_df = day_df.copy()
        day_df["datetime"] = day_df["datetime"].dt.tz_localize(None)
    return day_df[(day_df["datetime"] >= leg["window_start"].replace(tzinfo=None))
                  & (day_df["datetime"] <= clip_end.replace(tzinfo=None))].reset_index(drop=True)


def run_legs(legs: list[dict], app_id: str, access_token: str, resolution: str,
             time_start: dt.time, time_end: dt.time, strategy_kwargs: dict,
             progress_callback=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches each leg's option data, runs the consolidation-box order-block
    strategy independently per leg, and returns (window_summary_df,
    combined_trades_df)."""
    fetch_cache: dict = {}
    all_trades = []
    window_results = []

    for n, leg in enumerate(legs, start=1):
        try:
            df = _fetch_leg_df(leg, app_id, access_token, resolution, fetch_cache)
        except Exception as e:
            window_results.append({**leg, "status": "no_data", "error": str(e), "rows": 0,
                                    "trades": 0, "pnl_points": None, "win_rate_pct": None})
            if progress_callback:
                progress_callback(n, len(legs), leg, "no_data")
            continue

        if df is None or df.empty:
            window_results.append({**leg, "status": "empty", "rows": 0, "trades": 0,
                                    "pnl_points": None, "win_rate_pct": None})
            if progress_callback:
                progress_callback(n, len(legs), leg, "empty")
            continue

        params = StrategyParams(start_time=time_start, end_time=time_end, **strategy_kwargs)
        signals = generate_signals(df, params)
        trades = run_backtest(df, signals)

        if not trades.empty:
            trades = trades.copy()
            trades["window_start"] = leg["window_start"]
            trades["option_type"] = leg["option_type"]
            trades["leg_symbol"] = leg["symbol"]
            all_trades.append(trades)

        stats = summarize(trades)
        window_results.append({
            **leg, "status": "ok", "rows": len(df),
            "trades": stats["total_trades"],
            "pnl_points": stats["total_pnl_points"],
            "win_rate_pct": stats["win_rate_pct"],
        })
        if progress_callback:
            progress_callback(n, len(legs), leg, "ok")

    window_df = pd.DataFrame(window_results)
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return window_df, combined


def run_legs_candle_breakout(legs: list[dict], app_id: str, access_token: str, resolution: str,
                              time_start: dt.time, time_end: dt.time, breakout_kwargs: dict,
                              progress_callback=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same per-leg fetch/window machinery as run_legs, but drives the 30-min
    candle breakout-and-retest-fill strategy (core/candle_breakout.py)
    instead of the consolidation-box order block. Each leg (one CE window,
    one PE window) is its own independent run, so a CE trade and a PE trade
    can be open at the same time -- but a leg's OWN window no longer forces
    its trade closed: this fetches the leg's symbol data through
    leg["fetch_until"] (the overall backtest end, not just window_end) and
    passes entry_cutoff=window_end so no NEW position opens on this symbol
    once its window has passed, while a trade already open (or pending)
    keeps being tracked on this same leg's data all the way to its own
    natural exit -- SL, target/trailing stop, the max-hold cutoff, or truly
    running out of data. build_legs() already merges consecutive
    same-strike windows into one leg, so this only ever creates a genuine
    cutoff at an actual strike roll."""
    fetch_cache: dict = {}
    all_trades = []
    window_results = []

    for n, leg in enumerate(legs, start=1):
        try:
            df = _fetch_leg_df(leg, app_id, access_token, resolution, fetch_cache,
                                fetch_end=leg.get("fetch_until"))
        except Exception as e:
            window_results.append({**leg, "status": "no_data", "error": str(e), "rows": 0,
                                    "trades": 0, "pnl_points": None, "win_rate_pct": None})
            if progress_callback:
                progress_callback(n, len(legs), leg, "no_data")
            continue

        if df is None or df.empty:
            window_results.append({**leg, "status": "empty", "rows": 0, "trades": 0,
                                    "pnl_points": None, "win_rate_pct": None})
            if progress_callback:
                progress_callback(n, len(legs), leg, "empty")
            continue

        trades = run_candle_breakout_backtest(
            df, time_start=time_start, time_end=time_end,
            entry_cutoff=leg["window_end"].replace(tzinfo=None), **breakout_kwargs,
        )

        if not trades.empty:
            trades = trades.copy()
            trades["window_start"] = leg["window_start"]
            trades["option_type"] = leg["option_type"]
            trades["leg_symbol"] = leg["symbol"]
            all_trades.append(trades)

        stats = summarize(trades)
        window_results.append({
            **leg, "status": "ok", "rows": len(df),
            "trades": stats["total_trades"],
            "pnl_points": stats["total_pnl_points"],
            "win_rate_pct": stats["win_rate_pct"],
        })
        if progress_callback:
            progress_callback(n, len(legs), leg, "ok")

    window_df = pd.DataFrame(window_results)
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return window_df, combined