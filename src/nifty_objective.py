"""
A real, noisy, expensive-to-evaluate black box built from NIFTY 50 open/close prices.

Strategy (no look-ahead): at the close of day t, go long if Close_t > SMA_L(Close)_t,
otherwise stay in cash. The decision is executed at the next day's OPEN, and the
position is held open-to-open. A switch costs `cost` (one-way, as a log-return).

Tuning variable: the moving-average look-back L (trading days), searched on
x = log(L), L in [5, 250].

Noisy evaluation y(L): annualised excess log-return of the strategy over buy-and-hold
(both open-to-open) on ONE randomly drawn 1-year window (252 trading days) of the
2016-2026 sample. Each call is a fresh window, so repeated calls at the same L differ;
this mimics the usual BO situation of an objective observed with noise.
The target f(L) = mean over all windows (computable here, unknown in practice).

The price file data/nifty50_daily_open_close.csv is not distributed with the package
(see README, section Data). It is read on first use, so scripts that import this module
but do not use the NIFTY 50 problem run without it.
"""
import numpy as np
import pandas as pd

import os as _os, _paths
CSV = _os.path.join(_paths.DATA, "nifty50_daily_open_close.csv")
LMIN, LMAX, WARMUP, WIN = 5, 250, 250, 252
_cache = {}


def _load():
    """Read the price file once; return {'DATA', 'O', 'C'}."""
    if not _cache:
        if not _os.path.exists(CSV):
            raise FileNotFoundError(
                f"{CSV} not found. The NIFTY 50 prices are not distributed with this package; "
                "run  python scripts/get_nifty50.py  to download them (see README, section Data).")
        d = pd.read_csv(CSV, parse_dates=["Date"])
        _cache.update(DATA=d, O=d["Open"].to_numpy(), C=d["Close"].to_numpy())
    return _cache


def __getattr__(name):
    """Module attributes DATA, O and C, loaded on first access (keeps `no.C` working)."""
    if name in ("DATA", "O", "C"):
        return _load()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def daily_excess(L, cost=0.0005):
    """Per-day open-to-open log return of the strategy minus buy-and-hold, days t = WARMUP..T-3."""
    d = _load(); O, C = d["O"], d["C"]
    L = int(round(L))
    sma = pd.Series(C).rolling(L).mean().to_numpy()
    pos = (C > sma).astype(float)                 # decided at close t
    r_oo = np.log(O[2:] / O[1:-1])                # open_{t+1} -> open_{t+2}, indexed by t
    p = pos[:-2]
    switch = np.abs(np.diff(np.r_[0.0, p]))
    strat = p * r_oo - cost * switch
    return (strat - r_oo)[WARMUP:]                # excess over buy-and-hold


def window_starts():
    n = len(daily_excess(20))
    return np.arange(0, n - WIN + 1)


def evaluate(L, rng):
    """One noisy evaluation: annualised excess return on a random 1-year window."""
    ex = daily_excess(L)
    s = rng.integers(0, len(ex) - WIN + 1)
    return 252 * ex[s:s + WIN].mean()


def true_objective(L):
    """f(L): average over all 1-year windows (the quantity BO is trying to maximise)."""
    ex = daily_excess(L)
    cs = np.r_[0, np.cumsum(ex)]
    return (252 / WIN * (cs[WIN:] - cs[:-WIN])).mean()
