"""Transparent, rule-based snowpack indicators.

Every indicator takes a time-sorted list of :class:`Observation` and returns an
:class:`Indicator` (value, level none/watch/warning, human reason). ``now``
defaults to the last observation. Thresholds are textbook simplifications, not
calibrated for any site; see README "Limits".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

from .sources import SENSOR_HEIGHTS_CM, Observation

LEVELS = ("none", "watch", "warning")


@dataclass(frozen=True)
class Indicator:
    name: str
    value: float | None
    unit: str
    level: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _level(value: float, watch: float, warning: float) -> str:
    return "warning" if value >= warning else "watch" if value >= watch else "none"


def _window(obs: Sequence[Observation], now: datetime, hours: float) -> list[Observation]:
    start = now - timedelta(hours=hours)
    return [o for o in obs if start <= o.time <= now]


def _na(name: str, unit: str, why: str) -> Indicator:
    return Indicator(name, None, unit, "none", why)


def _median3(values: list[float]) -> list[float]:
    """Running median of 3 (ends untouched): removes single-sample spikes."""
    if len(values) < 3:
        return list(values)
    a = np.asarray(values, dtype=float)
    out = a.copy()
    out[1:-1] = np.median(np.stack([a[:-2], a[1:-1], a[2:]]), axis=0)
    return out.tolist()


def accumulate_new_snow(hs: list[float], min_step: float = 1.0) -> float:
    """Sum of HS rises of at least ``min_step`` cm above a falling reference.

    The reference follows HS downwards (settlement), and moves up only when HS
    exceeds it by ``min_step``; the whole rise is then counted. Slow snowfall
    therefore accumulates, sensor noise below ``min_step`` does not.
    """
    s = _median3(hs)
    if not s:
        return 0.0
    ref, total = s[0], 0.0
    for v in s[1:]:
        if v < ref:
            ref = v
        elif v - ref >= min_step:
            total += v - ref
            ref = v
    return round(total, 1)


def new_snow(obs: Sequence[Observation], now: datetime | None = None, hours: int = 24,
             watch: float | None = None, warning: float | None = None,
             min_step: float = 1.0) -> Indicator:
    """New snow HN over ``hours`` (cm) = accumulate_new_snow(HS series, min_step).

    Defaults: 24 h watch >= 20, warning >= 30 cm; 72 h watch >= 30, warning >= 50 cm
    (rough rules of thumb for critical new-snow loading).
    """
    name = f"new_snow_{hours}h"
    if not obs:
        return _na(name, "cm", "no data")
    now = now or obs[-1].time
    if watch is None or warning is None:
        watch, warning = (20.0, 30.0) if hours <= 24 else (30.0, 50.0)
    hs = [o.hs_cm for o in _window(obs, now, hours) if o.hs_cm is not None]
    if len(hs) < 3:
        return _na(name, "cm", "fewer than 3 snow height samples in window")
    hn = accumulate_new_snow(hs, min_step)
    return Indicator(name, hn, "cm", _level(hn, watch, warning),
                     f"{hn:g} cm new snow in {hours} h (thresholds {watch:g}/{warning:g} cm)")


def wind_drift(obs: Sequence[Observation], now: datetime | None = None,
               lo: float = 4.0, hi: float = 12.0,
               watch: float = 8.0, warning: float = 15.0) -> Indicator:
    """Wind-drift loading proxy (cm) = HN24 * share of 24 h samples with mean wind in [lo, hi] m/s.

    Wind in the 4-12 m/s band can move loose snow; weaker wind does not transport
    it, much stronger wind mostly scours. Watch >= 8, warning >= 15.
    """
    name = "wind_drift"
    if not obs:
        return _na(name, "cm", "no data")
    now = now or obs[-1].time
    winds = [o.vw_ms for o in _window(obs, now, 24) if o.vw_ms is not None]
    if not winds:
        return _na(name, "cm", "no wind data in window")
    hn = new_snow(obs, now, 24).value or 0.0
    share = sum(lo <= w <= hi for w in winds) / len(winds)
    value = round(hn * share, 1)
    return Indicator(name, value, "cm", _level(value, watch, warning),
                     f"HN24 {hn:g} cm x {share:.0%} of time with {lo:g}-{hi:g} m/s wind")


def gradient_c_per_m(o: Observation, margin_cm: float = 5.0, min_depth_m: float = 0.2) -> float | None:
    """Temperature gradient (C/m, positive = warmer below) over the measured snow depth.

    G = (T_low - TSS) / ((HS - h_low) / 100), where T_low is the lowest sensor buried by
    at least ``margin_cm`` and h_low its height above ground in cm. None if the
    measured depth is below ``min_depth_m`` or an input is missing.
    """
    if o.hs_cm is None or o.tss_c is None:
        return None
    for attr, h in SENSOR_HEIGHTS_CM:  # ascending height
        t = getattr(o, attr)
        if t is not None and h + margin_cm <= o.hs_cm:
            depth = (o.hs_cm - h) / 100.0
            return (t - o.tss_c) / depth if depth >= min_depth_m else None
    return None


def temp_gradient(obs: Sequence[Observation], now: datetime | None = None,
                  limit: float = 10.0) -> Indicator:
    """Mean snowpack temperature gradient over 24 h (C/m).

    Watch if mean24 > 10 C/m (a commonly used threshold for faceting, i.e. weak
    layer growth). Warning if also mean72 > 10 C/m with >= 48 h of valid data
    (persistent gradient).
    """
    name = "temp_gradient"
    if not obs:
        return _na(name, "C/m", "no data")
    now = now or obs[-1].time

    def stats(hours: int) -> tuple[float | None, float]:
        pts = [(o.time, g) for o in _window(obs, now, hours) if (g := gradient_c_per_m(o)) is not None]
        if len(pts) < 6:
            return None, 0.0
        span = (pts[-1][0] - pts[0][0]).total_seconds() / 3600
        return float(np.mean([g for _, g in pts])), span

    m24, _ = stats(24)
    if m24 is None:
        return _na(name, "C/m", "too little snow (<0.2 m measured) or missing sensors")
    m72, span72 = stats(72)
    persistent = m72 is not None and m72 > limit and span72 >= 48
    level = "none" if m24 <= limit else "warning" if persistent else "watch"
    why = f"mean gradient {m24:.1f} C/m over 24 h"
    if m72 is not None and span72 >= 48:
        why += f", {m72:.1f} C/m over 72 h"
    return Indicator(name, round(m24, 1), "C/m", level, why + f" (limit {limit:g})")


def rapid_warming(obs: Sequence[Observation], now: datetime | None = None,
                  watch: float = 10.0, warning: float = 15.0,
                  wet_tss: float = -0.5) -> Indicator:
    """Air temperature rise (C) = mean TA of last hour - min TA of last 24 h.

    Watch >= 10, warning >= 15 C. Wet-snow rule: snow >= 10 cm, mean TA of last hour
    > 0 C and mean TSS >= -0.5 C (surface at melting point) is a warning.
    """
    name = "rapid_warming"
    if not obs:
        return _na(name, "C", "no data")
    now = now or obs[-1].time
    w24 = _window(obs, now, 24)
    ta_all = [o.ta_c for o in w24 if o.ta_c is not None]
    recent = [o for o in w24 if o.time > now - timedelta(hours=1)]
    ta_recent = [o.ta_c for o in recent if o.ta_c is not None]
    if len(ta_all) < 6 or not ta_recent:
        return _na(name, "C", "not enough air temperature data")
    ta_now = float(np.mean(ta_recent))
    rise = round(ta_now - min(ta_all), 1)
    level, why = _level(rise, watch, warning), f"air temperature up {rise:g} C in 24 h"
    tss = [o.tss_c for o in recent if o.tss_c is not None]
    hs = [o.hs_cm for o in recent if o.hs_cm is not None]
    if tss and hs and min(hs) >= 10 and ta_now > 0 and float(np.mean(tss)) >= wet_tss:
        level = "warning"
        why += f"; TA {ta_now:.1f} C > 0 with surface at {float(np.mean(tss)):.1f} C: wet-snow risk"
    return Indicator(name, rise, "C", level, why)


def settlement(obs: Sequence[Observation], now: datetime | None = None,
               watch: float = 5.0, warning: float = 10.0) -> Indicator:
    """Settlement rate (% of snow depth per day) over 24 h without new snow.

    rate = (median(first 3 HS) - median(last 3 HS)) / HS_first / dt * 24 * 100, evaluated
    only if HN24 is 0. Watch >= 5 %/day, warning >= 10 %/day (fast compaction, e.g. melt).
    """
    name = "settlement"
    if not obs:
        return _na(name, "%/day", "no data")
    now = now or obs[-1].time
    if (new_snow(obs, now, 24).value or 0) > 0:
        return _na(name, "%/day", "new snow in last 24 h; settlement not evaluated")
    pts = [(o.time, o.hs_cm) for o in _window(obs, now, 24) if o.hs_cm is not None]
    if len(pts) < 8 or (pts[-2][0] - pts[1][0]).total_seconds() < 12 * 3600:
        return _na(name, "%/day", "need >= 12 h of snow height data")
    h0 = float(np.median([h for _, h in pts[:3]]))
    h1 = float(np.median([h for _, h in pts[-3:]]))
    if h0 <= 0:
        return _na(name, "%/day", "no snow cover")
    dt_h = (pts[-2][0] - pts[1][0]).total_seconds() / 3600
    pct = round((h0 - h1) / h0 / dt_h * 24 * 100, 1)
    return Indicator(name, pct, "%/day", _level(pct, watch, warning),
                     f"snow depth {h0:g} -> {h1:g} cm, {pct:g} %/day")


def evaluate_all(obs: Sequence[Observation], now: datetime | None = None) -> list[Indicator]:
    """Run every indicator in a fixed order."""
    return [
        new_snow(obs, now, 24),
        new_snow(obs, now, 72),
        wind_drift(obs, now),
        temp_gradient(obs, now),
        rapid_warming(obs, now),
        settlement(obs, now),
    ]
