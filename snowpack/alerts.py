"""Combine indicators into per-station alert JSON, with de-duplication."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from typing import Any, Sequence

from . import DISCLAIMER
from .indicators import LEVELS, Indicator, evaluate_all
from .sources import Observation


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class AlertTracker:
    """Emit an alert when an indicator escalates, or repeats after ``repeat_after_h`` hours.

    Falling levels and flapping around a threshold stay silent until the
    repeat interval has passed, so a persistent condition is a reminder, not spam.
    """

    def __init__(self, repeat_after_h: float = 24.0) -> None:
        self.repeat_after = timedelta(hours=repeat_after_h)
        self._last: dict[tuple[str, str], tuple[int, datetime]] = {}

    def update(self, station: str, now: datetime, indicators: Sequence[Indicator]) -> list[dict[str, Any]]:
        emitted = []
        for ind in indicators:
            rank = LEVELS.index(ind.level)
            if rank == 0:
                continue
            key = (station, ind.name)
            prev = self._last.get(key)
            if prev is None or rank > prev[0] or now - prev[1] >= self.repeat_after:
                self._last[key] = (rank, now)
                emitted.append({
                    "station": station, "time": iso(now), "indicator": ind.name,
                    "level": ind.level, "value": ind.value, "unit": ind.unit,
                    "reason": ind.reason,
                })
        return emitted


def station_report(station: str, now: datetime, indicators: Sequence[Indicator],
                   alerts: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Alert JSON for one station at one instant."""
    overall = max((i.level for i in indicators), key=LEVELS.index, default="none")
    return {
        "station": station,
        "generated_at": iso(now),
        "overall_level": overall,
        "indicators": [i.to_dict() for i in indicators],
        "alerts": list(alerts),
        "disclaimer": DISCLAIMER,
    }


def replay(station: str, obs: Sequence[Observation], step_h: float = 1.0,
           lookback_h: float = 72.0, tracker: AlertTracker | None = None) -> list[dict[str, Any]]:
    """Walk through the series hourly, evaluating on a trailing window; return de-duplicated alerts."""
    if not obs:
        return []
    tracker = tracker or AlertTracker()
    times = [o.time for o in obs]
    alerts: list[dict[str, Any]] = []
    t, end = times[0], times[-1]
    while t <= end:
        lo = bisect_left(times, t - timedelta(hours=lookback_h))
        hi = bisect_right(times, t)
        window = obs[lo:hi]
        alerts += tracker.update(station, t, evaluate_all(window, t))
        t += timedelta(hours=step_h)
    return alerts
