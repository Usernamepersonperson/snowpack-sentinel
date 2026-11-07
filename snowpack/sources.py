"""Data sources: live SLF IMIS stations and a deterministic simulated station."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np

from . import codec

SLF_BASE = "https://measurement-api.slf.ch/public/api/imis"
USER_AGENT = "snowpack-sentinel/0.1 (+https://github.com; educational use)"

# Sensor heights above ground (cm) for the in-snow temperature sensors.
SENSOR_HEIGHTS_CM: tuple[tuple[str, int], ...] = (
    ("t0_c", 0),
    ("t25_c", 25),
    ("t50_c", 50),
    ("t100_c", 100),
)


class SourceError(RuntimeError):
    """Raised when a data source cannot be reached or returns garbage."""


@dataclass(frozen=True)
class Observation:
    """One station sample. ``None`` means missing/invalid."""

    time: datetime
    station: str
    hs_cm: float | None = None  # snow height
    ta_c: float | None = None  # air temperature
    rh_pct: float | None = None
    tss_c: float | None = None  # snow surface temperature
    t0_c: float | None = None  # sensor at ground level
    t25_c: float | None = None  # sensor 25 cm above ground
    t50_c: float | None = None
    t100_c: float | None = None
    vw_ms: float | None = None  # wind mean
    vw_max_ms: float | None = None  # wind gust
    dw_deg: float | None = None  # wind direction


@dataclass(frozen=True)
class Station:
    code: str
    label: str
    lon: float | None
    lat: float | None
    elevation: float | None
    country_code: str | None
    canton_code: str | None
    type: str | None


def _num(value: Any) -> float | None:
    """Return a finite float or None (handles null, strings, NaN, bool)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_time(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_SLF_FIELDS = {
    "hs_cm": "HS",
    "ta_c": "TA_30MIN_MEAN",
    "rh_pct": "RH_30MIN_MEAN",
    "tss_c": "TSS_30MIN_MEAN",
    "t0_c": "TS0_30MIN_MEAN",
    "t25_c": "TS25_30MIN_MEAN",
    "t50_c": "TS50_30MIN_MEAN",
    "t100_c": "TS100_30MIN_MEAN",
    "vw_ms": "VW_30MIN_MEAN",
    "vw_max_ms": "VW_30MIN_MAX",
    "dw_deg": "DW_30MIN_MEAN",
}


def parse_slf_record(rec: dict[str, Any]) -> Observation | None:
    """Parse one SLF measurement record; returns None if it has no valid timestamp."""
    when = _parse_time(rec.get("measure_date"))
    if when is None:
        return None
    values = {attr: _num(rec.get(key)) for attr, key in _SLF_FIELDS.items()}
    return Observation(time=when, station=str(rec.get("station_code", "")), **values)


def parse_slf_measurements(records: list[dict[str, Any]]) -> list[Observation]:
    """Parse and time-sort records, silently skipping malformed ones."""
    obs = [o for r in records if isinstance(r, dict) and (o := parse_slf_record(r))]
    return sorted(obs, key=lambda o: o.time)


def _http_get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise SourceError(f"request failed for {url}: {exc}") from exc


class SlfImisSource:
    """Public SLF IMIS measurement API (WSL Institute for Snow and Avalanche Research SLF).

    Data terms and attribution: see https://measurement-api.slf.ch and SLF's terms of use.
    ``fetch`` is injectable so tests never touch the network.
    """

    def __init__(self, base_url: str = SLF_BASE, timeout: float = 20.0,
                 fetch: Callable[[str], Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._fetch = fetch or (lambda url: _http_get_json(url, timeout))

    def stations(self, canton: str | None = None, type: str | None = None) -> list[Station]:
        raw = self._fetch(f"{self.base_url}/stations")
        if not isinstance(raw, list):
            raise SourceError("unexpected stations response")
        names = {f.name for f in fields(Station)}
        out = []
        for r in raw:
            if not isinstance(r, dict) or "code" not in r:
                continue
            st = Station(**{k: r.get(k) for k in names})
            if canton and (st.canton_code or "").upper() != canton.upper():
                continue
            if type and (st.type or "").upper() != type.upper():
                continue
            out.append(st)
        return sorted(out, key=lambda s: s.code)

    def observations(self, code: str, hours: float = 72.0) -> list[Observation]:
        """Fetch the station's records and keep the trailing ``hours`` window."""
        raw = self._fetch(f"{self.base_url}/station/{code.upper()}/measurements")
        if not isinstance(raw, list):
            raise SourceError("unexpected measurements response")
        obs = parse_slf_measurements(raw)
        if not obs:
            return []
        cutoff = obs[-1].time - timedelta(hours=hours)
        return [o for o in obs if o.time >= cutoff]


# ---------------------------------------------------------------- uplink bridge

def observation_from_uplink(data: dict[str, Any], station: str, when: datetime) -> Observation:
    """Build an Observation from a decoded codec dict."""
    def scaled(key: str, f: float) -> float | None:
        v = data.get(key)
        return None if v is None else v * f

    return Observation(
        time=when, station=station,
        hs_cm=scaled("hs_mm", 0.1), ta_c=data.get("ta_c"), tss_c=data.get("tss_c"),
        t25_c=data.get("t25_c"), t50_c=data.get("t50_c"), t100_c=data.get("t100_c"),
        vw_ms=data.get("wind_mean_ms"), vw_max_ms=data.get("wind_gust_ms"),
        dw_deg=data.get("wind_dir_deg"),
    )


def observation_to_uplink(o: Observation, battery_mv: int = 3700) -> bytes:
    """Quantise an Observation into the 16-byte uplink frame."""
    return codec.encode({
        "hs_mm": None if o.hs_cm is None else o.hs_cm * 10,
        "ta_c": o.ta_c, "tss_c": o.tss_c, "t25_c": o.t25_c, "t50_c": o.t50_c,
        "t100_c": o.t100_c, "wind_mean_ms": o.vw_ms, "wind_gust_ms": o.vw_max_ms,
        "wind_dir_deg": o.dw_deg, "battery_mv": battery_mv,
    })


def through_codec(observations: list[Observation]) -> list[Observation]:
    """Round-trip every observation through the LoRaWAN codec (what the cloud would see)."""
    return [observation_from_uplink(codec.decode(observation_to_uplink(o)), o.station, o.time)
            for o in observations]


# ------------------------------------------------------------------- simulation

def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class SimulatedStation:
    """Eight-day synthetic scenario (hours from start in brackets).

    [0-30] calm, mild, 55 cm base; [30-48] storm: ~38 cm snow with 8-11 m/s wind;
    [48-96] settling, wind easing; [96-168] clear, cold nights over a 90 cm pack;
    [168-192] rapid warming to above 0 C.
    """

    code: str = "SIM1"
    start: datetime = field(default_factory=lambda: datetime(2026, 1, 10, tzinfo=timezone.utc))
    days: int = 8
    step_min: int = 30
    seed: int = 7

    def observations(self) -> list[Observation]:
        rng = np.random.default_rng(self.seed)
        n = self.days * 24 * 60 // self.step_min
        t = np.arange(n) * self.step_min / 60.0
        s = np.sin(2 * np.pi * (t % 24 - 9) / 24)  # daily cycle: min 03:00, max 15:00

        knots = [0, 24, 48, 72, 96, 120, 144, 168, 192]
        mean = np.interp(t, knots, [-2, -3, -5, -8, -11, -12, -11, -9, 4])
        amp = np.interp(t, knots, [2, 2, 3, 4, 4, 4, 4, 4, 6])
        clear = np.interp(t, knots, [0.1, 0.1, 0.2, 0.5, 1, 1, 1, 1, 1])
        ta = mean + amp * s + rng.normal(0, 0.15, n)
        tss = np.minimum(0.0, ta - clear * 4 * np.maximum(0, -s) + clear * np.maximum(0, s))

        new_snow = 38 * _smoothstep((t - 30) / 18)
        hs = (55 - 0.01 * t + new_snow
              - 0.15 * 38 * (1 - np.exp(-np.maximum(t - 48, 0) / 36))
              - 5 * _smoothstep((t - 174) / 16) + rng.normal(0, 0.2, n))

        storm = np.exp(-(((t - 45) / 14) ** 2))
        vw = np.clip(2.0 + 8.0 * storm + 1.0 * _smoothstep((t - 96) / 24) + rng.normal(0, 0.5, n), 0, None)
        gust = vw * 1.5 + rng.uniform(0, 1.0, n)
        dw = (300 + 20 * rng.standard_normal(n)) % 360

        out = []
        for i in range(n):
            snow = hs[i]
            ground = -0.2

            def profile(h: int, i: int = i, snow: float = snow) -> float:
                if h + 5 <= snow:  # sensor buried: linear profile ground -> surface
                    return round(ground + (tss[i] - ground) * h / snow, 2)
                return round(float(ta[i]), 2)  # sensor in air

            out.append(Observation(
                time=self.start + timedelta(minutes=self.step_min * i), station=self.code,
                hs_cm=round(float(snow), 1), ta_c=round(float(ta[i]), 2), rh_pct=80.0,
                tss_c=round(float(tss[i]), 2), t0_c=ground, t25_c=profile(25),
                t50_c=profile(50), t100_c=profile(100), vw_ms=round(float(vw[i]), 1),
                vw_max_ms=round(float(gust[i]), 1), dw_deg=round(float(dw[i]), 0),
            ))
        return out


__all__ = [
    "Observation", "Station", "SlfImisSource", "SimulatedStation", "SourceError",
    "parse_slf_record", "parse_slf_measurements", "observation_from_uplink",
    "observation_to_uplink", "through_codec",
]
