"""Compact 16-byte LoRaWAN uplink for a DIY snow station.

Byte layout (big-endian), also implemented in ``codec/ttn_chirpstack.js``:

======  =====  ==========================================================
Offset  Bytes  Field
======  =====  ==========================================================
0-1     2      bits 15..13 flags, bits 12..0 snow height HS in mm
2-3     2      air temperature TA, int16, 0.01 C
4-5     2      snow surface temperature TSS, int16, 0.01 C
6-7     2      temperature 25 cm above ground, int16, 0.01 C
8-9     2      temperature 50 cm above ground, int16, 0.01 C
10-11   2      temperature 100 cm above ground, int16, 0.01 C
12      1      wind mean, uint8, 0.2 m/s
13      1      wind gust, uint8, 0.2 m/s
14      1      wind direction, uint8, 2 degrees
15      1      battery, uint8, 2000 mV + 10 mV * value
======  =====  ==========================================================

Flags: bit0 low_battery, bit1 sensor_fault, bit2 event (wake-up by threshold).
Invalid / missing sentinels: HS 0x1FFF, temperatures 0x8000, uint8 fields 0xFF.
Out-of-range values saturate to the nearest representable value.
"""

from __future__ import annotations

import struct
from typing import Any, Mapping

FRAME_LEN = 16
FLAG_BITS = {"low_battery": 0, "sensor_fault": 1, "event": 2}
HS_INVALID = 0x1FFF
T_INVALID = -32768
U8_INVALID = 0xFF
_TEMP_FIELDS = ("ta_c", "tss_c", "t25_c", "t50_c", "t100_c")


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _u8(value: float | None, scale: float, offset: float = 0.0) -> int:
    if value is None:
        return U8_INVALID
    return _clamp(round((value - offset) / scale), 0, 254)


def encode(data: Mapping[str, Any]) -> bytes:
    """Encode a dict (keys as returned by :func:`decode`) into a 16-byte frame.

    Missing or ``None`` fields are written as the invalid sentinel.
    """
    hs = data.get("hs_mm")
    hs_raw = HS_INVALID if hs is None else _clamp(round(hs), 0, HS_INVALID - 1)
    flags = data.get("flags") or {}
    flag_bits = sum(1 << bit for name, bit in FLAG_BITS.items() if flags.get(name))
    temps = []
    for key in _TEMP_FIELDS:
        v = data.get(key)
        temps.append(T_INVALID if v is None else _clamp(round(v * 100), -32767, 32767))
    wd = data.get("wind_dir_deg")
    wd_raw = U8_INVALID if wd is None else round(wd / 2) % 180
    return struct.pack(
        ">H5h4B",
        (flag_bits << 13) | hs_raw,
        *temps,
        _u8(data.get("wind_mean_ms"), 0.2),
        _u8(data.get("wind_gust_ms"), 0.2),
        wd_raw,
        _u8(data.get("battery_mv"), 10, 2000),
    )


def decode(payload: bytes) -> dict[str, Any]:
    """Decode a 16-byte frame. Raises ``ValueError`` if the length is wrong."""
    if len(payload) != FRAME_LEN:
        raise ValueError(f"expected {FRAME_LEN} bytes, got {len(payload)}")
    w0, ta, tss, t25, t50, t100, wm, wg, wd, bat = struct.unpack(">H5h4B", payload)
    hs = w0 & 0x1FFF
    out: dict[str, Any] = {"hs_mm": None if hs == HS_INVALID else hs}
    for key, raw in zip(_TEMP_FIELDS, (ta, tss, t25, t50, t100)):
        out[key] = None if raw == T_INVALID else raw / 100
    out["wind_mean_ms"] = None if wm == U8_INVALID else wm * 2 / 10
    out["wind_gust_ms"] = None if wg == U8_INVALID else wg * 2 / 10
    out["wind_dir_deg"] = None if wd >= 180 else wd * 2
    out["battery_mv"] = None if bat == U8_INVALID else 2000 + bat * 10
    flag_bits = w0 >> 13
    out["flags"] = {name: bool(flag_bits >> bit & 1) for name, bit in FLAG_BITS.items()}
    return out


def decode_hex(text: str) -> dict[str, Any]:
    """Decode a hex string (spaces and ``0x`` prefix tolerated)."""
    cleaned = "".join(text.split()).removeprefix("0x")
    try:
        raw = bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError(f"not valid hex: {text!r}") from exc
    return decode(raw)
