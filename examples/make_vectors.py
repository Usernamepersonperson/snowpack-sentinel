"""Regenerate examples/vectors.json (the first vector is hand-computed and kept as is)."""
import json
from pathlib import Path

from snowpack import codec

out = Path(__file__).with_name("vectors.json")
vectors = json.loads(out.read_text())[:1]
cases = {
    "all_invalid": {},
    "deep_snow_storm": {
        "hs_mm": 4210, "ta_c": -12.34, "tss_c": -15.5, "t25_c": -8.0, "t50_c": -4.25,
        "t100_c": -1.0, "wind_mean_ms": 10.4, "wind_gust_ms": 25.0, "wind_dir_deg": 358,
        "battery_mv": 3300, "flags": {"event": True},
    },
    "warm_wet_snow": {
        "hs_mm": 650, "ta_c": 7.5, "tss_c": 0.0, "t25_c": 0.0, "t50_c": 0.0,
        "t100_c": 3.2, "wind_mean_ms": 0.0, "wind_gust_ms": 1.2, "wind_dir_deg": 0,
        "battery_mv": 2900, "flags": {"low_battery": True, "sensor_fault": True},
    },
    "saturation_limits": {
        "hs_mm": 8190, "ta_c": 327.67, "tss_c": -327.67, "t25_c": 0.01, "t50_c": -0.01,
        "t100_c": 0.0, "wind_mean_ms": 50.8, "wind_gust_ms": 50.8, "wind_dir_deg": 358,
        "battery_mv": 4540,
        "flags": {"low_battery": True, "sensor_fault": True, "event": True},
    },
}
for name, d in cases.items():
    raw = codec.encode(d)
    vectors.append({"name": name, "hex": raw.hex(), "data": codec.decode(raw)})
vectors.append({"name": "truncated_10_bytes", "hex": "04d2fea2fd8ffe3eff38", "error": True})
vectors.append({"name": "too_long_17_bytes", "hex": "00" * 17, "error": True})
out.write_text(json.dumps(vectors, indent=2) + "\n")
