import json
from pathlib import Path

import pytest

from snowpack import codec

VECTORS = json.loads((Path(__file__).parents[1] / "examples" / "vectors.json").read_text())
GOOD = [v for v in VECTORS if not v.get("error")]
BAD = [v for v in VECTORS if v.get("error")]


@pytest.mark.parametrize("v", GOOD, ids=lambda v: v["name"])
def test_vectors_decode_and_encode(v):
    raw = bytes.fromhex(v["hex"])
    assert len(raw) == 16
    assert codec.decode(raw) == v["data"]
    assert codec.encode(v["data"]) == raw


@pytest.mark.parametrize("v", BAD, ids=lambda v: v["name"])
def test_vectors_wrong_length_rejected(v):
    with pytest.raises(ValueError):
        codec.decode(bytes.fromhex(v["hex"]))


def test_round_trip_quantisation():
    data = {"hs_mm": 1503, "ta_c": -7.126, "tss_c": -9.994, "t25_c": 0.004, "t50_c": 1.0,
            "t100_c": -0.5, "wind_mean_ms": 5.1, "wind_gust_ms": 13.3, "wind_dir_deg": 91,
            "battery_mv": 3654, "flags": {"event": True}}
    out = codec.decode(codec.encode(data))
    assert out["hs_mm"] == 1503
    assert out["ta_c"] == pytest.approx(-7.13, abs=0.005)
    assert out["tss_c"] == pytest.approx(-9.99, abs=0.005)
    assert out["wind_mean_ms"] == pytest.approx(5.1, abs=0.11)
    assert out["wind_gust_ms"] == pytest.approx(13.3, abs=0.11)
    assert out["wind_dir_deg"] == 90 or out["wind_dir_deg"] == 92
    assert out["battery_mv"] == pytest.approx(3654, abs=5)
    assert out["flags"] == {"low_battery": False, "sensor_fault": False, "event": True}


def test_edge_values_saturate():
    out = codec.decode(codec.encode({"hs_mm": 99999, "ta_c": 999, "tss_c": -999,
                                     "wind_mean_ms": 400, "battery_mv": 9000}))
    assert out["hs_mm"] == 8190
    assert out["ta_c"] == 327.67 and out["tss_c"] == -327.67
    assert out["wind_mean_ms"] == 50.8
    assert out["battery_mv"] == 4540


def test_zero_and_negative_zero_height():
    assert codec.decode(codec.encode({"hs_mm": 0}))["hs_mm"] == 0
    assert codec.decode(codec.encode({"hs_mm": -5}))["hs_mm"] == 0


def test_wind_direction_wraps():
    assert codec.decode(codec.encode({"wind_dir_deg": 360}))["wind_dir_deg"] == 0
    assert codec.decode(codec.encode({"wind_dir_deg": 359}))["wind_dir_deg"] == 0


def test_missing_fields_become_none():
    out = codec.decode(codec.encode({}))
    assert all(v is None for k, v in out.items() if k != "flags")


@pytest.mark.parametrize("n", [0, 1, 15, 17])
def test_wrong_length_raises(n):
    with pytest.raises(ValueError, match="expected 16 bytes"):
        codec.decode(bytes(n))


def test_decode_hex_tolerates_spaces_and_prefix():
    v = GOOD[0]
    spaced = " ".join(v["hex"][i:i + 2] for i in range(0, 32, 2))
    assert codec.decode_hex("0x" + spaced) == v["data"]
    with pytest.raises(ValueError):
        codec.decode_hex("zz")
