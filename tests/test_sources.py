import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from snowpack.sources import (SimulatedStation, SlfImisSource, SourceError,
                              parse_slf_measurements, through_codec)

FIX = Path(__file__).parent / "fixtures"
MEAS = json.loads((FIX / "slf_measurements_synthetic.json").read_text())
STATIONS = json.loads((FIX / "slf_stations_synthetic.json").read_text())


def test_parser_skips_bad_records_and_sorts():
    obs = parse_slf_measurements(MEAS)
    assert [o.time.strftime("%H:%M") for o in obs] == ["09:30", "10:00", "10:30", "11:00"]
    assert all(o.time.tzinfo is not None for o in obs)


def test_parser_full_record():
    o = parse_slf_measurements(MEAS)[1]
    assert (o.hs_cm, o.ta_c, o.tss_c, o.t0_c, o.t25_c, o.t50_c, o.t100_c) == (
        80.0, -4.5, -6.0, -0.1, -1.0, -2.5, -4.4)
    assert (o.vw_ms, o.vw_max_ms, o.dw_deg, o.rh_pct) == (3.0, 6.0, 200.0, 70.0)


def test_parser_nulls_and_missing_become_none():
    o = parse_slf_measurements(MEAS)[2]
    assert o.hs_cm is None and o.t25_c is None and o.vw_ms is None and o.rh_pct is None
    assert o.ta_c == -4.0 and o.t100_c == -4.3


def test_parser_string_number_and_junk():
    first, _, _, junk = parse_slf_measurements(MEAS)
    assert first.hs_cm == 79.5 and first.tss_c is None
    assert junk.hs_cm is None  # "n/a"
    assert junk.ta_c is None  # bool is not a temperature
    assert junk.tss_c is None  # inf


def test_stations_filter():
    src = SlfImisSource(fetch=lambda url: STATIONS)
    assert [s.code for s in src.stations()] == ["AAA1", "BBB2", "CCC3"]
    assert [s.code for s in src.stations(canton="gr")] == ["AAA1", "CCC3"]
    assert [s.code for s in src.stations(canton="GR", type="wind")] == ["CCC3"]


def test_observations_url_and_hours_window():
    seen = []

    def fetch(url):
        seen.append(url)
        return MEAS

    src = SlfImisSource(base_url="https://example.invalid/api/", fetch=fetch)
    obs = src.observations("tst1", hours=0.75)
    assert seen == ["https://example.invalid/api/station/TST1/measurements"]
    assert [o.time.strftime("%H:%M") for o in obs] == ["10:30", "11:00"]


def test_unexpected_response_raises():
    with pytest.raises(SourceError):
        SlfImisSource(fetch=lambda url: {"error": "x"}).observations("X")
    with pytest.raises(SourceError):
        SlfImisSource(fetch=lambda url: {}).stations()


def test_simulation_deterministic_and_shaped():
    a = SimulatedStation(seed=3).observations()
    assert a == SimulatedStation(seed=3).observations()
    assert a != SimulatedStation(seed=4).observations()
    assert len(a) == 8 * 48 and a[0].time == datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert max(o.hs_cm for o in a) > 85 and min(o.tss_c for o in a) < -10


def test_through_codec_preserves_values_within_resolution():
    sim = SimulatedStation().observations()
    rt = through_codec(sim)
    for o, r in zip(sim[::37], rt[::37]):
        assert r.hs_cm == pytest.approx(o.hs_cm, abs=0.05)
        assert r.ta_c == pytest.approx(o.ta_c, abs=0.006)
        assert r.vw_ms == pytest.approx(o.vw_ms, abs=0.11)
