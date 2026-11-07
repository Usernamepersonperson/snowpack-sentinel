from datetime import datetime, timedelta, timezone

import pytest

from snowpack import indicators as ind
from snowpack.sources import Observation

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def series(n, step_min=60, **cols):
    """Build n observations; each column is a list (len n) or a constant."""
    out = []
    for i in range(n):
        kw = {k: (v[i] if isinstance(v, list) else v) for k, v in cols.items()}
        out.append(Observation(time=T0 + timedelta(minutes=step_min * i), station="T", **kw))
    return out


# ---------------------------------------------------------------- new snow
def test_new_snow_ignores_noise():
    obs = series(8, 30, hs_cm=[50, 50, 50, 50.4, 50, 50.6, 50, 50])
    r = ind.new_snow(obs, hours=24)
    assert r.value == 0.0 and r.level == "none"


def test_new_snow_sums_steps():
    obs = series(7, 30, hs_cm=[50, 50, 50, 55, 60, 60, 60])
    assert ind.new_snow(obs).value == 10.0


def test_new_snow_slow_snowfall_accumulates():
    hs = [50 + 0.5 * i for i in range(12)]  # 0.5 cm per sample: below min_step per step
    assert ind.new_snow(series(12, 30, hs_cm=hs)).value == 5.0


def test_new_snow_ignores_settlement_then_counts_rise():
    hs = [60, 59, 58, 57, 57, 57, 62, 67, 67, 67]
    assert ind.new_snow(series(10, 30, hs_cm=hs)).value == 10.0


def test_new_snow_single_spike_removed():
    assert ind.new_snow(series(7, 30, hs_cm=[50, 50, 50, 80, 50, 50, 50])).value == 0.0


@pytest.mark.parametrize("total,level", [(15, "none"), (25, "watch"), (35, "warning")])
def test_new_snow_levels_24h(total, level):
    hs = [50] * 3 + [50 + total * k / 5 for k in range(1, 6)] + [50 + total] * 3
    r = ind.new_snow(series(len(hs), 30, hs_cm=hs), hours=24)
    assert r.value == pytest.approx(total, abs=0.1) and r.level == level


def test_new_snow_insufficient_data():
    r = ind.new_snow(series(2, hs_cm=[10, 20]))
    assert r.value is None and r.level == "none"
    assert ind.new_snow(series(5, hs_cm=None)).value is None


def test_new_snow_window_excludes_old_rise():
    hs = [0] * 3 + [30] * 3 + [30] * 60
    obs = series(len(hs), 60, hs_cm=hs)  # rise happened ~60 h before the end
    assert ind.new_snow(obs, hours=24).value == 0.0
    assert ind.new_snow(obs, hours=72).value == 30.0


# -------------------------------------------------------------- wind drift
def test_wind_drift_known_answer():
    hs = [50, 50, 50, 55, 60, 65, 70, 70, 70, 70]
    vw = [2, 2, 2, 6, 6, 6, 6, 6, 6, 6]  # 7 of 10 in the 4-12 band
    r = ind.wind_drift(series(10, 30, hs_cm=hs, vw_ms=vw))
    assert r.value == 14.0 and r.level == "watch"


def test_wind_drift_outside_band_is_zero():
    hs = [50, 50, 50, 55, 60, 65, 70, 70, 70, 70]
    assert ind.wind_drift(series(10, 30, hs_cm=hs, vw_ms=15.0)).value == 0.0
    assert ind.wind_drift(series(10, 30, hs_cm=hs, vw_ms=1.0)).level == "none"


def test_wind_drift_warning_and_missing_wind():
    hs = [50, 50, 50, 60, 70, 80, 90, 90, 90, 90]
    assert ind.wind_drift(series(10, 30, hs_cm=hs, vw_ms=8.0)).level == "warning"
    r = ind.wind_drift(series(10, 30, hs_cm=hs))
    assert r.value is None


# ------------------------------------------------------------- temp gradient
def test_gradient_single_observation():
    o = series(1, hs_cm=100, tss_c=-12.0, t0_c=0.0)[0]
    assert ind.gradient_c_per_m(o) == pytest.approx(12.0)


def test_gradient_uses_lowest_buried_sensor():
    o = series(1, hs_cm=100, tss_c=-12.0, t0_c=None, t25_c=-2.0)[0]
    assert ind.gradient_c_per_m(o) == pytest.approx(10 / 0.75)
    # sensor at 100 cm is not buried by >= 5 cm when HS = 100
    o = series(1, hs_cm=100, tss_c=-12.0, t100_c=-1.0)[0]
    assert ind.gradient_c_per_m(o) is None


def test_gradient_too_shallow_or_missing():
    assert ind.gradient_c_per_m(series(1, hs_cm=15, tss_c=-8.0, t0_c=0.0)[0]) is None
    assert ind.gradient_c_per_m(series(1, hs_cm=100, tss_c=None, t0_c=0.0)[0]) is None


def test_temp_gradient_levels():
    mk = lambda tss, n=73: series(n, 60, hs_cm=100, tss_c=tss, t0_c=0.0)
    assert ind.temp_gradient(mk(-10.0)).level == "none"  # exactly 10: not > 10
    assert ind.temp_gradient(mk(-12.0, n=13)).level == "watch"  # only 12 h of data
    r = ind.temp_gradient(mk(-12.0))
    assert r.level == "warning" and r.value == 12.0
    assert ind.temp_gradient(mk(-3.0)).level == "none"


def test_temp_gradient_not_persistent_stays_watch():
    tss = [-3.0] * 48 + [-14.0] * 25  # cold only in the last 25 h
    r = ind.temp_gradient(series(73, 60, hs_cm=100, tss_c=tss, t0_c=0.0))
    assert r.level == "watch"


def test_temp_gradient_no_snow():
    assert ind.temp_gradient(series(30, 60, hs_cm=5, tss_c=-5.0, t0_c=0.0)).value is None


# ----------------------------------------------------------- rapid warming
def test_rapid_warming_levels():
    cold = [-10.0] * 24
    assert ind.rapid_warming(series(25, 60, ta_c=cold + [6.0], tss_c=-5.0, hs_cm=50)).level == "warning"
    r = ind.rapid_warming(series(25, 60, ta_c=cold + [2.0], tss_c=-5.0, hs_cm=50))
    assert r.value == 12.0 and r.level == "watch"
    assert ind.rapid_warming(series(25, 60, ta_c=cold + [-3.0], tss_c=-5.0, hs_cm=50)).level == "none"


def test_rapid_warming_wet_snow_rule():
    ta = [-10.0] * 24 + [1.0]
    wet = ind.rapid_warming(series(25, 60, ta_c=ta, tss_c=[-5.0] * 24 + [-0.2], hs_cm=50))
    assert wet.level == "warning" and "wet-snow" in wet.reason
    thin = ind.rapid_warming(series(25, 60, ta_c=ta, tss_c=[-5.0] * 24 + [-0.2], hs_cm=5))
    assert thin.level == "watch"
    cold_surface = ind.rapid_warming(series(25, 60, ta_c=ta, tss_c=-5.0, hs_cm=50))
    assert cold_surface.level == "watch"


def test_rapid_warming_missing_temperature():
    assert ind.rapid_warming(series(25, 60, ta_c=None)).value is None


# --------------------------------------------------------------- settlement
def test_settlement_known_answer():
    hs = [100 - 10 * i / 24 for i in range(25)]
    r = ind.settlement(series(25, 60, hs_cm=hs))
    assert r.value == pytest.approx(10.0, abs=0.1) and r.level == "warning"


def test_settlement_flat_and_new_snow():
    assert ind.settlement(series(25, 60, hs_cm=100.0)).level == "none"
    hs = [100.0] * 20 + [110.0] * 5
    r = ind.settlement(series(25, 60, hs_cm=hs))
    assert r.value is None and "new snow" in r.reason


def test_evaluate_all_order_and_names():
    names = [i.name for i in ind.evaluate_all(series(10, 30, hs_cm=50.0))]
    assert names == ["new_snow_24h", "new_snow_72h", "wind_drift", "temp_gradient",
                     "rapid_warming", "settlement"]
