import json
from datetime import datetime, timedelta, timezone

from snowpack import DISCLAIMER
from snowpack.alerts import AlertTracker, replay, station_report
from snowpack.cli import main
from snowpack.indicators import Indicator, evaluate_all
from snowpack.sources import SimulatedStation, through_codec

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def ind(level, name="new_snow_24h"):
    return Indicator(name, 1.0, "cm", level, "r")


def test_tracker_dedupes_and_escalates():
    tr = AlertTracker(repeat_after_h=24)
    h = lambda n: T0 + timedelta(hours=n)
    assert len(tr.update("S", h(0), [ind("watch")])) == 1
    assert tr.update("S", h(1), [ind("watch")]) == []  # repeat
    assert len(tr.update("S", h(2), [ind("warning")])) == 1  # escalation
    assert tr.update("S", h(3), [ind("watch")]) == []  # de-escalation is silent
    assert tr.update("S", h(4), [ind("none")]) == []
    assert tr.update("S", h(5), [ind("warning")]) == []  # flapping suppressed
    assert len(tr.update("S", h(27), [ind("warning")])) == 1  # reminder after 24 h


def test_tracker_is_per_station_and_indicator():
    tr = AlertTracker()
    assert len(tr.update("A", T0, [ind("watch")])) == 1
    assert len(tr.update("B", T0, [ind("watch")])) == 1
    assert len(tr.update("A", T0, [ind("watch", "wind_drift")])) == 1


def test_station_report_json():
    inds = [ind("none"), ind("warning", "wind_drift")]
    rep = station_report("SIM1", T0, inds, [{"x": 1}])
    assert rep["overall_level"] == "warning" and rep["generated_at"] == "2026-01-01T00:00:00Z"
    assert rep["disclaimer"] == DISCLAIMER and len(rep["indicators"]) == 2
    json.dumps(rep)  # serialisable
    assert station_report("S", T0, [ind("none")])["overall_level"] == "none"


def first_fires(alerts):
    seen = []
    for a in alerts:
        key = (a["indicator"], a["level"])
        if key not in seen:
            seen.append(key)
    return seen


EXPECTED_ORDER = [
    ("wind_drift", "watch"),
    ("new_snow_24h", "watch"),
    ("wind_drift", "warning"),
    ("new_snow_24h", "warning"),
    ("new_snow_72h", "watch"),
    ("temp_gradient", "watch"),
    ("temp_gradient", "warning"),
    ("rapid_warming", "watch"),
    ("rapid_warming", "warning"),
    ("settlement", "watch"),
]


def test_simulated_scenario_alert_order():
    alerts = replay("SIM1", SimulatedStation().observations())
    assert first_fires(alerts) == EXPECTED_ORDER
    times = [a["time"] for a in alerts]
    assert times == sorted(times)
    assert times[0] >= "2026-01-11T00:00:00Z"  # nothing before the storm starts
    assert not any(a["indicator"] == "new_snow_72h" and a["level"] == "warning" for a in alerts)
    # warm-day alerts only on the last day
    assert all(a["time"] >= "2026-01-17" for a in alerts if a["indicator"] in ("rapid_warming", "settlement"))


def test_simulated_scenario_through_codec_gives_same_order():
    obs = through_codec(SimulatedStation().observations())
    assert first_fires(replay("SIM1", obs)) == EXPECTED_ORDER


def test_no_repeated_alert_within_24h_for_same_level():
    last: dict = {}
    for a in replay("SIM1", SimulatedStation().observations()):
        key = (a["indicator"], a["level"])
        if key in last:
            gap = datetime.fromisoformat(a["time"].replace("Z", "+00:00")) - last[key]
            assert gap >= timedelta(hours=24)
        last[key] = datetime.fromisoformat(a["time"].replace("Z", "+00:00"))


def test_final_state_is_warm_day_watch_or_worse():
    obs = SimulatedStation().observations()
    levels = {i.name: i.level for i in evaluate_all(obs)}
    assert levels["rapid_warming"] != "none" or levels["settlement"] != "none"


def test_cli_simulate_json_and_decode(capsys):
    assert main(["simulate", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["station"] == "SIM1" and rep["alerts"] and "NOT an avalanche forecast" in rep["disclaimer"]
    assert main(["decode", "04d2fea2fd8ffe3eff38ffce153187a0"]) == 0
    assert json.loads(capsys.readouterr().out)["hs_mm"] == 1234
    assert main(["decode", "04d2"]) == 1
