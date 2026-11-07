"""Command line interface: ``python -m snowpack <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Sequence

from . import DISCLAIMER, codec
from .alerts import AlertTracker, replay, station_report
from .indicators import evaluate_all
from .sources import (Observation, SimulatedStation, SlfImisSource, SourceError,
                      through_codec)


def _fmt(v: float | None, spec: str = ".1f") -> str:
    return "-" if v is None else format(v, spec)


def _print_analysis(station: str, obs: Sequence[Observation], as_json: bool, mqtt_host: str | None,
                    mqtt_port: int, title: str) -> None:
    now = obs[-1].time
    indicators = evaluate_all(obs, now)
    alerts = replay(station, obs, tracker=AlertTracker())
    report = station_report(station, now, indicators, alerts)
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        last = obs[-1]
        print(f"{title}: {station}, {len(obs)} samples, "
              f"{obs[0].time:%Y-%m-%d %H:%M} -> {now:%Y-%m-%d %H:%M} UTC")
        print(f"latest: HS {_fmt(last.hs_cm)} cm, TA {_fmt(last.ta_c)} C, TSS {_fmt(last.tss_c)} C, "
              f"wind {_fmt(last.vw_ms)} m/s\n")
        print(f"{'indicator':<16}{'value':>9} {'unit':<7}{'level':<9}reason")
        for i in indicators:
            print(f"{i.name:<16}{_fmt(i.value):>9} {i.unit:<7}{i.level:<9}{i.reason}")
        print(f"\noverall: {report['overall_level']}")
        if alerts:
            print("\nalerts (de-duplicated, oldest first):")
            for a in alerts:
                print(f"  {a['time']}  {a['level']:<8}{a['indicator']:<15}{a['reason']}")
        else:
            print("\nno alerts in window")
        print(f"\n{DISCLAIMER}")
    if mqtt_host:
        from .mqtt_out import publish_report
        topic = publish_report(mqtt_host, report, mqtt_port)
        print(f"published to {mqtt_host}:{mqtt_port} {topic}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="snowpack", description="Snow station indicators. " + DISCLAIMER)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stations", help="list live SLF IMIS stations")
    s.add_argument("--canton", help="canton code, e.g. GR")
    s.add_argument("--type", help="station type, e.g. SNOW_FLAT, WIND")

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--json", action="store_true", help="print the alert report as JSON")
        sp.add_argument("--mqtt", metavar="HOST", help="publish report to snow/<station>/alerts")
        sp.add_argument("--mqtt-port", type=int, default=1883)

    lv = sub.add_parser("live", help="fetch a station from SLF and compute indicators")
    lv.add_argument("station", help="IMIS code, e.g. DAV2")
    lv.add_argument("--hours", type=float, default=72.0)
    common(lv)

    sm = sub.add_parser("simulate", help="run the simulated 8-day scenario")
    sm.add_argument("--seed", type=int, default=7)
    sm.add_argument("--via-codec", action="store_true", help="quantise through the LoRaWAN codec first")
    common(sm)

    d = sub.add_parser("decode", help="decode a 16-byte uplink given as hex")
    d.add_argument("hex")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "decode":
            print(json.dumps(codec.decode_hex(args.hex), indent=2))
        elif args.cmd == "stations":
            for st in SlfImisSource().stations(args.canton, args.type):
                print(f"{st.code:<7}{st.type or '-':<12}{st.canton_code or '-':<4}"
                      f"{_fmt(st.elevation, '.0f'):>6} m  {st.label}")
        elif args.cmd == "live":
            obs = SlfImisSource().observations(args.station, args.hours)
            if not obs:
                print(f"no observations returned for {args.station}", file=sys.stderr)
                return 1
            _print_analysis(args.station.upper(), obs, args.json, args.mqtt, args.mqtt_port, "live")
        elif args.cmd == "simulate":
            obs = SimulatedStation(seed=args.seed).observations()
            if args.via_codec:
                obs = through_codec(obs)
            _print_analysis(obs[0].station, obs, args.json, args.mqtt, args.mqtt_port, "simulated")
    except (SourceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
