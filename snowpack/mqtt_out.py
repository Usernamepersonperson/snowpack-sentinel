"""Optional MQTT publishing (needs ``pip install snowpack-sentinel[mqtt]``)."""

from __future__ import annotations

import json
from typing import Any


def publish_report(host: str, report: dict[str, Any], port: int = 1883) -> str:
    """Publish a station report to ``snow/<station>/alerts``; returns the topic."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("paho-mqtt not installed: pip install 'snowpack-sentinel[mqtt]'") from exc
    topic = f"snow/{report['station']}/alerts"
    if hasattr(mqtt, "CallbackAPIVersion"):  # paho-mqtt 2.x
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    else:  # pragma: no cover
        client = mqtt.Client()
    client.connect(host, port, keepalive=30)
    client.loop_start()
    info = client.publish(topic, json.dumps(report), qos=1)
    info.wait_for_publish(timeout=10)
    client.loop_stop()
    client.disconnect()
    return topic
