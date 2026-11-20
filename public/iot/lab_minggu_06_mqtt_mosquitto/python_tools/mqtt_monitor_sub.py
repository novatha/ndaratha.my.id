"""
MQTT Monitor & Persistent Session Tester — Lab-4 Minggu 6 Teknologi IoT UNIB.
Berlangganan menggunakan wildcard multi-level (#) atau single-level (+)
untuk mengamati pesan telemetri, Retained Message, QoS 0/1/2, dan LWT (offline).
"""
from __future__ import annotations

import argparse
import time
import paho.mqtt.client as mqtt


def main() -> None:
    parser = argparse.ArgumentParser(description="UNIB IoT Lab-4 MQTT Subscriber Monitor")
    parser.add_argument("--host", default="127.0.0.1", help="Alamat broker Mosquitto")
    parser.add_argument("--port", type=int, default=1883, help="Port MQTT (default: 1883)")
    parser.add_argument("--topic", default="unib/rejang-lebong/kebun-kopi/#", help="Filter topik/wildcard")
    parser.add_argument("--qos", type=int, choices=[0, 1, 2], default=1, help="QoS langganan (0, 1, 2)")
    parser.add_argument("--client-id", default="unib-monitor-01", help="Client ID tetap untuk uji Persistent Session")
    parser.add_argument("--persistent", action="store_true", help="Aktifkan Persistent Session (clean_session=False)")
    args = parser.parse_args()

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=args.client_id,
        clean_session=not args.persistent,
        protocol=mqtt.MQTTv311,
    )

    def on_connect(c, userdata, flags, reason_code, properties):
        print(f"[SUB] Terhubung ke {args.host}:{args.port} | RC={reason_code} | SessionPresent={flags.session_present}")
        c.subscribe(args.topic, qos=args.qos)
        print(f"[SUB] Berlangganan topik='{args.topic}' dengan QoS={args.qos} (Persistent={args.persistent})")
        print("-" * 78)

    def on_message(c, userdata, msg):
        ts = time.strftime("%H:%M:%S")
        flag_ret = "RETAINED" if msg.retain else "LIVE    "
        payload_str = msg.payload.decode("utf-8", errors="replace")
        print(f"[{ts}] [{flag_ret}] [QoS={msg.qos}] {msg.topic} -> {payload_str}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[SUB] Memutuskan koneksi secara normal (DISCONNECT)...")
        client.disconnect()


if __name__ == "__main__":
    main()
