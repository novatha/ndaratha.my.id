"""
Simulator Node ESP32 MQTT — Lab-4 Minggu 6 Teknologi IoT UNIB.
Mendukung eksperimen:
  1. Hierarki topik terstruktur (Kasus B: Kebun Kopi Rejang Lebong)
  2. Perbandingan pengiriman QoS 0, QoS 1, dan QoS 2
  3. Retained Message status online
  4. Simulasi pemutusan koneksi mendadak (--crash-after) untuk menguji
     Last Will and Testament (LWT) pada broker Mosquitto.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import paho.mqtt.client as mqtt


def main() -> None:
    parser = argparse.ArgumentParser(description="UNIB IoT Lab-4 MQTT Device Simulator")
    parser.add_argument("--host", default="127.0.0.1", help="Alamat broker Mosquitto")
    parser.add_argument("--port", type=int, default=1883, help="Port MQTT")
    parser.add_argument("--node-id", default="node-01", help="ID node sensor")
    parser.add_argument("--qos", type=int, choices=[0, 1, 2], default=1, help="QoS publikasi telemetri")
    parser.add_argument("--keepalive", type=int, default=10, help="Keep-Alive detik (LWT terpicu ~1.5x keepalive)")
    parser.add_argument(
        "--crash-after",
        type=int,
        default=0,
        help="Jika > 0, simulasikan perangkat mati mendadak (tanpa DISCONNECT) setelah N detik untuk memicu LWT",
    )
    args = parser.parse_args()

    base_topic = f"unib/rejang-lebong/kebun-kopi/blok-a/{args.node_id}"
    topic_status = f"{base_topic}/status"
    topic_telemetry = f"{base_topic}/telemetry"
    topic_cmd = f"{base_topic}/cmd/pump"
    topic_ack = f"{base_topic}/ack/pump"

    pump_state = "OFF"

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"sim-{args.node_id}",
        clean_session=True,
        protocol=mqtt.MQTTv311,
    )

    # 1. Konfigurasi Last Will and Testament (LWT) SEBELUM connect()
    lwt_payload = json.dumps(
        {"node_id": args.node_id, "status": "offline", "reason": "lwt_unexpected_disconnect"}
    )
    client.will_set(topic_status, payload=lwt_payload, qos=1, retain=True)

    def on_connect(c, userdata, flags, reason_code, properties):
        print(f"[DEV] Terhubung ke {args.host}:{args.port} (Keep-Alive={args.keepalive}s, QoS={args.qos})")
        # Publikasi status ONLINE sebagai Retained Message
        online_payload = json.dumps({"node_id": args.node_id, "status": "online", "ts": int(time.time())})
        c.publish(topic_status, payload=online_payload, qos=1, retain=True)
        c.subscribe(topic_cmd, qos=1)
        print(f"[DEV] Status ONLINE (Retained) dipublikasi ke '{topic_status}'")
        print(f"[DEV] Berlangganan perintah aktuator di '{topic_cmd}'")

    def on_message(c, userdata, msg):
        nonlocal pump_state
        raw = msg.payload.decode("utf-8", errors="replace").strip().upper()
        if raw in ("ON", "OFF"):
            pump_state = raw
        else:
            try:
                data = json.loads(raw)
                pump_state = str(data.get("state", pump_state)).upper()
            except Exception:
                pass
        print(f"[DEV-CMD] Perintah pompa diterima: {pump_state}")
        ack = json.dumps({"node_id": args.node_id, "pump_state": pump_state, "ts": int(time.time())})
        c.publish(topic_ack, payload=ack, qos=1, retain=False)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=args.keepalive)
    client.loop_start()

    seq = 0
    t_start = time.time()
    try:
        while True:
            if args.crash_after > 0 and (time.time() - t_start) >= args.crash_after:
                print(
                    f"\n[CRASH] Simulasi mati listrik mendadak pada t={args.crash_after}s "
                    f"(KELUAR TANPA paket DISCONNECT)! Amati LWT di subscriber dalam ~{int(args.keepalive * 1.5)}s..."
                )
                os._exit(1)

            seq += 1
            payload = {
                "node_id": args.node_id,
                "seq": seq,
                "temp_c": round(23.5 + random.uniform(-1.2, 2.8), 2),
                "soil_moisture_pct": round(48.0 + random.uniform(-8.0, 12.0), 1),
                "pump_state": pump_state,
                "qos_used": args.qos,
                "ts": int(time.time()),
            }
            info = client.publish(topic_telemetry, payload=json.dumps(payload), qos=args.qos, retain=False)
            info.wait_for_publish(timeout=3.0)
            print(f"[PUB #{seq}] QoS={args.qos} -> {topic_telemetry} | {payload}")
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\n[DEV] Pemutusan normal (mengirim status offline eksplisit & DISCONNECT)...")
        client.publish(
            topic_status,
            payload=json.dumps({"node_id": args.node_id, "status": "offline", "reason": "graceful_shutdown"}),
            qos=1,
            retain=True,
        )
        time.sleep(0.3)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
