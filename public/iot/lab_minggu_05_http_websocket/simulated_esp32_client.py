"""
Simulator Klien ESP32 (Mode Tanpa Perangkat Keras) untuk Lab-3 Minggu 5.
Mengirim telemetri bergantian menggunakan HTTP POST dan WebSocket (RFC 6455),
sekaligus mengukur waktu respons (RTT) untuk dibandingkan dengan inspeksi Wireshark.
"""
from __future__ import annotations

import asyncio
import json
import random
import time

import httpx
import websockets

NODE_ID = "esp32-bengkulu-01"
HTTP_URL = "http://127.0.0.1:8000/api/v1/telemetry"
WS_URL = f"ws://127.0.0.1:8000/ws/device/{NODE_ID}"

pump_state = "OFF"


async def run_simulation() -> None:
    global pump_state
    seq = 0
    async with httpx.AsyncClient(timeout=5.0) as http_client:
        async with websockets.connect(WS_URL) as ws:
            print(f"[WS] Terhubung ke {WS_URL}")

            async def listen_commands() -> None:
                global pump_state
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "ACTUATOR_CMD":
                        recv_ms = int(time.time() * 1000)
                        latency = recv_ms - msg.get("server_ts_ms", recv_ms)
                        pump_state = msg.get("pump_state", "OFF")
                        print(f"[WS-PUSH] Perintah diterima instan ({latency} ms): POMPA = {pump_state}")
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "ACK",
                                    "node_id": NODE_ID,
                                    "pump_state": pump_state,
                                    "rx_latency_ms": latency,
                                }
                            )
                        )

            listener = asyncio.create_task(listen_commands())
            try:
                while True:
                    seq += 1
                    payload = {
                        "node_id": NODE_ID,
                        "seq": seq,
                        "temp_c": round(28.5 + random.uniform(-1.0, 2.5), 2),
                        "hum_pct": round(78.0 + random.uniform(-4.0, 6.0), 1),
                        "water_level_cm": round(115.0 + random.uniform(0.0, 48.0), 1),
                        "pump_state": pump_state,
                        "client_ts_ms": int(time.time() * 1000),
                    }

                    if seq % 2 == 1:
                        t0 = time.perf_counter()
                        resp = await http_client.post(HTTP_URL, json=payload)
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        data = resp.json()
                        pump_state = data.get("desired_pump_state", pump_state)
                        print(
                            f"[HTTP POST #{seq}] Status={resp.status_code} "
                            f"RTT={dt_ms:.2f} ms | Size={len(json.dumps(payload))} B"
                        )
                    else:
                        t0 = time.perf_counter()
                        ws_payload = {"type": "TELEMETRY", **payload}
                        await ws.send(json.dumps(ws_payload))
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        print(
                            f"[WS SEND   #{seq}] Terkirim non-blocking "
                            f"SendTime={dt_ms:.2f} ms | Size={len(json.dumps(ws_payload))} B"
                        )

                    await asyncio.sleep(2.5)
            finally:
                listener.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan.")
