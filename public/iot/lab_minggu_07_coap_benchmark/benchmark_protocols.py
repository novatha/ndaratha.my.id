"""
Skrip Benchmarking Eksperimental 3 Protokol IoT: HTTP/1.1 vs MQTT vs CoAP
Lab-5 Minggu 7 Teknologi IoT — Prodi S1 Informatika FT UNIB.

Jalankan setelah ketiga server (HTTP :8000, Mosquitto :1883, CoAP :5683) aktif:
    python3 benchmark_protocols.py --samples 40 --label "loss_0pct_delay_0ms"
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from pathlib import Path

import aiocoap
import httpx
import paho.mqtt.client as mqtt


SAMPLE_PAYLOAD = json.dumps(
    {"node": "unib-01", "water_cm": 142.5, "temp_c": 29.4, "hum": 78.0},
    separators=(",", ":"),
).encode("utf-8")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil((p / 100.0) * len(s)) - 1))
    return s[idx]


async def bench_http(host: str, port: int, n: int, keep_alive: bool) -> dict[str, object]:
    url = f"http://{host}:{port}/api/v1/telemetry"
    headers = {"Content-Type": "application/json"}
    if not keep_alive:
        headers["Connection"] = "close"

    latencies: list[float] = []
    ok = 0
    async with httpx.AsyncClient(timeout=4.0) as client:
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                r = await client.post(url, content=SAMPLE_PAYLOAD, headers=headers)
                if r.status_code in (200, 201):
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    ok += 1
            except Exception:
                pass
    mode = "HTTP_POST_KeepAlive" if keep_alive else "HTTP_POST_Close"
    return {
        "protocol": mode,
        "sent": n,
        "success": ok,
        "delivery_pct": round((ok / n) * 100.0, 1),
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else -1.0,
        "p50_ms": round(percentile(latencies, 50), 2) if latencies else -1.0,
        "p95_ms": round(percentile(latencies, 95), 2) if latencies else -1.0,
        "est_app_header_bytes": 185,
    }


def bench_mqtt(host: str, port: int, n: int, qos: int) -> dict[str, object]:
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"bench-mqtt-q{qos}-{int(time.time())}",
        protocol=mqtt.MQTTv311,
    )
    latencies: list[float] = []
    ok = 0
    topic = "unib/bench/node-01/telemetry"
    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()
        time.sleep(0.15)
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                info = client.publish(topic, payload=SAMPLE_PAYLOAD, qos=qos)
                info.wait_for_publish(timeout=4.0)
                if info.is_published():
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    ok += 1
            except Exception:
                pass
    finally:
        client.loop_stop()
        client.disconnect()
    return {
        "protocol": f"MQTT_QoS{qos}",
        "sent": n,
        "success": ok,
        "delivery_pct": round((ok / n) * 100.0, 1),
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else -1.0,
        "p50_ms": round(percentile(latencies, 50), 2) if latencies else -1.0,
        "p95_ms": round(percentile(latencies, 95), 2) if latencies else -1.0,
        "est_app_header_bytes": 2 + 2 + len(topic) + (2 if qos > 0 else 0),
    }


async def bench_coap(host: str, port: int, n: int, confirmable: bool) -> dict[str, object]:
    ctx = await aiocoap.Context.create_client_context()
    uri = f"coap://{host}:{port}/sensor/water"
    mtype = aiocoap.CON if confirmable else aiocoap.NON
    latencies: list[float] = []
    ok = 0
    try:
        for _ in range(n):
            req = aiocoap.Message(
                code=aiocoap.POST,
                uri=uri,
                mtype=mtype,
                payload=SAMPLE_PAYLOAD,
                content_format=50,
            )
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(ctx.request(req).response, timeout=4.5)
                if resp.code.is_successful():
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    ok += 1
            except Exception:
                pass
    finally:
        await ctx.shutdown()

    return {
        "protocol": "CoAP_CON" if confirmable else "CoAP_NON",
        "sent": n,
        "success": ok,
        "delivery_pct": round((ok / n) * 100.0, 1),
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else -1.0,
        "p50_ms": round(percentile(latencies, 50), 2) if latencies else -1.0,
        "p95_ms": round(percentile(latencies, 95), 2) if latencies else -1.0,
        "est_app_header_bytes": 4 + 14,  # 4B fixed header + URI-Path & Content-Format options
    }


async def run_all(args: argparse.Namespace) -> None:
    print(f"=== UNIB IoT Lab-5 Benchmark | Skenario: {args.label} | N={args.samples} | Payload={len(SAMPLE_PAYLOAD)} B ===")
    results = [
        await bench_http(args.http_host, args.http_port, args.samples, keep_alive=False),
        await bench_http(args.http_host, args.http_port, args.samples, keep_alive=True),
        bench_mqtt(args.mqtt_host, args.mqtt_port, args.samples, qos=0),
        bench_mqtt(args.mqtt_host, args.mqtt_port, args.samples, qos=1),
        await bench_coap(args.coap_host, args.coap_port, args.samples, confirmable=True),
        await bench_coap(args.coap_host, args.coap_port, args.samples, confirmable=False),
    ]

    print(f"\n{'Protokol':<22} {'Sukses':>8} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'AppHdr(B)':>10}")
    print("-" * 74)
    for r in results:
        print(
            f"{r['protocol']:<22} {r['delivery_pct']:>7.1f}% "
            f"{r['mean_ms']:>10.2f} {r['p50_ms']:>10.2f} {r['p95_ms']:>10.2f} {r['est_app_header_bytes']:>10}"
        )

    out_csv = Path(args.output)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["scenario", "protocol", "sent", "success", "delivery_pct", "mean_ms", "p50_ms", "p95_ms", "est_app_header_bytes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({"scenario": args.label, **r})
    print(f"\n[OK] Hasil disimpan ke {out_csv.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmarking HTTP vs MQTT vs CoAP (Lab-5)")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--label", default="baseline_0pct_loss")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--coap-host", default="127.0.0.1")
    parser.add_argument("--coap-port", type=int, default=5683)
    parser.add_argument("--output", default="benchmark_results.csv")
    args = parser.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
