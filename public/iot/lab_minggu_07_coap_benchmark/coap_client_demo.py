"""
Klien CoAP (RFC 7252 & RFC 7641 Observe) — Lab-5 Minggu 7 Teknologi IoT UNIB.

Contoh penggunaan:
    python3 coap_client_demo.py --mode discover
    python3 coap_client_demo.py --mode get --mtype CON
    python3 coap_client_demo.py --mode get --mtype NON
    python3 coap_client_demo.py --mode put --payload ON
    python3 coap_client_demo.py --mode observe
"""
from __future__ import annotations

import argparse
import asyncio
import time

import aiocoap


async def run_client(args: argparse.Namespace) -> None:
    ctx = await aiocoap.Context.create_client_context()
    base = f"coap://{args.host}:{args.port}"
    mtype = aiocoap.CON if args.mtype == "CON" else aiocoap.NON

    try:
        if args.mode == "discover":
            uri = f"{base}/.well-known/core"
            req = aiocoap.Message(code=aiocoap.GET, uri=uri, mtype=aiocoap.CON)
            t0 = time.perf_counter()
            resp = await ctx.request(req).response
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[DISCOVERY] Code={resp.code} RTT={dt_ms:.2f} ms")
            print(resp.payload.decode("utf-8", errors="replace"))

        elif args.mode == "get":
            uri = f"{base}/sensor/water"
            req = aiocoap.Message(code=aiocoap.GET, uri=uri, mtype=mtype)
            t0 = time.perf_counter()
            resp = await ctx.request(req).response
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[GET {args.mtype}] Code={resp.code} RTT={dt_ms:.2f} ms | Payload={resp.payload.decode()}")

        elif args.mode == "put":
            uri = f"{base}/actuator/siren"
            req = aiocoap.Message(
                code=aiocoap.PUT,
                uri=uri,
                mtype=mtype,
                payload=args.payload.encode("utf-8"),
            )
            t0 = time.perf_counter()
            resp = await ctx.request(req).response
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[PUT {args.mtype}] Code={resp.code} RTT={dt_ms:.2f} ms | Resp={resp.payload.decode()}")

        elif args.mode == "observe":
            uri = f"{base}/sensor/water"
            req = aiocoap.Message(code=aiocoap.GET, uri=uri, observe=0, mtype=aiocoap.CON)
            pr = ctx.request(req)
            first = await pr.response
            print(f"[OBS INIT] Seq={first.opt.observe} Code={first.code} | {first.payload.decode()}")
            async for resp in pr.observation:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] [OBS PUSH] Seq={resp.opt.observe} | {resp.payload.decode()}")
    finally:
        await ctx.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="UNIB IoT Lab-5 CoAP Client Demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5683)
    parser.add_argument("--mode", choices=["discover", "get", "put", "observe"], default="discover")
    parser.add_argument("--mtype", choices=["CON", "NON"], default="CON")
    parser.add_argument("--payload", default="ON")
    args = parser.parse_args()
    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        print("\n[CoAP-CLI] Dihentikan.")


if __name__ == "__main__":
    main()
