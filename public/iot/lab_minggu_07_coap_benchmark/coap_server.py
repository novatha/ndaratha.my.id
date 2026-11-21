"""
Server CoAP (RFC 7252) + Observe (RFC 7641) + Resource Discovery (RFC 6690)
Lab-5 Minggu 7 Teknologi IoT — Prodi S1 Informatika FT UNIB.

Jalankan:
    python3 coap_server.py
Mendengarkan UDP port 5683 pada 0.0.0.0.
"""
from __future__ import annotations

import asyncio
import json
import random
import time

import aiocoap
import aiocoap.resource as resource


class WaterLevelResource(resource.ObservableResource):
    """Resource telemetri tinggi muka air DAS Air Bengkulu dengan dukungan RFC 7641 Observe."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = 0
        self.water_cm = 120.0
        self.temp_c = 28.5
        self._task: asyncio.Task[None] | None = None

    def get_link_description(self) -> dict[str, str]:
        return {
            "title": "Water Level Sensor DAS Air Bengkulu",
            "rt": "unib.sensor.water",
            "if": "sensor",
            "obs": "",
            "ct": "50",  # application/json
        }

    async def periodic_notify(self) -> None:
        while True:
            await asyncio.sleep(3.0)
            self.seq += 1
            self.water_cm = round(115.0 + random.uniform(0.0, 55.0), 1)
            self.temp_c = round(28.0 + random.uniform(-1.0, 2.5), 2)
            self.updated_state()

    def _build_payload(self) -> bytes:
        data = {
            "node_id": "coap-bengkulu-01",
            "seq": self.seq,
            "water_cm": self.water_cm,
            "temp_c": self.temp_c,
            "alert": self.water_cm >= 150.0,
            "ts": int(time.time()),
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        return aiocoap.Message(
            code=aiocoap.CONTENT,
            payload=self._build_payload(),
            content_format=50,  # application/json
        )

    async def render_post(self, request: aiocoap.Message) -> aiocoap.Message:
        # Endpoint sink untuk benchmarking pengiriman telemetri CoAP POST
        return aiocoap.Message(
            code=aiocoap.CREATED,
            payload=b'{"ok":true}',
            content_format=50,
        )


class SirenActuatorResource(resource.Resource):
    """Resource aktuator sirene peringatan banjir (GET/PUT)."""

    def __init__(self) -> None:
        super().__init__()
        self.state = "OFF"

    def get_link_description(self) -> dict[str, str]:
        return {
            "title": "Flood Warning Siren Actuator",
            "rt": "unib.actuator.siren",
            "if": "actuator",
            "ct": "50",
        }

    async def render_get(self, request: aiocoap.Message) -> aiocoap.Message:
        body = json.dumps({"siren": self.state}).encode("utf-8")
        return aiocoap.Message(code=aiocoap.CONTENT, payload=body, content_format=50)

    async def render_put(self, request: aiocoap.Message) -> aiocoap.Message:
        raw = request.payload.decode("utf-8", errors="replace").strip().upper()
        if "ON" in raw:
            self.state = "ON"
        elif "OFF" in raw:
            self.state = "OFF"
        print(f"[CoAP-SRV] PUT /actuator/siren -> status baru: {self.state}")
        body = json.dumps({"siren": self.state, "updated": True}).encode("utf-8")
        return aiocoap.Message(code=aiocoap.CHANGED, payload=body, content_format=50)


async def main() -> None:
    root = resource.Site()
    root.add_resource([".well-known", "core"], resource.WKCResource(root.get_resources_as_linkheader))

    water_res = WaterLevelResource()
    siren_res = SirenActuatorResource()
    root.add_resource(["sensor", "water"], water_res)
    root.add_resource(["actuator", "siren"], siren_res)

    await aiocoap.Context.create_server_context(root, bind=("0.0.0.0", 5683))
    print("[CoAP-SRV] Server CoAP (RFC 7252) aktif pada udp://0.0.0.0:5683")
    print("[CoAP-SRV] Resource tersedia: /.well-known/core | /sensor/water (obs) | /actuator/siren")

    notifier = asyncio.create_task(water_res.periodic_notify())
    try:
        await asyncio.get_running_loop().create_future()
    finally:
        notifier.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CoAP-SRV] Server dihentikan.")
