from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="UNIB IoT Lab-3 — HTTP/REST & WebSocket Full-Duplex Gateway",
    version="1.0.0",
    description="Backend praktikum Minggu 5 Teknologi IoT S1 Informatika FT UNIB",
)


class TelemetryPayload(BaseModel):
    node_id: str = Field(..., min_length=3, max_length=40, examples=["esp32-bengkulu-01"])
    seq: int = Field(..., ge=0)
    temp_c: float = Field(..., ge=-40.0, le=85.0)
    hum_pct: float = Field(..., ge=0.0, le=100.0)
    water_level_cm: float = Field(..., ge=0.0, le=500.0)
    pump_state: Literal["ON", "OFF"] = "OFF"
    client_ts_ms: int = Field(default=0, ge=0)


class ActuatorCommand(BaseModel):
    node_id: str = Field(default="esp32-bengkulu-01")
    pump_state: Literal["ON", "OFF"]
    issued_ts_ms: int | None = None


telemetry_history: deque[dict[str, Any]] = deque(maxlen=50)
device_state: dict[str, str] = {"esp32-bengkulu-01": "OFF"}
device_sockets: dict[str, WebSocket] = {}
dashboard_sockets: set[WebSocket] = set()


async def broadcast_dashboards(message: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    for ws in list(dashboard_sockets):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        dashboard_sockets.discard(ws)


async def push_command_to_device(node_id: str, pump_state: str, source: str) -> bool:
    device_state[node_id] = pump_state
    cmd_msg = {
        "type": "ACTUATOR_CMD",
        "node_id": node_id,
        "pump_state": pump_state,
        "source": source,
        "server_ts_ms": int(time.time() * 1000),
    }
    ws = device_sockets.get(node_id)
    delivered_via_ws = False
    if ws is not None:
        try:
            await ws.send_json(cmd_msg)
            delivered_via_ws = True
        except Exception:
            device_sockets.pop(node_id, None)
    await broadcast_dashboards(
        {
            "type": "COMMAND_LOG",
            "node_id": node_id,
            "pump_state": pump_state,
            "delivered_via_ws": delivered_via_ws,
            "server_ts_ms": cmd_msg["server_ts_ms"],
        }
    )
    return delivered_via_ws


@app.post("/api/v1/telemetry", status_code=status.HTTP_201_CREATED)
async def ingest_telemetry_http(payload: TelemetryPayload) -> dict[str, Any]:
    server_ts_ms = int(time.time() * 1000)
    record = payload.model_dump()
    record["transport"] = "HTTP_POST"
    record["server_ts_ms"] = server_ts_ms
    record["alert"] = payload.water_level_cm >= 150.0
    telemetry_history.append(record)

    await broadcast_dashboards({"type": "TELEMETRY", "data": record})

    return {
        "status": "accepted",
        "protocol": "HTTP/1.1",
        "desired_pump_state": device_state.get(payload.node_id, "OFF"),
        "server_ts_ms": server_ts_ms,
    }


@app.get("/api/v1/telemetry/latest")
async def get_latest_telemetry(limit: int = 10) -> dict[str, Any]:
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="Parameter limit harus 1..50")
    items = list(telemetry_history)[-limit:]
    return {
        "count": len(items),
        "desired_pump_state": device_state.get("esp32-bengkulu-01", "OFF"),
        "items": items,
    }


@app.post("/api/v1/actuator")
async def set_actuator_http(cmd: ActuatorCommand) -> dict[str, Any]:
    delivered = await push_command_to_device(cmd.node_id, cmd.pump_state, source="HTTP_REST")
    return {
        "status": "updated",
        "node_id": cmd.node_id,
        "pump_state": cmd.pump_state,
        "ws_pushed": delivered,
    }


@app.websocket("/ws/device/{node_id}")
async def websocket_device(websocket: WebSocket, node_id: str) -> None:
    await websocket.accept()
    device_sockets[node_id] = websocket
    await broadcast_dashboards({"type": "DEVICE_STATUS", "node_id": node_id, "online": True})
    try:
        while True:
            msg = await websocket.receive_json()
            server_ts_ms = int(time.time() * 1000)
            msg_type = msg.get("type", "TELEMETRY")
            if msg_type == "TELEMETRY":
                record = {
                    "node_id": node_id,
                    "seq": msg.get("seq", 0),
                    "temp_c": msg.get("temp_c", 0.0),
                    "hum_pct": msg.get("hum_pct", 0.0),
                    "water_level_cm": msg.get("water_level_cm", 0.0),
                    "pump_state": msg.get("pump_state", device_state.get(node_id, "OFF")),
                    "client_ts_ms": msg.get("client_ts_ms", 0),
                    "transport": "WEBSOCKET",
                    "server_ts_ms": server_ts_ms,
                    "alert": float(msg.get("water_level_cm", 0.0)) >= 150.0,
                }
                telemetry_history.append(record)
                await broadcast_dashboards({"type": "TELEMETRY", "data": record})
            elif msg_type == "ACK":
                await broadcast_dashboards({"type": "ACK", "data": msg, "server_ts_ms": server_ts_ms})
    except WebSocketDisconnect:
        device_sockets.pop(node_id, None)
        await broadcast_dashboards({"type": "DEVICE_STATUS", "node_id": node_id, "online": False})


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket) -> None:
    await websocket.accept()
    dashboard_sockets.add(websocket)
    await websocket.send_json(
        {
            "type": "INIT",
            "history": list(telemetry_history)[-15:],
            "online_devices": list(device_sockets.keys()),
            "desired_pump_state": device_state.get("esp32-bengkulu-01", "OFF"),
        }
    )
    try:
        while True:
            cmd = await websocket.receive_json()
            if cmd.get("type") == "SET_PUMP":
                target = cmd.get("node_id", "esp32-bengkulu-01")
                state = "ON" if cmd.get("state") == "ON" else "OFF"
                await push_command_to_device(target, state, source="WS_DASHBOARD")
    except WebSocketDisconnect:
        dashboard_sockets.discard(websocket)


@app.get("/", response_class=HTMLResponse)
async def dashboard_page() -> str:
    return """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <title>Lab-3 UNIB IoT — HTTP/REST vs WebSocket Monitor</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; background: #f8fafc; color: #0f172a; }
    .header { background: #0b2e59; color: #fff; padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1.2rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.2rem; }
    .card { background: #fff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 1rem; }
    .val { font-size: 1.7rem; font-weight: 800; color: #0b2e59; }
    button { padding: 0.6rem 1.1rem; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; margin-right: 0.5rem; }
    .btn-on { background: #1e6f50; color: #fff; }
    .btn-off { background: #b91c1c; color: #fff; }
    table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; font-size: 0.88rem; }
    th, td { border-bottom: 1px solid #e2e8f0; padding: 0.55rem 0.7rem; text-align: left; }
    th { background: #ebf2fa; color: #0b2e59; }
    .badge { padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .ws { background: #dcfce7; color: #166534; }
    .http { background: #fef3c7; color: #92400e; }
  </style>
</head>
<body>
  <div class="header">
    <h2 style="margin:0;">Lab-3 Teknologi IoT UNIB — HTTP/REST &amp; WebSocket Full-Duplex</h2>
    <p style="margin:0.35rem 0 0; opacity:0.9;">Status Koneksi WebSocket Dashboard: <strong id="wsStatus">Menghubungkan...</strong></p>
  </div>
  <div class="grid">
    <div class="card"><div>Tinggi Muka Air</div><div class="val" id="valWater">-- cm</div></div>
    <div class="card"><div>Suhu Udara</div><div class="val" id="valTemp">-- &deg;C</div></div>
    <div class="card"><div>Kelembapan</div><div class="val" id="valHum">-- %</div></div>
    <div class="card"><div>Kendali Pompa (Push WS)</div>
      <div style="margin-top:0.5rem;">
        <button class="btn-on" onclick="setPump('ON')">Nyalakan Pompa</button>
        <button class="btn-off" onclick="setPump('OFF')">Matikan</button>
      </div>
    </div>
  </div>
  <h3>Log Telemetri Real-Time (HTTP POST vs WebSocket)</h3>
  <table>
    <thead><tr><th>Seq</th><th>Protokol</th><th>Tinggi Air</th><th>Suhu</th><th>Status Pompa</th><th>Waktu Server</th></tr></thead>
    <tbody id="logBody"></tbody>
  </table>
  <script>
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/dashboard`);
    const logBody = document.getElementById("logBody");
    function addRow(d) {
      document.getElementById("valWater").textContent = `${Number(d.water_level_cm).toFixed(1)} cm`;
      document.getElementById("valTemp").textContent = `${Number(d.temp_c).toFixed(1)} °C`;
      document.getElementById("valHum").textContent = `${Number(d.hum_pct).toFixed(1)} %`;
      const tr = document.createElement("tr");
      const badgeCls = d.transport === "WEBSOCKET" ? "ws" : "http";
      tr.innerHTML = `<td>#${d.seq}</td><td><span class="badge ${badgeCls}">${d.transport}</span></td><td>${d.water_level_cm} cm</td><td>${d.temp_c} °C</td><td>${d.pump_state}</td><td>${new Date(d.server_ts_ms).toLocaleTimeString()}</td>`;
      logBody.prepend(tr);
      while (logBody.children.length > 15) logBody.removeChild(logBody.lastChild);
    }
    ws.onopen = () => { document.getElementById("wsStatus").textContent = "TERHUBUNG (Full-Duplex)"; };
    ws.onclose = () => { document.getElementById("wsStatus").textContent = "TERPUTUS"; };
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "INIT") msg.history.forEach(addRow);
      if (msg.type === "TELEMETRY") addRow(msg.data);
    };
    function setPump(state) {
      ws.send(JSON.stringify({ type: "SET_PUMP", node_id: "esp32-bengkulu-01", state }));
    }
  </script>
</body>
</html>"""
