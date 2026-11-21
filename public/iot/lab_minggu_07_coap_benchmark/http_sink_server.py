"""
Server HTTP/1.1 FastAPI minimal sebagai target pembanding pada Benchmarking Lab-5.
Jalankan: uvicorn http_sink_server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Request

app = FastAPI(title="Lab-5 HTTP Benchmark Sink")


@app.post("/api/v1/telemetry", status_code=201)
async def telemetry_sink(request: Request) -> dict[str, bool]:
    await request.body()
    return {"ok": True}
