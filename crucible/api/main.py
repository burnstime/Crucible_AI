
# Healthcheck endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import os
import uuid
from crucible.api.model_server import generate_response
from crucible.api.canary import select_model_tag
from crucible.tools.storage import log_interaction
from prometheus_client import make_asgi_app

app = FastAPI()
app.add_middleware(
    BaseHTTPMiddleware,
    dispatch=lambda req, call_next: latency_middleware(req, call_next),
)


@app.post("/generate")
async def generate(request: Request):
    payload = await request.json()
    input_text = payload["input"]
    session_id = payload.get("session_id") or str(uuid.uuid4())
    metadata = payload.get("metadata", {})
    model_tag = select_model_tag(session_id)
    try:
        t0 = time.time()
        output = generate_response(input_text, model_tag=model_tag)
        latency = time.time() - t0
        log_interaction(
            {
                "timestamp": time.time(),
                "session_id": session_id,
                "input": input_text,
                "model_response": output,
                "model_tag": model_tag,
                "latency": latency,
                "metadata": metadata,
                "served_by": os.getenv("HOSTNAME", "local"),
            }
        )
        return {"output": output, "session_id": session_id, "model": model_tag}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/metrics")
def metrics():
    return make_asgi_app()


async def latency_middleware(request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - t0
        response.headers["X-Request-Latency"] = str(latency)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
