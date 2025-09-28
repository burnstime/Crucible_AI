
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import os
import uuid
import re
from crucible.api.model_server import generate_response
from crucible.api.canary import select_model_tag
from crucible.tools.storage import log_interaction
from prometheus_client import make_asgi_app
from pydantic import BaseModel, validator
from typing import Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input validation models
class GenerateRequest(BaseModel):
    input: str
    session_id: Optional[str] = None
    metadata: dict = {}
    
    @validator('input')
    def validate_input(cls, v):
        if not v or not v.strip():
            raise ValueError('Input cannot be empty')
        if len(v) > 10000:  # Prevent DoS attacks
            raise ValueError('Input too long (max 10000 characters)')
        # Basic XSS protection
        if re.search(r'<script|javascript:|data:', v, re.IGNORECASE):
            raise ValueError('Potentially malicious input detected')
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v and len(v) > 100:
            raise ValueError('Session ID too long')
        return v

# Healthcheck endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

app.add_middleware(
    BaseHTTPMiddleware,
    dispatch=lambda req, call_next: latency_middleware(req, call_next),
)


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        input_text = request.input
        session_id = request.session_id or str(uuid.uuid4())
        metadata = request.metadata or {}
        model_tag = select_model_tag(session_id)
        
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the error but don't expose internal details
        print(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
