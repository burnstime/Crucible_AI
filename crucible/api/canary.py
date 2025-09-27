
from typing import Optional
import os
import hashlib
from prometheus_client import Counter, Histogram


CANARY_PERCENTAGE = int(os.getenv("CANARY_PERCENTAGE", "10"))

REQUESTS = Counter(
    "crucible_requests_total",
    "Total requests",
    ["served_by", "status"],
)
LATENCY = Histogram(
    "crucible_latency_seconds",
    "Request latency",
    ["served_by", "latency_bucket"],
)

from typing import Optional


def select_model_tag(
    session_id: Optional[str] = None,
) -> str:
    if session_id:
        h = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
        if (h % 100) < CANARY_PERCENTAGE:
            return "vnext"
        return "stable"
    import random

    return "vnext" if random.randint(0, 99) < CANARY_PERCENTAGE else "stable"
