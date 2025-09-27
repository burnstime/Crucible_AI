from fastapi.testclient import TestClient
from crucible.api.main import app
import os


def test_generate_and_logging(tmp_path):
    client = TestClient(app)
    payload = {
        "input": "Hello, world!",
        "session_id": "test123",
        "metadata": {"foo": "bar"},
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert (
        "output" in data and "session_id" in data and "model" in data
    )
    # Check log file
    log_path = "logs/interaction_logs.jsonl"
    assert os.path.exists(log_path)
    with open(log_path) as f:
        lines = f.readlines()
        assert any("test123" in line for line in lines)
