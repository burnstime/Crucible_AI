import os
import json
from typing import Dict


def log_interaction(entry: Dict):
    os.makedirs("logs", exist_ok=True)
    with open("logs/interaction_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def promote_candidate(candidate_tag: str, promote: bool):
    registry_path = "models/registry.json"
    os.makedirs("models", exist_ok=True)
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            reg = json.load(f)
    else:
        reg = {}
    if promote:
        reg["stable"] = candidate_tag
    else:
        reg["candidate"] = candidate_tag
    with open(registry_path, "w") as f:
        json.dump(reg, f, indent=2)
