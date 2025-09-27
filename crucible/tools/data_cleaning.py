import re
import json
import os


def strip_pii(text: str) -> str:
    # Remove emails
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "<EMAIL>", text)
    # Remove phone numbers
    text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "<PHONE>", text)
    # Remove SSN-like
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "<SSN>", text)
    # Mask simple names (heuristic)
    text = re.sub(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", "<NAME>", text)
    return text


def normalize_text(text: str, max_tokens: int = 256) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    if len(text.split()) > max_tokens:
        return ""
    return text.strip()


def dedupe_by_input(path_in: str, path_out: str):
    seen = set()
    with open(path_in, "r", encoding="utf-8") as fin, open(
        path_out, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            entry = json.loads(line)
            inp = normalize_text(entry.get("input", ""))
            inp_hash = hash(inp)
            if inp_hash not in seen and inp:
                seen.add(inp_hash)
                fout.write(json.dumps(entry) + "\n")


def clean_logs():
    in_path = "logs/interaction_logs.jsonl"
    out_path = "data/processed/cleaned.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            entry = json.loads(line)
            entry["input"] = strip_pii(normalize_text(entry.get("input", "")))
            entry["model_response"] = strip_pii(
                normalize_text(entry.get("model_response", ""))
            )
            fout.write(json.dumps(entry) + "\n")
    dedupe_by_input(out_path, out_path)
