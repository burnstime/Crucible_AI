import re
import json
import os
import hashlib


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
            try:
                entry = json.loads(line.strip())
                inp = normalize_text(entry.get("input", ""))
                # Use SHA-256 for secure hashing
                inp_hash = hashlib.sha256(inp.encode('utf-8')).hexdigest()
                if inp_hash not in seen and inp:
                    seen.add(inp_hash)
                    fout.write(json.dumps(entry) + "\n")
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue


def clean_logs():
    in_path = "logs/interaction_logs.jsonl"
    out_path = "data/processed/cleaned.jsonl"
    temp_path = "data/processed/cleaned_temp.jsonl"
    
    # Ensure input file exists
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file {in_path} not found")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Use atomic write to prevent data corruption
    try:
        with open(in_path, "r", encoding="utf-8") as fin, open(
            temp_path, "w", encoding="utf-8"
        ) as fout:
            for line_num, line in enumerate(fin, 1):
                try:
                    entry = json.loads(line.strip())
                    entry["input"] = strip_pii(normalize_text(entry.get("input", "")))
                    entry["model_response"] = strip_pii(
                        normalize_text(entry.get("model_response", ""))
                    )
                    fout.write(json.dumps(entry) + "\n")
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
        
        # Deduplicate using temp file
        dedupe_by_input(temp_path, out_path)
        
        # Remove temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to clean logs: {e}")
