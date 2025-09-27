from crucible.trainer.train import main as train_main


def test_training_smoke(monkeypatch):
    # Use a small synthetic log file
    import os
    import json

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/interaction_logs.jsonl"
    with open(log_path, "w") as f:
        for idx in range(4):
            f.write(
                json.dumps({
                    "input": f"Q{idx}",
                    "model_response": f"A{idx}"
                }) + "\n"
            )
    try:
        train_main()
    except SystemExit:
        pass
    assert os.path.exists("models/latest/epoch1/metadata.json")
