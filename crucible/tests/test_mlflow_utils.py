from crucible.tools.mlflow_utils import log_mlflow_run
import tempfile
import os
import json

def test_log_mlflow_run(tmp_path, monkeypatch):
    # Patch mlflow to avoid real network calls
    class DummyMLflow:
        def set_tracking_uri(self, uri):
            self.uri = uri
        def start_run(self):
            class DummyRun:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return DummyRun()
        def log_params(self, params): pass
        def log_metrics(self, metrics): pass
        def log_artifacts(self, path): pass
        def set_tag(self, k, v): pass
        def log_artifact(self, path): pass
        def register_model(self, path, name): pass
        def active_run(self):
            class Info: info = type('info', (), {'run_id': 'dummy'})
            return type('run', (), {'info': Info()})
    monkeypatch.setattr("crucible.tools.mlflow_utils.mlflow", DummyMLflow())
    meta = {"hyperparams": {}, "metrics": {}, "git_sha": "abc", "dataset_hash": "xyz"}
    d = tmp_path / "artifacts"
    d.mkdir()
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f)
    log_mlflow_run(meta, str(d))
