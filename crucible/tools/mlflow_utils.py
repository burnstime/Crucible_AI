import mlflow
import os


def log_mlflow_run(meta: dict, artifact_path: str):
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    with mlflow.start_run():
        mlflow.log_params(meta.get("hyperparams", {}))
        mlflow.log_metrics(meta.get("metrics", {}))
        mlflow.log_artifacts(artifact_path)
        mlflow.set_tag("git_sha", meta.get("git_sha", "unknown"))
        mlflow.set_tag("dataset_hash", meta.get("dataset_hash", "unknown"))
        mlflow.log_artifact(os.path.join(artifact_path, "metadata.json"))
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
            meta.get("hyperparams", {}).get("checkpoint", "crucible_model"),
        )
