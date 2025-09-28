import mlflow
import os
import json


def log_mlflow_run(meta: dict, artifact_path: str):
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        
        # Verify artifact path exists
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact path {artifact_path} does not exist")
        
        with mlflow.start_run():
            # Log parameters safely
            hyperparams = meta.get("hyperparams", {})
            if hyperparams:
                # Convert non-serializable values to strings
                safe_params = {}
                for k, v in hyperparams.items():
                    try:
                        # Test if value is JSON serializable
                        json.dumps(v)
                        safe_params[k] = v
                    except (TypeError, ValueError):
                        safe_params[k] = str(v)
                mlflow.log_params(safe_params)
            
            # Log metrics safely
            metrics = meta.get("metrics", {})
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Log artifacts
            mlflow.log_artifacts(artifact_path)
            
            # Set tags safely
            mlflow.set_tag("git_sha", str(meta.get("git_sha", "unknown")))
            mlflow.set_tag("dataset_hash", str(meta.get("dataset_hash", "unknown")))
            
            # Log metadata file if it exists
            metadata_file = os.path.join(artifact_path, "metadata.json")
            if os.path.exists(metadata_file):
                mlflow.log_artifact(metadata_file)
            
            # Register model safely
            try:
                model_name = meta.get("hyperparams", {}).get("checkpoint", "crucible_model")
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                mlflow.register_model(model_uri, model_name)
            except Exception as e:
                print(f"Warning: Failed to register model: {e}")
                
    except Exception as e:
        print(f"Warning: MLflow logging failed: {e}")
        # Don't raise the exception to prevent training from failing
