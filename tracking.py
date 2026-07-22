from __future__ import annotations
import mlflow
import mlflow.tensorflow


def log_tf_model(model, model_name: str, params: dict = None, metrics: dict = None):
    """Log a TensorFlow model to MLflow with params and metrics."""
    mlflow.set_experiment("AI_Insights_Engine")
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, model_name)
        for k, v in (params or {}).items():
            mlflow.log_param(k, v)
        for k, v in (metrics or {}).items():
            mlflow.log_metric(k, v)
