"""MLflow Model Registry wrapper.

Manages model promotion through lifecycle stages:
  None → Staging → Production

Built on top of ``ExperimentTracker`` but focused exclusively on
the model registry lifecycle.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Thin wrapper around MLflow's model registry."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        model_name: str = "rdkit-best-model",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
            self._ensure_registered_model()
        return self._client

    def _ensure_registered_model(self) -> None:
        """Create the registered model if it doesn't exist yet."""
        client = self._client
        try:
            client.get_registered_model(self.model_name)
        except Exception:
            client.create_registered_model(
                self.model_name,
                description="Best model from the R&D multi-agent pipeline",
            )
            logger.info("registered_model_created", name=self.model_name)

    def register(self, run_id: str, artifact_path: str = "model") -> str:
        """Register a model version from an MLflow run.

        Returns:
            The new version number as a string.
        """
        import mlflow

        client = self._ensure_client()
        model_uri = f"runs:/{run_id}/{artifact_path}"

        result = mlflow.register_model(model_uri, self.model_name)
        version = result.version
        logger.info("model_registered", run_id=run_id, version=version)
        return str(version)

    def promote(self, version: str, stage: str = "Staging") -> None:
        """Transition a model version to Staging or Production."""
        client = self._ensure_client()
        valid_stages = {"Staging", "Production", "Archived", "None"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage {stage!r}. Must be one of {valid_stages}")

        client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
        )
        logger.info("model_promoted", version=version, stage=stage)

    def get_best(
        self,
        metric: str = "val_auc",
        stage: str | None = None,
    ) -> dict[str, Any] | None:
        """Get the best model version by a given metric.

        If ``stage`` is provided, only versions in that stage are
        considered.
        """
        client = self._ensure_client()
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)

        versions = client.search_model_versions(f"name='{self.model_name}'")
        if not versions:
            return None

        if stage:
            versions = [v for v in versions if v.current_stage == stage]

        best: dict[str, Any] | None = None
        best_metric = float("-inf")

        for v in versions:
            try:
                run = client.get_run(v.run_id)
                val = run.data.metrics.get(metric, float("-inf"))
                if val > best_metric:
                    best_metric = val
                    best = {
                        "version": v.version,
                        "run_id": v.run_id,
                        "stage": v.current_stage,
                        "metric": metric,
                        "metric_value": val,
                    }
            except Exception:
                continue

        return best

    def list_versions(self, stage: str | None = None) -> list[dict[str, Any]]:
        """List all registered model versions."""
        client = self._ensure_client()
        versions = client.search_model_versions(f"name='{self.model_name}'")
        results = []
        for v in versions:
            if stage and v.current_stage != stage:
                continue
            results.append({
                "version": v.version,
                "run_id": v.run_id,
                "stage": v.current_stage,
                "status": v.status,
                "created_at": v.creation_timestamp,
            })
        return results
