"""MLflow experiment tracking wrapper.

Provides a thin, typed interface over MLflow so agents don't need to
know MLflow internals.  All interactions go through ``ExperimentTracker``.
"""

from __future__ import annotations

from typing import Any

import structlog

from rdkit_core.models.spec import RunDiff

logger = structlog.get_logger(__name__)


class ExperimentTracker:
    """Wrapper around MLflow tracking for experiment lifecycle management."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "rdkit-multiagent",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._client: Any = None
        self._experiment_id: str | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)

            experiment = self._client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self._experiment_id = self._client.create_experiment(self.experiment_name)
            else:
                self._experiment_id = experiment.experiment_id

            logger.info(
                "tracker_initialized",
                uri=self.tracking_uri,
                experiment=self.experiment_name,
                experiment_id=self._experiment_id,
            )
        return self._client

    @property
    def experiment_id(self) -> str:
        self._ensure_client()
        assert self._experiment_id is not None
        return self._experiment_id

    def log_run(
        self,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        artifacts: dict[str, str] | None = None,
        run_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        """Start an MLflow run, log everything, and return the run_id."""
        import mlflow

        client = self._ensure_client()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        nested = parent_run_id is not None
        tags_dict = dict(tags or {})
        if parent_run_id:
            tags_dict["mlflow.parentRunId"] = parent_run_id

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested,
            tags=tags_dict,
        ) as run:
            if params:
                mlflow.log_params(self._flatten(params))
            if metrics:
                mlflow.log_metrics(metrics)
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

            run_id = run.info.run_id
            logger.info("run_logged", run_id=run_id, metrics=metrics)
            return run_id

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Fetch a run by ID, returning params + metrics as a dict."""
        client = self._ensure_client()
        run = client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }

    def diff_runs(self, run_id_a: str, run_id_b: str) -> RunDiff:
        """Compare two runs: metric deltas and feature importance shifts."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        metrics_a: dict[str, float] = run_a["metrics"]
        metrics_b: dict[str, float] = run_b["metrics"]

        all_keys = set(metrics_a) | set(metrics_b)
        deltas = {}
        for k in all_keys:
            val_a = metrics_a.get(k, 0.0)
            val_b = metrics_b.get(k, 0.0)
            deltas[k] = round(val_b - val_a, 6)

        fi_a = self._extract_feature_importance(run_a)
        fi_b = self._extract_feature_importance(run_b)
        fi_keys = set(fi_a) | set(fi_b)
        fi_shifts = {k: round(fi_b.get(k, 0.0) - fi_a.get(k, 0.0), 6) for k in fi_keys}

        return RunDiff(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            metric_deltas=deltas,
            feature_importance_shifts=fi_shifts,
            summary=self._build_diff_summary(deltas),
        )

    def list_runs(
        self,
        max_results: int = 50,
        order_by: str = "metrics.val_auc DESC",
    ) -> list[dict[str, Any]]:
        """List runs in the current experiment, ordered by a metric."""
        client = self._ensure_client()
        from mlflow.entities import ViewType

        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_results,
            order_by=[order_by],
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        return [
            {
                "run_id": r.info.run_id,
                "status": r.info.status,
                "metrics": dict(r.data.metrics),
                "params": dict(r.data.params),
            }
            for r in runs
        ]

    def get_best_run(self, metric: str = "val_auc") -> dict[str, Any] | None:
        """Return the single best run by a given metric (descending)."""
        runs = self.list_runs(max_results=1, order_by=f"metrics.{metric} DESC")
        return runs[0] if runs else None

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
        """Flatten a nested dict for MLflow param logging."""
        flat: dict[str, str] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(ExperimentTracker._flatten(v, key))
            else:
                flat[key] = str(v)
        return flat

    @staticmethod
    def _extract_feature_importance(run_data: dict[str, Any]) -> dict[str, float]:
        """Pull feature importance from metrics (convention: fi_{name} keys)."""
        return {
            k.removeprefix("fi_"): v
            for k, v in run_data["metrics"].items()
            if k.startswith("fi_")
        }

    @staticmethod
    def _build_diff_summary(deltas: dict[str, float]) -> str:
        parts = []
        for k, v in sorted(deltas.items()):
            direction = "+" if v > 0 else ""
            parts.append(f"{k}: {direction}{v:.4f}")
        return ", ".join(parts) if parts else "no metric differences"
