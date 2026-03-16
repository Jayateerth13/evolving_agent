"""Versioned dataset storage with content-based hashing.

Stores DataFrames under ``data/versions/{version_id}/`` where the
version_id is a stable hash of the content.  This gives automatic
deduplication and full lineage tracking.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DataStore:
    """Content-addressed versioned dataset store.

    Supports local filesystem out of the box.  For S3, pass an
    ``s3://bucket/prefix`` as ``base_path`` and install the ``s3``
    extra (``pip install rdkit-core[s3]``).
    """

    def __init__(self, base_path: str | Path = "data/versions") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a DataFrame and return a deterministic version_id."""
        content_hash = self._hash_dataframe(df)
        version_id = f"{name}_{content_hash[:12]}"
        version_dir = self.base_path / version_id

        if version_dir.exists():
            logger.info("dataset_exists", version_id=version_id)
            return version_id

        version_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(version_dir / "data.parquet", index=False)

        meta = {
            "version_id": version_id,
            "name": name,
            "content_hash": content_hash,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        (version_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info("dataset_saved", version_id=version_id, shape=df.shape)
        return version_id

    def load_dataset(self, version_id: str) -> pd.DataFrame:
        """Load a dataset by its version_id."""
        path = self.base_path / version_id / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Dataset version not found: {version_id}")
        return pd.read_parquet(path)

    def get_metadata(self, version_id: str) -> dict[str, Any]:
        """Load metadata for a dataset version."""
        path = self.base_path / version_id / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found for: {version_id}")
        return json.loads(path.read_text())

    def list_versions(self, name: str | None = None) -> list[dict[str, Any]]:
        """List all dataset versions, optionally filtered by name."""
        versions = []
        for d in sorted(self.base_path.iterdir()):
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if name and meta.get("name") != name:
                continue
            versions.append(meta)
        return versions

    def version_exists(self, version_id: str) -> bool:
        return (self.base_path / version_id / "data.parquet").exists()

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        """Deterministic hash of a DataFrame's contents."""
        h = hashlib.sha256()
        h.update(str(sorted(df.columns.tolist())).encode())
        h.update(str(df.dtypes.tolist()).encode())
        h.update(pd.util.hash_pandas_object(df).values.tobytes())
        return h.hexdigest()
