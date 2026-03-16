"""Tests for the versioned DataStore."""

from __future__ import annotations

import pandas as pd
import pytest

from rdkit_core.tools.datastore import DataStore


class TestDataStore:
    def test_save_and_load(self, tmp_path):
        store = DataStore(base_path=tmp_path / "versions")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        vid = store.save_dataset(df, name="test_ds")
        assert vid.startswith("test_ds_")
        assert store.version_exists(vid)

        loaded = store.load_dataset(vid)
        pd.testing.assert_frame_equal(df, loaded)

    def test_content_hash_deterministic(self, tmp_path):
        store = DataStore(base_path=tmp_path / "versions")
        df = pd.DataFrame({"x": [10, 20]})

        vid1 = store.save_dataset(df, name="ds")
        vid2 = store.save_dataset(df, name="ds")
        assert vid1 == vid2

    def test_different_data_different_hash(self, tmp_path):
        store = DataStore(base_path=tmp_path / "versions")
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})

        vid1 = store.save_dataset(df1, name="ds")
        vid2 = store.save_dataset(df2, name="ds")
        assert vid1 != vid2

    def test_metadata(self, tmp_path):
        store = DataStore(base_path=tmp_path / "versions")
        df = pd.DataFrame({"col1": [1.0, 2.0], "col2": ["a", "b"]})

        vid = store.save_dataset(df, name="meta_test", metadata={"source": "csv"})
        meta = store.get_metadata(vid)

        assert meta["name"] == "meta_test"
        assert meta["n_rows"] == 2
        assert meta["n_cols"] == 2
        assert "col1" in meta["columns"]
        assert meta["source"] == "csv"

    def test_list_versions(self, tmp_path):
        store = DataStore(base_path=tmp_path / "versions")
        store.save_dataset(pd.DataFrame({"a": [1]}), name="alpha")
        store.save_dataset(pd.DataFrame({"b": [2]}), name="beta")

        all_versions = store.list_versions()
        assert len(all_versions) == 2

        alpha_only = store.list_versions(name="alpha")
        assert len(alpha_only) == 1
        assert alpha_only[0]["name"] == "alpha"

    def test_load_nonexistent(self, tmp_path):
        store = DataStore(base_path=tmp_path / "versions")
        with pytest.raises(FileNotFoundError):
            store.load_dataset("nonexistent_abc123")
