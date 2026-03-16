"""Tests for the code executor (local subprocess backend)."""

from __future__ import annotations

from rdkit_core.tools.executor import LocalExecutor, create_executor


class TestLocalExecutor:
    def test_simple_execution(self, tmp_path):
        executor = LocalExecutor(work_dir=tmp_path)
        result = executor.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout
        assert result.exit_code == 0

    def test_execution_with_error(self, tmp_path):
        executor = LocalExecutor(work_dir=tmp_path)
        result = executor.execute("raise ValueError('test error')")
        assert not result.success
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    def test_timeout(self, tmp_path):
        executor = LocalExecutor(work_dir=tmp_path)
        result = executor.execute("import time; time.sleep(10)", timeout=1)
        assert result.timed_out
        assert not result.success
        assert result.exit_code == -1

    def test_multiline_code(self, tmp_path):
        executor = LocalExecutor(work_dir=tmp_path)
        code = """
x = 10
y = 20
print(x + y)
"""
        result = executor.execute(code)
        assert result.success
        assert "30" in result.stdout

    def test_artifacts(self, tmp_path):
        executor = LocalExecutor(work_dir=tmp_path)
        code = """
_artifacts["result"] = "42"
_artifacts["model"] = "lightgbm"
"""
        result = executor.execute(code)
        assert result.success
        assert result.artifacts.get("result") == "42"
        assert result.artifacts.get("model") == "lightgbm"

    def test_elapsed_time(self, tmp_path):
        executor = LocalExecutor(work_dir=tmp_path)
        result = executor.execute("import time; time.sleep(0.1)")
        assert result.success
        assert result.elapsed_seconds >= 0.1


class TestCreateExecutor:
    def test_local_backend(self):
        executor = create_executor(backend="local")
        assert isinstance(executor, LocalExecutor)

    def test_invalid_backend(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown executor backend"):
            create_executor(backend="invalid")
