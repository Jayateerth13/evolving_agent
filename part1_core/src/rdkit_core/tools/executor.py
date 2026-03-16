"""Sandboxed code execution for running untrusted Python.

Provides two backends:
  - **DockerExecutor** (production): runs code in an isolated Docker
    container with hard CPU / memory / network limits.
  - **LocalExecutor** (development / hackathon): runs code in a subprocess
    with a timeout — no isolation, but zero setup.

Both implement the same ``execute()`` interface and return an
``ExecutionResult``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog

from rdkit_core.models.spec import ExecutionResult

logger = structlog.get_logger(__name__)

_RUNNER_TEMPLATE = """\
import json, sys, os, traceback

os.chdir("{workdir}")
_artifacts = {{}}

try:
{indented_code}
except Exception:
    traceback.print_exc()
    sys.exit(1)

# Write collected artifacts
with open("{artifacts_path}", "w") as _f:
    json.dump(_artifacts, _f)
"""


class BaseExecutor(ABC):
    """Common interface for all executor backends."""

    @abstractmethod
    def execute(
        self,
        code: str,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        ...


class LocalExecutor(BaseExecutor):
    """Subprocess-based executor — no Docker required.

    Suitable for development and hackathon use.  Provides timeout
    enforcement but no resource isolation.
    """

    def __init__(self, work_dir: str | Path | None = None) -> None:
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="rdkit_exec_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        code: str,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        run_dir = Path(tempfile.mkdtemp(dir=self.work_dir))
        script_path = run_dir / "run.py"
        artifacts_path = run_dir / "artifacts.json"

        indented = "\n".join(f"    {line}" for line in code.splitlines())
        script_path.write_text(
            _RUNNER_TEMPLATE.format(
                workdir=str(run_dir),
                indented_code=indented,
                artifacts_path=str(artifacts_path),
            )
        )

        run_env = {**os.environ, **(env or {})}
        start = time.perf_counter()

        try:
            proc = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
                cwd=str(run_dir),
            )
            elapsed = time.perf_counter() - start
            artifacts = self._read_artifacts(artifacts_path)

            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                artifacts=artifacts,
                elapsed_seconds=round(elapsed, 3),
                success=proc.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            logger.warning("execution_timeout", timeout=timeout, elapsed=elapsed)
            return ExecutionResult(
                stderr=f"Execution timed out after {timeout}s",
                exit_code=-1,
                elapsed_seconds=round(elapsed, 3),
                timed_out=True,
                success=False,
            )

    @staticmethod
    def _read_artifacts(path: Path) -> dict[str, str]:
        if path.exists():
            import json

            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}


class DockerExecutor(BaseExecutor):
    """Docker-based sandbox with hard resource limits.

    Requires Docker daemon running.  Falls back to ``LocalExecutor``
    if Docker is unavailable (with a warning).
    """

    DEFAULT_IMAGE = "python:3.11-slim"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        cpu_limit: int = 4,
        memory_limit: str = "4g",
        network_disabled: bool = True,
    ) -> None:
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.network_disabled = network_disabled
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
                self._client.ping()
            except Exception as exc:
                raise RuntimeError(
                    "Docker is not available. Use LocalExecutor for development."
                ) from exc
        return self._client

    def execute(
        self,
        code: str,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        client = self._get_client()

        with tempfile.TemporaryDirectory(prefix="rdkit_docker_") as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "run.py"
            artifacts_path = tmp_path / "artifacts.json"

            indented = "\n".join(f"    {line}" for line in code.splitlines())
            script.write_text(
                _RUNNER_TEMPLATE.format(
                    workdir="/workspace",
                    indented_code=indented,
                    artifacts_path="/workspace/artifacts.json",
                )
            )

            start = time.perf_counter()
            container = None
            try:
                container = client.containers.run(
                    self.image,
                    command=["python", "/workspace/run.py"],
                    volumes={str(tmp_path): {"bind": "/workspace", "mode": "rw"}},
                    nano_cpus=self.cpu_limit * int(1e9),
                    mem_limit=self.memory_limit,
                    network_disabled=self.network_disabled,
                    detach=True,
                    remove=False,
                    environment=env or {},
                )

                result = container.wait(timeout=timeout)
                elapsed = time.perf_counter() - start
                logs = container.logs().decode("utf-8", errors="replace")
                exit_code = result.get("StatusCode", -1)

                stdout_parts = []
                stderr_parts = []
                for line in logs.splitlines():
                    stdout_parts.append(line)

                stats = container.stats(stream=False)
                peak_mem = stats.get("memory_stats", {}).get("max_usage", 0) / (1024 * 1024)

                artifacts = {}
                if artifacts_path.exists():
                    import json

                    try:
                        artifacts = json.loads(artifacts_path.read_text())
                    except (json.JSONDecodeError, OSError):
                        pass

                return ExecutionResult(
                    stdout="\n".join(stdout_parts),
                    stderr="\n".join(stderr_parts),
                    exit_code=exit_code,
                    artifacts=artifacts,
                    peak_memory_mb=round(peak_mem, 2),
                    elapsed_seconds=round(elapsed, 3),
                    success=exit_code == 0,
                )

            except Exception as exc:
                elapsed = time.perf_counter() - start
                is_timeout = "timed out" in str(exc).lower() or "read timed out" in str(exc).lower()

                if container:
                    try:
                        container.kill()
                    except Exception:
                        pass

                if is_timeout:
                    logger.warning("docker_timeout", timeout=timeout)
                    return ExecutionResult(
                        stderr=f"Docker execution timed out after {timeout}s",
                        exit_code=-1,
                        elapsed_seconds=round(elapsed, 3),
                        timed_out=True,
                        success=False,
                    )
                raise

            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass


def create_executor(
    backend: str = "local",
    **kwargs: Any,
) -> BaseExecutor:
    """Factory: create an executor by backend name.

    Args:
        backend: ``"docker"`` or ``"local"``
        **kwargs: forwarded to the chosen executor class.
    """
    if backend == "docker":
        return DockerExecutor(**kwargs)
    if backend == "local":
        return LocalExecutor(**kwargs)
    raise ValueError(f"Unknown executor backend: {backend!r}")


CodeExecutor = create_executor
