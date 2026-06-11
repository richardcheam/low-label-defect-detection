"""Thin MLflow experiment-tracking wrapper with a no-op fallback.

This module provides :class:`ExperimentTracker`, a small convenience layer
around the MLflow tracking API. It exists so that training/evaluation code
can log params, metrics, and artifacts without every CLI/test having to
worry about whether tracking is enabled or whether ``mlflow`` is even
installed.

Two situations must "just work" without raising:

1. ``mlflow`` is not importable in the current environment (e.g. a minimal
   CI image). In that case ``ExperimentTracker`` silently disables itself.
2. The caller explicitly passes ``enabled=False`` (e.g. tests, quick
   debugging runs, or environments where local run files are unwanted).

In both cases every method becomes a no-op and the :meth:`ExperimentTracker.run`
context manager simply yields ``self`` without starting an MLflow run, so
calling code does not need to branch on whether tracking is active.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


class ExperimentTracker:
    """A minimal MLflow wrapper that degrades gracefully to a no-op.

    Args:
        enabled: Whether tracking should be active. Forced to ``False`` if
            ``mlflow`` cannot be imported.
        tracking_uri: Optional MLflow tracking URI (e.g. a local
            ``file://`` path). Ignored when tracking is disabled.
        experiment: Optional MLflow experiment name to log runs under.
            Ignored when tracking is disabled.
    """

    def __init__(
        self,
        enabled: bool = True,
        tracking_uri: str | None = None,
        experiment: str | None = None,
    ) -> None:
        self.enabled = enabled and _HAS_MLFLOW

        if self.enabled:
            if tracking_uri is not None:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment is not None:
                mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, run_name: str | None = None) -> Iterator[ExperimentTracker]:
        """Context manager wrapping an MLflow run (no-op when disabled)."""
        if not self.enabled:
            yield self
            return

        with mlflow.start_run(run_name=run_name):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a batch of hyperparameters/config values (no-op when disabled)."""
        if not self.enabled:
            return
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a batch of metrics, cast to ``float`` (no-op when disabled)."""
        if not self.enabled:
            return
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Log a local file or directory as an MLflow artifact (no-op when disabled)."""
        if not self.enabled:
            return
        mlflow.log_artifact(str(path))
