"""Local training-metrics logging: a JSONL file, always; a local ``mlflow``
run, optionally. No external services required -- when enabled, mlflow is
pointed at a local ``file:`` tracking URI (e.g. ``judge/mlruns/``) rather
than a remote tracking server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(
        self,
        path: Path | str,
        use_mlflow: bool = False,
        mlflow_dir: Path | str | None = None,
        run_name: str | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mlflow: Any = None
        if use_mlflow:
            try:
                import mlflow

                mlflow.set_tracking_uri(f"file:{Path(mlflow_dir or 'judge/mlruns').resolve()}")
                mlflow.start_run(run_name=run_name)
                self._mlflow = mlflow
            except ImportError:
                print("mlflow not installed; falling back to JSONL-only metrics logging.")

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self._mlflow is not None:
            step = record.get("step")
            numeric = {
                k: v for k, v in record.items() if isinstance(v, (int, float)) and k != "step"
            }
            if numeric:
                self._mlflow.log_metrics(numeric, step=step)

    def close(self) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()
