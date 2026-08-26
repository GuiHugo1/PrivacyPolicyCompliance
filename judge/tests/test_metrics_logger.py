import json

from judge.metrics_logger import MetricsLogger


def test_metrics_logger_writes_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(path)

    logger.log({"step": 1, "train_loss": 1.2, "eval_loss": 1.5, "json_validity_rate": 0.8})
    logger.log({"step": 2, "train_loss": 1.0, "eval_loss": 1.3, "json_validity_rate": 0.9})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "step": 1,
        "train_loss": 1.2,
        "eval_loss": 1.5,
        "json_validity_rate": 0.8,
    }
    assert json.loads(lines[1])["step"] == 2


def test_metrics_logger_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "metrics.jsonl"
    logger = MetricsLogger(path)
    logger.log({"step": 1})
    assert path.exists()


def test_metrics_logger_close_without_mlflow_is_noop(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.jsonl")
    logger.close()  # should not raise
