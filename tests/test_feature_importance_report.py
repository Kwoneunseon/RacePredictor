from types import SimpleNamespace

import numpy as np

from core import feature_importance_report as fir


class _FakeModel:
    def __init__(self, importances):
        self.feature_importances_ = np.array(importances)


def _fake_trainer():
    return SimpleNamespace(
        feature_columns=["horse_age", "race_distance"],
        models={"RandomForest": _FakeModel([0.6, 0.4])},
        top3_models={},
        top3_ranker_models={},
    )


def test_generate_report_writes_file_from_in_memory_trainer(tmp_path, monkeypatch):
    """재학습 직후 디스크 재로딩 없이 메모리상의 trainer로 바로 리포트를 생성할 수 있어야 한다."""
    monkeypatch.setattr(fir, "REPORT_PATH", tmp_path / "feature_importance.html")
    meta = {"version": "20260712_000000", "created_at": "2026-07-12T00:00:00", "feature_count": 2, "metrics": {}}

    path = fir.generate_report(_fake_trainer(), meta)

    assert path is not None
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert "20260712_000000" in html
    assert "horse_age" in html


def test_generate_report_returns_none_when_no_importable_models(tmp_path, monkeypatch):
    monkeypatch.setattr(fir, "REPORT_PATH", tmp_path / "feature_importance.html")
    empty_trainer = SimpleNamespace(
        feature_columns=["horse_age"], models={}, top3_models={}, top3_ranker_models={},
    )
    path = fir.generate_report(empty_trainer, {})
    assert path is None
    assert not (tmp_path / "feature_importance.html").exists()
