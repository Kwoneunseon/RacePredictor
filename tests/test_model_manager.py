import pytest
import joblib
from pathlib import Path
from unittest.mock import MagicMock
from core.model_manager import ModelManager


def test_save_creates_pkl_file(tmp_path, mocker):
    mocker.patch("core.model_manager.MODEL_DIR", tmp_path)
    trainer = MagicMock()
    trainer.models = {"RF": "serializable_model_object"}
    trainer.top3_models = {}
    trainer.top3_ranker_models = {}
    trainer.scaler = "serializable_scaler_object"
    trainer.feature_columns = ["col1", "col2"]
    trainer.win_calibrator = None
    trainer.top3_calibrator = None
    preprocessor = MagicMock()
    preprocessor.label_encoders = {}

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics={"auc": 0.7})
    assert version is not None
    assert (tmp_path / f"model_{version}.pkl").exists()


def test_load_restores_trainer(tmp_path, mocker):
    mocker.patch("core.model_manager.MODEL_DIR", tmp_path)
    trainer = MagicMock()
    trainer.models = {"RF": "fake_model"}
    trainer.top3_models = {"RF": "fake_top3_model"}
    trainer.top3_ranker_models = {}
    trainer.scaler = "fake_scaler"
    trainer.feature_columns = ["col1"]
    trainer.win_calibrator = None
    trainer.top3_calibrator = None
    preprocessor = MagicMock()
    preprocessor.label_encoders = {"horse_class": "enc"}

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics={})

    new_trainer = MagicMock()
    new_preprocessor = MagicMock()
    result = mm.load(new_trainer, new_preprocessor, version=version)
    assert result is True
    assert new_trainer.models == {"RF": "fake_model"}
    assert new_trainer.feature_columns == ["col1"]


def test_is_outdated_with_no_models(tmp_path, mocker):
    mocker.patch("core.model_manager.MODEL_DIR", tmp_path)
    mm = ModelManager()
    assert mm.is_outdated() is True


def test_save_load_preserves_top3_ranker_models(tmp_path, mocker):
    mocker.patch("core.model_manager.MODEL_DIR", tmp_path)
    trainer = MagicMock()
    trainer.models = {}
    trainer.top3_models = {}
    trainer.top3_ranker_models = {"XGBoost": "fake_xgb", "LightGBM": "fake_lgb"}
    trainer.scaler = "fake_scaler"
    trainer.feature_columns = ["col1"]
    trainer.win_calibrator = None
    trainer.top3_calibrator = None
    preprocessor = MagicMock()
    preprocessor.label_encoders = {}

    mm = ModelManager()
    version = mm.save(trainer, preprocessor)

    new_trainer = MagicMock()
    new_preprocessor = MagicMock()
    mm.load(new_trainer, new_preprocessor, version=version)

    assert new_trainer.top3_ranker_models == {"XGBoost": "fake_xgb", "LightGBM": "fake_lgb"}
