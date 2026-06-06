import json
import joblib
from datetime import datetime
from config import MODEL_DIR, MODEL_CONFIG
from Utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)

    def save(self, trainer, preprocessor, metrics: dict = None) -> str | None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = MODEL_DIR / f"model_{version}.pkl"
        meta_path = MODEL_DIR / f"model_{version}_meta.json"

        data = {
            "models": trainer.models,
            "top3_models": trainer.top3_models,
            "scaler": trainer.scaler,
            "feature_columns": trainer.feature_columns,
            "label_encoders": preprocessor.label_encoders,
        }
        meta = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "feature_count": len(trainer.feature_columns),
            "metrics": metrics or {},
        }

        try:
            joblib.dump(data, pkl_path)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"모델 저장: {pkl_path}")
            return version
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return None

    def load(self, trainer, preprocessor, version: str = "latest") -> bool:
        if version == "latest":
            files = sorted(MODEL_DIR.glob("model_*.pkl"))
            if not files:
                logger.warning("저장된 모델 없음")
                return False
            pkl_path = files[-1]
        else:
            pkl_path = MODEL_DIR / f"model_{version}.pkl"

        if not pkl_path.exists():
            logger.error(f"모델 파일 없음: {pkl_path}")
            return False

        try:
            data = joblib.load(pkl_path)
            trainer.models = data["models"]
            trainer.top3_models = data.get("top3_models", {})
            trainer.scaler = data["scaler"]
            trainer.feature_columns = data["feature_columns"]
            preprocessor.label_encoders = data["label_encoders"]
            logger.info(f"모델 로드: {pkl_path}")
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def is_outdated(self, version: str = "latest") -> bool:
        if version == "latest":
            files = sorted(MODEL_DIR.glob("model_*_meta.json"))
            if not files:
                return True
            meta_path = files[-1]
        else:
            meta_path = MODEL_DIR / f"model_{version}_meta.json"

        if not meta_path.exists():
            return True

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        created = datetime.fromisoformat(meta["created_at"])
        days_old = (datetime.now() - created).days
        return days_old > MODEL_CONFIG["model_expiry_days"]
