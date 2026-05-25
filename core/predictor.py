import pandas as pd
import numpy as np
from Utils.logger import get_logger

logger = get_logger(__name__)


class RacePredictor:
    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.trainer.feature_columns
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0)

        probas = self.trainer.predict_proba(X)

        result = df.copy()
        result["win_probability"] = probas
        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["predicted_rank"] = range(1, len(result) + 1)

        logger.info(f"예측 완료: {len(result)}마리, 최고 확률={probas.max():.3f}")
        return result

    def predict_from_db(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
        from Utils.database import get_client
        client = get_client()

        query = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).eq("race_date", race_date)

        if meet_code:
            query = query.eq("races.meet_code", meet_code)

        result = query.execute()
        if not result.data:
            logger.warning(f"{race_date} 출전 데이터 없음")
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df = self.preprocessor.transform(df)
        return self.predict(df)
