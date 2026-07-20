import pandas as pd
import numpy as np
from config import SOFTMAX_TEMPERATURE, LAMBDARANK_BLEND
from Utils.logger import get_logger
from core.feature_builder import RaceFeatureBuilder

logger = get_logger(__name__)


def _blend_group(grp: pd.DataFrame, lr_col: str, clf_col: str) -> pd.Series:
    """경주 내 ranker 점수와 분류 확률을 LAMBDARANK_BLEND 비율로 블렌딩 후 softmax."""
    lr = grp[lr_col].values
    clf = grp[clf_col].values
    lr_range = lr.max() - lr.min()
    lr_norm = (lr - lr.min()) / lr_range if lr_range > 1e-9 else np.full(len(lr), 0.5)
    blended = LAMBDARANK_BLEND * lr_norm + (1 - LAMBDARANK_BLEND) * clf
    return pd.Series(_softmax(blended), index=grp.index)


def _softmax(x: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE) -> np.ndarray:
    """
    경주 내 점수를 확률로 변환.
    1) min-max 정규화로 스케일 의존성 제거
    2) temperature < 1: 상위 말 확률을 더 높게 (기본 0.3)
    """
    s_min, s_max = x.min(), x.max()
    if s_max == s_min:
        return np.ones(len(x)) / len(x)
    normalized = (x - s_min) / (s_max - s_min)
    scaled = normalized / temperature
    e = np.exp(scaled - scaled.max())
    return e / e.sum()



class RacePredictor:
    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.trainer.feature_columns
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0)

        rank_scores      = self.trainer.predict_rank_score(X)
        rank_scores_top3 = self.trainer.predict_rank_score_top3(X)
        clf_probs        = self.trainer.predict_proba(X)
        clf_probs_top3   = self.trainer.predict_proba_top3(X)

        if not isinstance(rank_scores, np.ndarray):
            rank_scores = None
        if not isinstance(rank_scores_top3, np.ndarray):
            rank_scores_top3 = None
        if not isinstance(clf_probs_top3, np.ndarray):
            clf_probs_top3 = None

        result = df.copy()

        # ── 단승(win_probability) ──────────────────────────────────────
        if rank_scores is not None and "race_id" in result.columns:
            result["_lr"]  = rank_scores
            result["_clf"] = clf_probs
            win_probs = pd.Series(np.zeros(len(result)), index=result.index)
            for _, grp in result.groupby("race_id"):
                s = _blend_group(grp, "_lr", "_clf")
                win_probs[s.index] = s
            result["win_probability"] = win_probs
            result = result.drop(columns=["_lr", "_clf"])
            logger.debug(f"앙상블 블렌딩: LambdaRank {LAMBDARANK_BLEND:.0%} + 분류모델 {1-LAMBDARANK_BLEND:.0%}")
        elif rank_scores is not None:
            result["win_probability"] = _softmax(rank_scores)
        else:
            result["win_probability"] = clf_probs

        # ── 연승(top3_probability) ────────────────────────────────────
        # 이진 분류기 raw 확률을 그대로 사용.
        # 연승은 3마리가 top3에 들어오므로 경주 내 softmax 정규화가 부적합.
        # rank_scores_top3는 말 간 순서 개선에 활용하되, 확률값에는 미적용.
        if clf_probs_top3 is not None:
            result["top3_probability"] = clf_probs_top3
        else:
            result["top3_probability"] = result["win_probability"]

        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["predicted_rank"] = range(1, len(result) + 1)

        logger.info(f"예측 완료: {len(result)}마리, 최고 확률={result['win_probability'].max():.3f}")
        return result

    def predict_from_db(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
        fb = RaceFeatureBuilder()
        df = fb.build_for_prediction(race_date, meet_code)
        if df.empty:
            return pd.DataFrame()
        df = self.preprocessor.transform(df)
        return self.predict(df)



def predict_today(race_date: str) -> dict:
    """
    스케줄러 등 외부에서 재사용할 수 있는 예측 함수.
    반환: {
        "predictions": DataFrame,
        "race_info":   {race_id: {...}},
        "eligible_ids": set,
        "meet_name":   {meet_code: str},
    }
    데이터 없으면 None 반환.
    """
    from Utils.database import get_client
    from core.preprocessor import RacePreprocessor
    from core.trainer import RaceModelTrainer
    from core.model_manager import ModelManager
    from core.race_filter import RaceFilter

    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()
    if not mm.load(trainer, preprocessor):
        logger.error("로드할 모델 없음")
        return None

    predictor = RacePredictor(trainer, preprocessor)
    predictions = predictor.predict_from_db(race_date)
    if predictions.empty:
        return None

    client = get_client()
    races_res = client.table("races").select(
        "race_id, race_distance, race_grade, total_horses, race_no, meet_code"
    ).eq("race_date", race_date).execute()
    race_info = {r["race_id"]: r for r in (races_res.data or [])}

    race_filter = RaceFilter()
    eligible_ids = {rid for rid, info in race_info.items() if race_filter.is_eligible(info)}

    return {
        "predictions":  predictions,
        "race_info":    race_info,
        "eligible_ids": eligible_ids,
        "meet_name":    {1: "서울", 2: "제주", 3: "부산", 0: "미상"},
    }


def format_kakao_message(result: dict, race_date: str) -> str:
    """predict_today() 결과를 카카오톡 메시지 문자열로 변환"""
    predictions  = result["predictions"]
    race_info    = result["race_info"]
    eligible_ids = result["eligible_ids"]
    meet_name    = result["meet_name"]

    # 경마장별로 묶기
    groups: dict = {}
    for race_id, info in sorted(race_info.items(),
                                key=lambda x: (x[1].get("meet_code", 0),
                                               x[1].get("race_no", 0))):
        meet = info.get("meet_code", 0)
        groups.setdefault(meet, []).append((race_id, info))

    lines = [f"[경마 예측] {race_date}", ""]

    total_races = 0
    for meet_code, race_list in sorted(groups.items()):
        name = meet_name.get(meet_code, "미상")
        lines.append(f"■ {name} ({len(race_list)}경주)")

        for race_id, info in race_list:
            race_no = info.get("race_no", "?")
            dist    = info.get("race_distance", "?")
            flag    = "★" if race_id in eligible_ids else "  "

            # 해당 경주 1위 예측마
            race_preds = predictions[predictions["race_id"] == race_id]
            if race_preds.empty:
                continue
            top = race_preds.sort_values("win_probability", ascending=False).iloc[0]
            horse = top.get("horse_name") or top.get("horse_id", "?")
            horse_no = top.get("entry_number")
            try:
                horse_no = str(int(horse_no))
            except (TypeError, ValueError):
                horse_no = "?"
            prob  = top["win_probability"]

            lines.append(f" {flag}{race_no}경주 {dist}m | {horse_no}번 {horse} {prob:.1%}")
            total_races += 1

        lines.append("")

    lines.append(f"총 {total_races}경주 예측완료")
    return "\n".join(lines)
