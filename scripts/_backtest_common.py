"""backtest_top3.py / backtest_win.py 공통 로딩·예측 로직"""
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from Utils.database import get_client
from core.feature_builder import RaceFeatureBuilder
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.model_manager import ModelManager

REPORTS_DIR = Path(__file__).resolve().parent.parent / "BackTestReport"


class _Tee:
    """여러 스트림에 동시에 write하기 위한 헬퍼 (콘솔 출력 + 캡처용 버퍼)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class ReportCapture:
    """`with capture_report(...) as rc:` 블록 종료 후 저장된 경로를 담는 컨테이너."""

    def __init__(self, kind: str):
        self.kind = kind
        self.path: Path | None = None


@contextlib.contextmanager
def capture_report(kind: str):
    """print() 출력을 콘솔에 그대로 보여주면서 동시에 캡처해,
    블록 종료 시 BackTestReport/Result_{kind}_{날짜}_{시간}.md로 저장한다."""
    holder = ReportCapture(kind)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            yield holder
    finally:
        REPORTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"Result_{kind}_{ts}.md"
        path.write_text(f"```text\n{buf.getvalue()}\n```\n", encoding="utf-8")
        holder.path = path


def load_historical_data(start_date: str, end_date: str) -> pd.DataFrame:
    """final_rank 있는 과거 race_entries 로드"""
    client = get_client()
    all_entries = []
    page_size = 1000
    offset = 0
    while True:
        result = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).gte("race_date", start_date)\
         .lte("race_date", end_date)\
         .gt("final_rank", 0)\
         .order("id")\
         .range(offset, offset + page_size - 1).execute()
        if not result.data:
            break
        all_entries.extend(result.data)
        if len(result.data) < page_size:
            break
        offset += page_size

    if not all_entries:
        return pd.DataFrame()

    fb = RaceFeatureBuilder()
    return fb._flatten_entries(all_entries)


def run_predictions(start_date: str, end_date: str):
    """모델 로드 → 과거 데이터 로드 → 피처 빌드 → 전처리 → 예측까지 수행.

    반환 DataFrame 컬럼: race_id, race_date, final_rank, top3_prob, win_prob,
    prev_win_odds, prev_total_races, n_horses, is_top3_actual, is_win_actual.
    모델이 없거나 데이터가 없으면 None (오류 메시지는 내부에서 출력).
    """
    print("모델 로드 중...")
    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()
    if not mm.load(trainer, preprocessor):
        print("[오류] 저장된 모델 없음. python main.py --mode train 먼저 실행하세요.")
        return None

    print(f"과거 데이터 로드 중: {start_date} ~ {end_date} ...")
    df_raw = load_historical_data(start_date, end_date)
    if df_raw.empty:
        print("[오류] 데이터 없음")
        return None
    print(f"  로드: {len(df_raw)}건, {df_raw['race_id'].nunique() if 'race_id' in df_raw.columns else '?'}경주")

    print("피처 빌드 중...")
    fb = RaceFeatureBuilder()
    df_feat = fb.build_for_training(df_raw, end_date)
    df_feat = df_feat.dropna(subset=["horse_age", "race_distance"])

    final_ranks   = df_feat["final_rank"].copy()
    race_ids      = df_feat["race_id"].copy() if "race_id" in df_feat.columns else pd.Series(range(len(df_feat)))
    race_dates    = df_feat["race_date"].copy() if "race_date" in df_feat.columns else pd.Series([""] * len(df_feat))
    prev_win_odds = df_feat["prev_win_odds"].copy() if "prev_win_odds" in df_feat.columns else pd.Series([0.0] * len(df_feat))
    prev_total_races = df_feat["prev_total_races"].copy() if "prev_total_races" in df_feat.columns else pd.Series([0] * len(df_feat))

    n_horses_map = df_feat.groupby("race_id").size().rename("n_horses")

    print("전처리 중...")
    df_proc = preprocessor.transform(df_feat)

    print("예측 중...")
    features  = trainer.feature_columns
    available = [c for c in features if c in df_proc.columns]
    X = df_proc[available].fillna(0)

    top3_probs = trainer.predict_proba_top3(X)
    win_probs  = trainer.predict_proba(X)

    if top3_probs is None:
        print("[경고] top3_models 없음. win_probability로 대체합니다.")
        top3_probs = win_probs

    result_df = pd.DataFrame({
        "race_id":       race_ids.values,
        "race_date":     race_dates.values,
        "final_rank":    final_ranks.values,
        "top3_prob":     top3_probs,
        "win_prob":      win_probs,
        "prev_win_odds": prev_win_odds.values,
        "prev_total_races": prev_total_races.values,
    }, index=df_feat.index)

    result_df["is_top3_actual"] = (result_df["final_rank"] <= 3).astype(int)
    result_df["is_win_actual"]  = (result_df["final_rank"] == 1).astype(int)
    result_df = result_df.join(n_horses_map, on="race_id")

    return result_df
