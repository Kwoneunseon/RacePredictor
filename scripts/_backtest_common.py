"""backtest_top3.py / backtest_win.py 공통 로딩·예측 로직"""
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from Utils.database import get_client
from core.feature_builder import RaceFeatureBuilder
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.model_manager import ModelManager
from core.rank_distribution import top3_marginal_probabilities, pair_top3_probabilities

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
    return fb.get_flatten_entries(all_entries)


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


def attach_harville_top3(result_df: pd.DataFrame) -> pd.DataFrame:
    """race_id별로 win_prob을 강도점수 삼아 Harville 순차소거로 유도한
    연승확률(harville_top3_prob) 컬럼을 추가해 반환한다.

    core.rank_distribution.top3_marginal_probabilities()는 출전마 3두 미만이거나
    강도 합이 0 이하면 ValueError를 던지므로, 그런 경주는 harville_top3_prob을
    NaN으로 채우고 건너뛴다(전체 백테스트를 중단시키지 않기 위함,
    [[project_multi_bet_probability_model]] 참고).

    groupby(...)["win_prob"].transform(...)를 쓰는 이유: groupby.apply()에
    그룹핑 컬럼(race_id)이 포함된 프레임을 통째로 넘기면 pandas가
    DeprecationWarning을 낸다(향후 그룹핑 컬럼이 자동 제외될 예정) - transform은
    win_prob 컬럼 하나만 다루므로 이 문제가 없다.
    """
    def _harville_col(s: pd.Series) -> pd.Series:
        strengths = s.to_numpy()
        if len(strengths) < 3 or strengths.sum() <= 0:
            return pd.Series(np.nan, index=s.index)
        return pd.Series(top3_marginal_probabilities(strengths), index=s.index)

    out = result_df.copy()
    out["harville_top3_prob"] = out.groupby("race_id", sort=False)["win_prob"].transform(_harville_col)

    n_skipped_races = out.loc[out["harville_top3_prob"].isna(), "race_id"].nunique()
    if n_skipped_races:
        print(f"  [주의] Harville 연승확률 계산 불가 경주 {n_skipped_races}건 스킵 "
              f"(출전마 3두 미만 또는 win_prob 합<=0)")
    return out


def build_harville_pair_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """race_id별로 win_prob을 강도점수 삼아 pair_top3_probabilities()를 계산하고,
    확률이 가장 높은 페어(복연승 픽)와 적중 여부를 담은 경주 단위 테이블을 반환한다.

    기존 임시 휴리스틱(scripts/backtest_quinella_place_heuristic.py의
    build_quinella_table — "확률 상위 2마리를 그냥 묶는" 방식)과 달리, 두 말이
    동시에 top3에 들 결합확률 자체를 최대화하는 페어를 고른다
    ([[project_multi_bet_probability_model]] Phase 2).

    반환 컬럼: race_id, race_date, n_horses, harville_pair_prob(픽된 페어의
    확률), harville_pair_hit(픽된 페어가 실제로 둘 다 1~3착인지),
    random_baseline(무작위 2마리 픽 기대 적중률 = 6/(n*(n-1))).

    출전마 3두 미만이거나 win_prob 합이 0 이하인 경주는 결과에서 제외한다.
    """
    rows = []
    skipped_small = 0
    skipped_degenerate = 0
    for race_id, grp in result_df.groupby("race_id", sort=False):
        n = len(grp)
        if n < 3:
            skipped_small += 1
            continue

        strengths = grp["win_prob"].to_numpy()
        if strengths.sum() <= 0:
            skipped_degenerate += 1
            continue

        pair = pair_top3_probabilities(strengths)
        iu = np.triu_indices(n, k=1)
        best_idx = np.argmax(pair[iu])
        a, b = iu[0][best_idx], iu[1][best_idx]

        final_ranks = grp["final_rank"].to_numpy()
        hit = bool(final_ranks[a] <= 3 and final_ranks[b] <= 3)

        rows.append({
            "race_id": race_id,
            "race_date": str(grp["race_date"].iloc[0]),
            "n_horses": n,
            "harville_pair_prob": float(pair[a, b]),
            "harville_pair_hit": hit,
            "random_baseline": 6.0 / (n * (n - 1)),
        })

    if skipped_small:
        print(f"  [주의] 출전마 3두 미만 경주 {skipped_small}건은 복연승 정의상 스킵")
    if skipped_degenerate:
        print(f"  [주의] win_prob 합<=0 경주 {skipped_degenerate}건 스킵")

    return pd.DataFrame(rows)
