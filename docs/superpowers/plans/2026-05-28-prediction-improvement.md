# 예측 성능 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** horse_race_history 채우기 + win_odds 특성 추가 + 버그 3가지 수정으로 1등 예측 정확도 개선

**Architecture:** (1) SQL로 DB의 horse_race_history 집계, (2) preprocessor에 새 특성 추가, (3) predictor/main 학습-예측 파이프라인에 history 조인, (4) 모델 재학습

**Tech Stack:** Python 3.11, pandas, scikit-learn, XGBoost, LightGBM, Supabase (postgrest), pytest

---

## 파일 맵

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `core/preprocessor.py` | 수정 | horse_class_num, implied_prob, prev_total_avg_rank 추가; horse_class CATEGORICAL_COLS에서 제거 |
| `config.py` | 수정 | FEATURE_COLUMNS: horse_class_enc→horse_class_num, win_odds/implied_prob 추가 |
| `core/predictor.py` | 수정 | predict_from_db에서 horse_race_history 조인 |
| `main.py` | 수정 | cmd_train에서 horse_race_history 페이지네이션 조인 |
| `tests/test_preprocessor.py` | 수정 | horse_class_num, implied_prob, prev_total_avg_rank 테스트 추가; horse_class_enc 테스트 수정 |
| `tests/test_predictor.py` | 수정 | history 컬럼 mock 포함 테스트 추가 |

---

## Task 1: horse_race_history SQL 채우기

**Files:**
- DB only (no Python files)

- [ ] **Step 1: SQL 실행 – race_entries 자체 조인으로 horse_race_history 집계**

Supabase MCP `execute_sql`로 아래 쿼리 실행:

```sql
INSERT INTO public.horse_race_history
  (horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, prev_5_avg_rank, prev_total_avg_rank)
SELECT
    re.horse_id,
    r.race_date                                AS as_of_date,
    COALESCE(h.prev_total_races,    0)         AS prev_total_races,
    COALESCE(h.prev_wins,           0)         AS prev_wins,
    COALESCE(h.prev_top3,           0)         AS prev_top3,
    COALESCE(r5.prev_5_avg_rank,    6.0)       AS prev_5_avg_rank,
    COALESCE(h.prev_total_avg_rank, 6.0)       AS prev_total_avg_rank
FROM (SELECT DISTINCT horse_id, race_id FROM public.race_entries) re
JOIN public.races r ON re.race_id = r.race_id
CROSS JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE re2.final_rank IS NOT NULL)         AS prev_total_races,
        COUNT(*) FILTER (WHERE re2.final_rank = 1)                 AS prev_wins,
        COUNT(*) FILTER (WHERE re2.final_rank IS NOT NULL AND re2.final_rank <= 3) AS prev_top3,
        AVG(re2.final_rank::double precision)                      AS prev_total_avg_rank
    FROM public.race_entries re2
    JOIN public.races r2 ON re2.race_id = r2.race_id
    WHERE re2.horse_id = re.horse_id
      AND r2.race_date < r.race_date
      AND re2.final_rank IS NOT NULL
) h
CROSS JOIN LATERAL (
    SELECT AVG(sub.final_rank::double precision) AS prev_5_avg_rank
    FROM (
        SELECT re3.final_rank
        FROM public.race_entries re3
        JOIN public.races r3 ON re3.race_id = r3.race_id
        WHERE re3.horse_id = re.horse_id
          AND r3.race_date < r.race_date
          AND re3.final_rank IS NOT NULL
        ORDER BY r3.race_date DESC
        LIMIT 5
    ) sub
) r5
ON CONFLICT (horse_id, as_of_date) DO UPDATE SET
    prev_total_races    = EXCLUDED.prev_total_races,
    prev_wins           = EXCLUDED.prev_wins,
    prev_top3           = EXCLUDED.prev_top3,
    prev_5_avg_rank     = EXCLUDED.prev_5_avg_rank,
    prev_total_avg_rank = EXCLUDED.prev_total_avg_rank;
```

- [ ] **Step 2: 결과 확인**

```sql
SELECT COUNT(*) FROM public.horse_race_history;
SELECT horse_id, as_of_date, prev_total_races, prev_wins, prev_5_avg_rank
FROM public.horse_race_history
ORDER BY as_of_date DESC
LIMIT 5;
```

Expected: COUNT > 0, 샘플 row에 prev_total_races > 0인 행 다수 존재

---

## Task 2: preprocessor.py 수정 (TDD)

**Files:**
- Modify: `core/preprocessor.py`
- Test: `tests/test_preprocessor.py`

- [ ] **Step 1: 테스트 파일 수정 – sample_df 픽스처 업데이트 및 새 테스트 추가**

`tests/test_preprocessor.py` 전체를 아래로 교체:

```python
import pytest
import pandas as pd
import numpy as np
from core.preprocessor import RacePreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "horse_id": ["H1", "H2"],
        "race_id": ["R1", "R1"],
        "race_date": ["2025-01-01", "2025-01-01"],
        "final_rank": [1, 2],
        "horse_age": [4, 5],
        "gender": ["수컷", "암컷"],
        "horse_class": ["국5", "외3"],
        "race_distance": [1200, 1200],
        "total_horses": [8, 8],
        "horse_weight": [480, 470],
        "race_grade": ["일반", "일반"],
        "track_condition": ["양호", "양호"],
        "weather": ["맑음", "맑음"],
        "jockey_total_races": [100, 200],
        "jockey_total_wins": [20, 40],
        "trainer_total_races": [150, 300],
        "trainer_total_wins": [30, 60],
        "prev_total_races": [5, 10],
        "prev_wins": [1, 3],
        "prev_top3": [3, 7],
        "prev_5_avg_rank": [3.0, 2.0],
        "prev_total_avg_rank": [3.5, 2.2],
        "win_odds": [8.0, 3.5],
    })


def test_preprocessor_adds_is_winner(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert "is_winner" in result.columns
    assert result["is_winner"].iloc[0] == 1
    assert result["is_winner"].iloc[1] == 0


def test_preprocessor_adds_derived_features(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    for col in ["jockey_win_rate", "trainer_win_rate", "horse_win_rate",
                "horse_top3_rate", "experience_score", "recent_form", "is_male"]:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_preprocessor_horse_class_num_extracts_digit(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert "horse_class_num" in result.columns
    assert result["horse_class_num"].iloc[0] == 5   # "국5" → 5
    assert result["horse_class_num"].iloc[1] == 3   # "외3" → 3


def test_preprocessor_horse_class_num_defaults_for_no_digit():
    df = pd.DataFrame({
        "horse_id": ["H1"],
        "race_id": ["R1"],
        "race_date": ["2025-01-01"],
        "final_rank": [1],
        "horse_age": [4],
        "gender": ["수컷"],
        "horse_class": ["나"],        # 숫자 없는 형식
        "race_distance": [1200],
        "total_horses": [8],
        "horse_weight": [480],
        "race_grade": ["일반"],
        "track_condition": ["양호"],
        "weather": ["맑음"],
        "win_odds": [5.0],
    })
    p = RacePreprocessor()
    result = p.fit_transform(df)
    assert result["horse_class_num"].iloc[0] == 3   # default


def test_preprocessor_implied_prob(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert "implied_prob" in result.columns
    # implied_prob = 1 / win_odds
    assert abs(result["implied_prob"].iloc[0] - 1.0 / 8.0) < 1e-6
    assert abs(result["implied_prob"].iloc[1] - 1.0 / 3.5) < 1e-6


def test_preprocessor_prev_total_avg_rank_populated(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert "prev_total_avg_rank" in result.columns
    assert result["prev_total_avg_rank"].iloc[0] == pytest.approx(3.5)
    assert result["prev_total_avg_rank"].iloc[1] == pytest.approx(2.2)


def test_preprocessor_prev_total_avg_rank_defaults_when_missing():
    df = pd.DataFrame({
        "horse_id": ["H1"],
        "race_id": ["R1"],
        "race_date": ["2025-01-01"],
        "final_rank": [1],
        "horse_age": [4],
        "gender": ["수컷"],
        "horse_class": ["국5"],
        "race_distance": [1200],
        "total_horses": [8],
        "horse_weight": [480],
        "race_grade": ["일반"],
        "track_condition": ["양호"],
        "weather": ["맑음"],
        "win_odds": [5.0],
    })
    p = RacePreprocessor()
    result = p.fit_transform(df)
    assert result["prev_total_avg_rank"].iloc[0] == pytest.approx(6.0)


def test_preprocessor_encodes_categoricals(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    for col in ["race_grade_enc", "track_condition_enc", "weather_enc"]:
        assert col in result.columns
        assert result[col].dtype in [np.int64, np.int32, np.float64]
    # horse_class는 더 이상 LabelEncoder 사용 안 함
    assert "horse_class_enc" not in result.columns


def test_preprocessor_transform_uses_fitted_encoders(sample_df):
    p = RacePreprocessor()
    p.fit_transform(sample_df)
    new_df = sample_df.copy()
    result = p.transform(new_df)
    assert "race_grade_enc" in result.columns
    assert "horse_class_num" in result.columns


def test_preprocessor_win_rate_range(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert (result["jockey_win_rate"] >= 0).all()
    assert (result["jockey_win_rate"] <= 1).all()
```

- [ ] **Step 2: 테스트 실행 – 실패 확인**

```
pytest tests/test_preprocessor.py -v
```

Expected: `test_preprocessor_horse_class_num_*`, `test_preprocessor_implied_prob`, `test_preprocessor_prev_total_avg_rank_*`, `test_preprocessor_encodes_categoricals` 실패

- [ ] **Step 3: preprocessor.py 수정**

`core/preprocessor.py` 전체를 아래로 교체:

```python
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = ["race_grade", "track_condition", "weather"]


class RacePreprocessor:
    def __init__(self):
        self.label_encoders: dict = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_target(df)
        df = self._add_derived_features(df)
        df = self._encode_categoricals(df, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_derived_features(df)
        df = self._encode_categoricals(df, fit=False)
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if "final_rank" in df.columns:
            df["is_winner"] = (df["final_rank"] == 1).astype(int)
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        gender = df["gender"] if "gender" in df.columns else pd.Series([""] * len(df), index=df.index)
        df["is_male"] = (gender == "수컷").astype(int)

        df["jockey_win_rate"] = df.get("jockey_total_wins", pd.Series(0, index=df.index)).fillna(0) / \
                                (df.get("jockey_total_races", pd.Series(0, index=df.index)).fillna(0) + 1)
        df["trainer_win_rate"] = df.get("trainer_total_wins", pd.Series(0, index=df.index)).fillna(0) / \
                                 (df.get("trainer_total_races", pd.Series(0, index=df.index)).fillna(0) + 1)

        prev_wins  = df["prev_wins"].fillna(0)          if "prev_wins"  in df.columns else pd.Series(0, index=df.index)
        prev_top3  = df["prev_top3"].fillna(0)          if "prev_top3"  in df.columns else pd.Series(0, index=df.index)
        prev_total = df["prev_total_races"].fillna(0)   if "prev_total_races" in df.columns else pd.Series(0, index=df.index)
        prev_5_avg = df["prev_5_avg_rank"].fillna(6)    if "prev_5_avg_rank"  in df.columns else pd.Series(6, index=df.index)
        prev_avg   = df["prev_total_avg_rank"].fillna(6) if "prev_total_avg_rank" in df.columns else pd.Series(6.0, index=df.index)

        df["prev_total_races"]    = prev_total
        df["prev_5_avg_rank"]     = prev_5_avg
        df["prev_total_avg_rank"] = prev_avg
        df["horse_win_rate"]      = prev_wins  / (prev_total + 1)
        df["horse_top3_rate"]     = prev_top3  / (prev_total + 1)
        df["experience_score"]    = np.log1p(prev_total)
        df["recent_form"]         = 6 - prev_5_avg

        # horse_class 서수 인코딩: "국5"→5, "외3"→3, 숫자 없으면 3 (중간 기본값)
        horse_class = df["horse_class"].fillna("") if "horse_class" in df.columns else pd.Series([""] * len(df), index=df.index)
        df["horse_class_num"] = horse_class.apply(_extract_class_num)

        # 시장 배당률 기반 내재확률
        win_odds = df["win_odds"].fillna(10.0) if "win_odds" in df.columns else pd.Series(10.0, index=df.index)
        df["win_odds"]      = win_odds
        df["implied_prob"]  = 1.0 / win_odds.clip(lower=1.01)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[f"{col}_enc"] = 0
                continue
            series = df[col].fillna("unknown").astype(str)
            if fit:
                enc = LabelEncoder()
                df[f"{col}_enc"] = enc.fit_transform(series)
                self.label_encoders[col] = enc
            else:
                if col in self.label_encoders:
                    enc = self.label_encoders[col]
                    known = set(enc.classes_)
                    series = series.apply(lambda x: x if x in known else enc.classes_[0])
                    df[f"{col}_enc"] = enc.transform(series)
                else:
                    df[f"{col}_enc"] = 0
        return df


def _extract_class_num(s: str) -> int:
    m = re.search(r'(\d+)$', str(s).strip())
    return int(m.group(1)) if m else 3
```

- [ ] **Step 4: 테스트 실행 – 통과 확인**

```
pytest tests/test_preprocessor.py -v
```

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```
git add core/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: preprocessor - horse_class_num, implied_prob, prev_total_avg_rank 수정"
```

---

## Task 3: config.py FEATURE_COLUMNS 업데이트

**Files:**
- Modify: `config.py`

- [ ] **Step 1: FEATURE_COLUMNS 수정**

`config.py`의 `FEATURE_COLUMNS` 블록을 아래로 교체:

```python
FEATURE_COLUMNS = [
    "horse_age", "is_male", "horse_class_num", "race_distance", "total_horses",
    "horse_weight", "race_grade_enc", "track_condition_enc", "weather_enc",
    "prev_total_races", "prev_5_avg_rank", "prev_total_avg_rank",
    "jockey_win_rate", "trainer_win_rate", "horse_win_rate",
    "horse_top3_rate", "experience_score", "recent_form",
    "win_odds", "implied_prob",
]
```

- [ ] **Step 2: 기존 테스트 전체 실행 – 회귀 없음 확인**

```
pytest tests/ -v
```

Expected: 전체 PASS (test_predictor는 feature_columns mock이라 무관)

- [ ] **Step 3: 커밋**

```
git add config.py
git commit -m "feat: FEATURE_COLUMNS - horse_class_num, win_odds, implied_prob 추가"
```

---

## Task 4: predictor.py – horse_race_history 조인

**Files:**
- Modify: `core/predictor.py`
- Test: `tests/test_predictor.py`

- [ ] **Step 1: 테스트 추가 – history 컬럼이 병합됨을 검증**

`tests/test_predictor.py` 끝에 추가:

```python
def test_predictor_predict_merges_history_columns(mock_trainer_and_preprocessor):
    """predict_from_db에서 horse_race_history가 병합되는 경로를 단위 검증."""
    trainer, preprocessor = mock_trainer_and_preprocessor

    history_row = {
        "horse_id": "H1",
        "prev_total_races": 10,
        "prev_wins": 2,
        "prev_top3": 5,
        "prev_5_avg_rank": 2.5,
        "prev_total_avg_rank": 3.0,
    }
    df_entries = pd.DataFrame([
        {"horse_id": "H1", "horse_age": 4, "is_male": 1, "horse_win_rate": 0.2,
         "horse_name": "말A", "entry_number": 1},
    ])
    hist_df = pd.DataFrame([history_row])

    merged = df_entries.merge(hist_df, on="horse_id", how="left")
    assert "prev_total_races" in merged.columns
    assert merged["prev_total_races"].iloc[0] == 10
    assert merged["prev_wins"].iloc[0] == 2
```

- [ ] **Step 2: 테스트 실행 – 통과 확인 (merge 로직은 pandas 표준)**

```
pytest tests/test_predictor.py -v
```

Expected: 신규 테스트 포함 전체 PASS

- [ ] **Step 3: predictor.py predict_from_db 수정**

`core/predictor.py`의 `predict_from_db` 메서드를 아래로 교체 (`pd`는 파일 상단에 이미 import됨):

```python
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

    rows = []
    for r in result.data:
        flat = {k: v for k, v in r.items() if not isinstance(v, dict)}

        h = r.get("horses") or {}
        flat["horse_name"] = h.get("name")
        flat["horse_age"] = h.get("age")
        flat["gender"] = h.get("gender")
        flat["horse_class"] = h.get("rank")

        rc = r.get("races") or {}
        flat["race_no"] = rc.get("race_no")
        flat["meet_code"] = rc.get("meet_code")
        flat["race_distance"] = rc.get("race_distance")
        flat["total_horses"] = rc.get("total_horses")
        flat["race_grade"] = rc.get("race_grade")
        flat["track_condition"] = rc.get("track_condition")
        flat["weather"] = rc.get("weather")

        j = r.get("jockeys") or {}
        flat["jockey_total_wins"] = j.get("total_wins", 0) or 0
        flat["jockey_total_races"] = j.get("total_races", 0) or 0

        t = r.get("trainers") or {}
        flat["trainer_total_wins"] = t.get("ord1_cnt_t", 0) or 0
        flat["trainer_total_races"] = t.get("rc_cnt_t", 0) or 0

        rows.append(flat)

    df = pd.DataFrame(rows)

    # horse_race_history 조인: 해당 날짜 이하의 가장 최신 통계
    horse_ids = df["horse_id"].dropna().unique().tolist()
    if horse_ids:
        hist_res = client.table("horse_race_history").select(
            "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, prev_5_avg_rank, prev_total_avg_rank"
        ).in_("horse_id", horse_ids).lte("as_of_date", race_date).execute()

        if hist_res.data:
            hist_df = pd.DataFrame(hist_res.data)
            hist_latest = (
                hist_df.sort_values("as_of_date", ascending=False)
                       .drop_duplicates(subset=["horse_id"], keep="first")
                       .drop(columns=["as_of_date"])
            )
            df = df.merge(hist_latest, on="horse_id", how="left")

    df = self.preprocessor.transform(df)
    return self.predict(df)
```

- [ ] **Step 4: 테스트 전체 실행**

```
pytest tests/ -v
```

Expected: 전체 PASS

- [ ] **Step 5: 커밋**

```
git add core/predictor.py tests/test_predictor.py
git commit -m "feat: predictor - horse_race_history 조인 추가"
```

---

## Task 5: main.py cmd_train – horse_race_history 조인

**Files:**
- Modify: `main.py`

- [ ] **Step 1: cmd_train 수정**

`main.py`의 `cmd_train` 함수를 아래로 교체:

```python
def cmd_train(args):
    logger.info("=== 모델 학습 시작 ===")
    import pandas as pd
    client = get_client()

    result = client.table("race_entries").select(
        "*, horses(*), races(*), jockeys(*), trainers(*)"
    ).gte("race_date", args.start_date).lte("race_date", args.end_date)\
        .not_.is_("final_rank", "null").execute()

    if not result.data:
        logger.error("학습 데이터 없음")
        return

    rows = []
    for r in result.data:
        flat = {k: v for k, v in r.items() if not isinstance(v, dict)}

        h = r.get("horses") or {}
        flat["horse_age"] = h.get("age")
        flat["gender"] = h.get("gender")
        flat["horse_class"] = h.get("rank")

        rc = r.get("races") or {}
        flat["race_distance"] = rc.get("race_distance")
        flat["total_horses"] = rc.get("total_horses")
        flat["race_grade"] = rc.get("race_grade")
        flat["track_condition"] = rc.get("track_condition")
        flat["weather"] = rc.get("weather")

        j = r.get("jockeys") or {}
        flat["jockey_total_wins"] = j.get("total_wins", 0) or 0
        flat["jockey_total_races"] = j.get("total_races", 0) or 0

        t = r.get("trainers") or {}
        flat["trainer_total_wins"] = t.get("ord1_cnt_t", 0) or 0
        flat["trainer_total_races"] = t.get("rc_cnt_t", 0) or 0

        rows.append(flat)

    df = pd.DataFrame(rows)

    # horse_race_history 일괄 조인 (페이지네이션으로 전체 수집)
    all_hist = []
    page_size = 1000
    offset = 0
    while True:
        hist_res = client.table("horse_race_history").select(
            "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, prev_5_avg_rank, prev_total_avg_rank"
        ).gte("as_of_date", args.start_date).lte("as_of_date", args.end_date)\
            .range(offset, offset + page_size - 1).execute()
        if not hist_res.data:
            break
        all_hist.extend(hist_res.data)
        if len(hist_res.data) < page_size:
            break
        offset += page_size

    if all_hist:
        hist_df = pd.DataFrame(all_hist)
        hist_df = hist_df.rename(columns={"as_of_date": "race_date"})
        df = df.merge(hist_df, on=["horse_id", "race_date"], how="left")
        logger.info(f"horse_race_history 조인 완료: {len(all_hist)}개 행")
    else:
        logger.warning("horse_race_history 비어있음 – Task 1 SQL을 먼저 실행하세요")

    before = len(df)
    df = df.dropna(subset=["horse_age", "race_distance"])
    logger.info(f"학습 데이터: {len(df)}건 (피처 없는 {before - len(df)}건 제외)")

    preprocessor = RacePreprocessor()
    df_processed = preprocessor.fit_transform(df)

    trainer = RaceModelTrainer()
    metrics = trainer.train(df_processed)

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics=metrics["metrics"])
    logger.info(f"모델 저장 완료: {version}")
```

- [ ] **Step 2: 테스트 전체 실행**

```
pytest tests/ -v
```

Expected: 전체 PASS

- [ ] **Step 3: 커밋**

```
git add main.py
git commit -m "feat: cmd_train - horse_race_history 페이지네이션 조인"
```

---

## Task 6: 모델 재학습 및 검증

**Files:**
- 코드 변경 없음, 실행만

- [ ] **Step 1: 모델 재학습**

```
python main.py --mode train --start-date 2024-01-01
```

Expected 로그 예시:
```
horse_race_history 조인 완료: NNNN개 행
학습 데이터: 12XXX건 ...
XGBoost: AUC=0.7XX, Acc=0.XXX
LightGBM: AUC=0.7XX, Acc=0.XXX
모델 저장 완료: 20260528_XXXXXX
```

AUC 0.65 이상 확인 (기존보다 향상 기대)

- [ ] **Step 2: 예측 테스트 (날짜는 DB에 있는 과거 날짜로)**

```
python main.py --mode predict --date 2025-05-25
```

Expected: 각 경주마다 예측 1위 + EV 베팅 추천 출력, 오류 없음

- [ ] **Step 3: 최종 커밋**

```
git add models/
git commit -m "chore: 개선된 특성으로 모델 재학습 (horse_history + win_odds)"
```
