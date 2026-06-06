# 코드베이스 정리 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 루트 파일 클러터 제거 + `predict_from_db()` 330줄 → `RaceFeatureBuilder` 분리로 코드 중복 해소 (기능 동작 변경 없음)

**Architecture:** `core/feature_builder.py`에 `RaceFeatureBuilder` 클래스 신규 생성. `predict_from_db()`와 `cmd_train()` 양쪽에서 동일한 FeatureBuilder를 재사용하여 피처 계산 중복 제거. `rebuild_horse_history.py`는 `core/history_rebuilder.py`로 이동해 sys.argv 조작 제거.

**Tech Stack:** Python 3.11, pandas, numpy, supabase-py, pytest, unittest.mock

---

## 파일 맵

| 상태 | 파일 |
|---|---|
| 신규 생성 | `core/feature_builder.py` |
| 신규 생성 | `tests/test_feature_builder.py` |
| 신규 생성 | `core/history_rebuilder.py` |
| 대폭 수정 | `core/predictor.py` |
| 대폭 수정 | `main.py` |
| 삭제 | `rebuild_horse_history.py` (→ core/history_rebuilder.py) |
| 이동 | `kakao_auth.py` → `Utils/kakao_auth.py` |
| 이동 | `backfill_log.txt` → `Logs/backfill_log.txt` |
| 삭제 | 15개 임시/디버그 파일 |
| 아카이브 | 6개 파일 → `_archive/` |

---

## Task 1: 루트 파일 정리

**Files:**
- Delete: `debug_compare.py`, `debug_data.py`, `debug_features.py`, `debug_leak.py`, `debug_result.py`
- Delete: `api_fields.txt`, `api_search.txt`, `api_search2.txt`, `compare_result.txt`, `leak_result.txt`, `pred_0524.txt`, `pred_0524_v2.txt`, `pred_result.txt`, `tmp_out.txt`, `morning_test.txt`
- Archive: `check_today_api.py`, `find_entry_api.py`, `find_entry_api2.py`, `analyze_accuracy.py`, `accuracy_report.csv`, `run_predict.py`
- Move: `kakao_auth.py` → `Utils/kakao_auth.py`
- Move: `backfill_log.txt` → `Logs/backfill_log.txt`

- [ ] **Step 1: 디버그/임시 파일 삭제**

```powershell
cd "F:\Project\경마 자동 투자 봇"
Remove-Item debug_compare.py, debug_data.py, debug_features.py, debug_leak.py, debug_result.py
Remove-Item api_fields.txt, api_search.txt, api_search2.txt
Remove-Item compare_result.txt, leak_result.txt, pred_0524.txt, pred_0524_v2.txt, pred_result.txt, tmp_out.txt, morning_test.txt
```

- [ ] **Step 2: 탐색 스크립트 아카이브**

```powershell
Move-Item check_today_api.py _archive\
Move-Item find_entry_api.py _archive\
Move-Item find_entry_api2.py _archive\
Move-Item analyze_accuracy.py _archive\
Move-Item accuracy_report.csv _archive\
Move-Item run_predict.py _archive\
```

- [ ] **Step 3: 유틸 파일 이동**

```powershell
Move-Item kakao_auth.py Utils\
Move-Item backfill_log.txt Logs\
```

- [ ] **Step 4: 임포트 이상 없는지 확인**

```powershell
python -c "from main import cmd_train, cmd_predict, cmd_status, cmd_update_results; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: 커밋**

```bash
git add -A
git commit -m "chore: 루트 파일 정리 - debug/임시 파일 삭제, 탐색 스크립트 아카이브"
```

---

## Task 2: core/history_rebuilder.py 생성 + sys.argv 핵 제거

**Files:**
- Create: `core/history_rebuilder.py`
- Modify: `main.py` (lines 298-332, `cmd_update_results`)
- Delete: `rebuild_horse_history.py`
- Test: `tests/test_history_rebuilder.py` (신규)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_history_rebuilder.py` 신규 생성:

```python
import pytest
import pandas as pd
import numpy as np
from core.history_rebuilder import compute_history


def test_compute_history_first_race_no_prior():
    """첫 경주는 직전 기록이 없으므로 prev_total_races=0"""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2024-01-01", "final_rank": 1, "horse_weight": 480},
    ])
    result = compute_history(df)
    row = result[result["horse_id"] == "H1"].iloc[0]
    assert row["prev_total_races"] == 0
    assert row["prev_wins"] == 0


def test_compute_history_second_race_uses_first():
    """두 번째 경주의 통계는 첫 경주 결과를 반영해야 한다"""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2024-01-01", "final_rank": 1, "horse_weight": 480},
        {"horse_id": "H1", "race_date": "2024-01-08", "final_rank": 2, "horse_weight": 482},
    ])
    result = compute_history(df)
    second = result[result["as_of_date"] == "2024-01-08"].iloc[0]
    assert second["prev_total_races"] == 1
    assert second["prev_wins"] == 1
    assert second["prev_top3"] == 1


def test_compute_history_multiple_horses():
    """말이 여러 마리일 때 각각 독립적으로 계산된다"""
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": "2024-01-01", "final_rank": 1, "horse_weight": 480},
        {"horse_id": "H2", "race_date": "2024-01-01", "final_rank": 3, "horse_weight": 500},
        {"horse_id": "H1", "race_date": "2024-01-08", "final_rank": 2, "horse_weight": 482},
    ])
    result = compute_history(df)
    assert len(result) == 3
    h2_rows = result[result["horse_id"] == "H2"]
    assert h2_rows.iloc[0]["prev_total_races"] == 0


def test_compute_history_prev5_avg_uses_last_five():
    """prev_5_avg_rank는 최근 5경주 평균이다"""
    ranks = [1, 2, 3, 4, 5, 1]
    df = pd.DataFrame([
        {"horse_id": "H1", "race_date": f"2024-0{i+1}-01", "final_rank": r, "horse_weight": 480}
        for i, r in enumerate(ranks)
    ])
    result = compute_history(df)
    last = result[result["horse_id"] == "H1"].sort_values("as_of_date").iloc[-1]
    # 직전 5경주: [2, 3, 4, 5, 1] → 평균 3.0
    assert abs(last["prev_5_avg_rank"] - 3.0) < 0.01
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_history_rebuilder.py -v
```

Expected: `ERROR` (모듈 없음)

- [ ] **Step 3: core/history_rebuilder.py 생성**

`rebuild_horse_history.py`의 내용을 가져오되, `main()` 대신 `rebuild(start_date, end_date)` 함수로 재정의한다.

```python
"""horse_race_history 테이블 재구성 모듈"""
import pandas as pd
import numpy as np
from Utils.database import get_client, upsert
from Utils.logger import get_logger

logger = get_logger(__name__)
PAGE_SIZE = 1000
BATCH_SIZE = 500


def load_all_entries(client) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        res = client.table("race_entries")\
            .select("horse_id, race_date, final_rank, horse_weight")\
            .gt("final_rank", 0)\
            .order("race_date").order("horse_id")\
            .range(offset, offset + PAGE_SIZE - 1)\
            .execute()
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        if offset % 5000 == 0:
            logger.info(f"  로드 중... {offset}건")
    return pd.DataFrame(rows)


def compute_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 (horse_id, race_date)에 대해 경주 직전 기준 누적 통계 계산.
    반환 컬럼: horse_id, as_of_date, prev_total_races, prev_wins,
               prev_top3, prev_5_avg_rank, prev_total_avg_rank
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["final_rank"] = pd.to_numeric(df["final_rank"], errors="coerce")
    df = df.dropna(subset=["horse_id", "race_date", "final_rank"])
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    results = []
    for horse_id, grp in df.groupby("horse_id", sort=False):
        grp = grp.sort_values("race_date").reset_index(drop=True)
        ranks = grp["final_rank"].values
        dates = grp["race_date"].values

        for i in range(len(grp)):
            prior = ranks[:i]
            if len(prior) == 0:
                prev_total, prev_wins, prev_top3 = 0, 0, 0
                prev_5_avg = float(ranks[i])
                prev_total_avg = float(ranks[i])
            else:
                prev_total = len(prior)
                prev_wins = int((prior == 1).sum())
                prev_top3 = int((prior <= 3).sum())
                last5 = prior[-5:] if len(prior) >= 5 else prior
                prev_5_avg = round(float(last5.mean()), 2)
                prev_total_avg = round(float(prior.mean()), 2)

            results.append({
                "horse_id": horse_id,
                "as_of_date": str(dates[i])[:10],
                "prev_total_races": prev_total,
                "prev_wins": prev_wins,
                "prev_top3": prev_top3,
                "prev_5_avg_rank": prev_5_avg,
                "prev_total_avg_rank": prev_total_avg,
            })

    return pd.DataFrame(results)


def rebuild(start_date: str = None, end_date: str = None) -> None:
    """
    horse_race_history 테이블 재구성.
    start_date, end_date: 특정 구간만 upsert할 때 지정 (None이면 전체).
    """
    client = get_client()

    logger.info("race_entries 전체 로드 중...")
    df_all = load_all_entries(client)
    logger.info(f"총 {len(df_all)}건 로드 완료")

    logger.info("누적 통계 계산 중...")
    hist_df = compute_history(df_all)
    logger.info(f"계산 완료: {len(hist_df)}행")

    if start_date:
        hist_df = hist_df[hist_df["as_of_date"] >= start_date]
    if end_date:
        hist_df = hist_df[hist_df["as_of_date"] <= end_date]
    logger.info(f"upsert 대상: {len(hist_df)}행 ({start_date or '전체'} ~ {end_date or '전체'})")

    rows = hist_df.to_dict("records")
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        ok = upsert("horse_race_history", batch, on_conflict="horse_id,as_of_date")
        if ok:
            total += len(batch)
            logger.info(f"  upsert {total}/{len(rows)}건 완료")
        else:
            logger.error(f"  배치 {i} upsert 실패")

    logger.info(f"horse_race_history 업데이트 완료: 총 {total}건")


if __name__ == "__main__":
    import sys
    rebuild(
        start_date=sys.argv[1] if len(sys.argv) > 1 else None,
        end_date=sys.argv[2] if len(sys.argv) > 2 else None,
    )
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/test_history_rebuilder.py -v
```

Expected: `4 passed`

- [ ] **Step 5: main.py — cmd_update_results() sys.argv 핵 제거**

`main.py`의 `cmd_update_results()` 함수 전체를 다음으로 교체:

```python
def cmd_update_results(args):
    from library.scraper_score import update_all_meets
    from core.history_rebuilder import rebuild

    date = args.date
    logger.info(f"=== 경주 결과 업데이트 시작: {date} ===")

    counts = update_all_meets(date)
    total = sum(counts.values())
    print(f"\n[1/2] win_odds 업데이트: {total}경주")
    for name, cnt in counts.items():
        print(f"  {name}: {cnt}경주")

    if total == 0:
        print(f"  {date} 경주 데이터 없음 - 경마일이 아닐 수 있음")
        return

    print(f"\n[2/2] horse_race_history 재구성 ({date} ~ {date})")
    try:
        rebuild(start_date=date, end_date=date)
        print("  완료")
    except Exception as e:
        logger.error(f"  horse_race_history 재구성 실패: {e}")

    logger.info(f"=== 경주 결과 업데이트 완료: {date} ===")
```

- [ ] **Step 6: rebuild_horse_history.py 루트 파일 삭제**

```powershell
Remove-Item rebuild_horse_history.py
```

- [ ] **Step 7: 임포트 확인**

```bash
python -c "from core.history_rebuilder import compute_history, rebuild; print('OK')"
python -c "from main import cmd_update_results; print('OK')"
```

Expected: `OK` / `OK`

- [ ] **Step 8: 테스트 전체 실행**

```bash
pytest tests/ -v --tb=short
```

Expected: 기존 테스트 전부 통과

- [ ] **Step 9: 커밋**

```bash
git add core/history_rebuilder.py tests/test_history_rebuilder.py main.py
git rm rebuild_horse_history.py
git commit -m "refactor: rebuild_horse_history → core/history_rebuilder, sys.argv 핵 제거"
```

---

## Task 3: core/feature_builder.py 생성

**Files:**
- Create: `core/feature_builder.py`
- Create: `tests/test_feature_builder.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_feature_builder.py` 신규 생성:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────
# _training_grade 단위 테스트 (DB 불필요)
# ──────────────────────────────────────────────
def test_training_grade_양호():
    from core.feature_builder import _training_grade
    assert _training_grade("양호") == 2


def test_training_grade_불량():
    from core.feature_builder import _training_grade
    assert _training_grade("불량") == 0


def test_training_grade_빈문자():
    from core.feature_builder import _training_grade
    assert _training_grade("") == 1


def test_training_grade_일반():
    from core.feature_builder import _training_grade
    assert _training_grade("보통") == 1


# ──────────────────────────────────────────────
# _flatten_entries 단위 테스트 (DB 불필요)
# ──────────────────────────────────────────────
def make_raw_entry(horse_id="H1", race_date="2026-06-07"):
    return {
        "id": 1,
        "race_id": f"{race_date}_서울_1",
        "race_date": race_date,
        "horse_id": horse_id,
        "entry_number": 1,
        "horse_weight": 480,
        "final_rank": None,
        "jk_no": "J001",
        "trainer_id": "T001",
        "win_odds": 0,
        "finish_time": None,
        "horses": {"name": "테스트마", "age": 4, "gender": "수컷", "rank": "국5"},
        "races": {
            "race_no": 1, "meet_code": 1, "race_distance": 1200,
            "total_horses": 8, "race_grade": "일반",
            "track_condition": "양호", "weather": "맑음",
        },
        "jockeys": {"total_wins": 100, "total_races": 1000},
        "trainers": {"ord1_cnt_t": 50, "rc_cnt_t": 500},
    }


def test_flatten_entries_extracts_nested_fields():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = fb._flatten_entries([make_raw_entry()])
    assert df["horse_name"].iloc[0] == "테스트마"
    assert df["horse_age"].iloc[0] == 4
    assert df["meet_code"].iloc[0] == 1
    assert df["race_distance"].iloc[0] == 1200
    assert df["jockey_total_wins"].iloc[0] == 100
    assert df["trainer_total_wins"].iloc[0] == 50


def test_flatten_entries_multiple_rows():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = fb._flatten_entries([make_raw_entry("H1"), make_raw_entry("H2")])
    assert len(df) == 2
    assert set(df["horse_id"]) == {"H1", "H2"}


# ──────────────────────────────────────────────
# build_for_prediction 통합 테스트 (DB mock)
# ──────────────────────────────────────────────
def make_mock_client(entry_data=None):
    """race_entries 쿼리에만 데이터 반환, 나머지는 빈 리스트"""
    client = MagicMock()
    mock_chain = client.table.return_value.select.return_value

    # race_entries 첫 조회 (.eq("race_date", ...))
    mock_chain.eq.return_value.execute.return_value.data = entry_data or []

    # 이후 피처 조회들 (.in_(), .lt_(), .gt_() 등) → 빈 리스트
    def empty_execute(*args, **kwargs):
        m = MagicMock()
        m.data = []
        return m

    for attr in ["in_", "lt", "gt", "gte", "lte", "not_", "order", "range"]:
        getattr(mock_chain, attr).return_value = MagicMock()
    # 체인 끝의 execute → 빈 데이터
    mock_chain.in_.return_value.lt.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.gte.return_value.lt.return_value.execute.return_value.data = []
    mock_chain.gte.return_value.lt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.not_.is_.return_value.execute.return_value.data = []

    return client


def test_build_for_prediction_empty_db():
    """DB에 데이터 없으면 빈 DataFrame 반환"""
    with patch("core.feature_builder.get_client", return_value=make_mock_client([])):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_build_for_prediction_returns_required_columns():
    """race_entries가 있으면 필수 컬럼을 포함한 DataFrame 반환"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    with patch("core.feature_builder.get_client", return_value=client):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    required = [
        "horse_id", "horse_name", "meet_code", "race_distance",
        "prev_win_odds", "training_grade_enc", "days_since_last_race",
        "prev_speed_figure", "speed_figure_trend",
    ]
    for col in required:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_build_for_prediction_fallback_values():
    """피처 DB 조회 결과 없을 때 폴백값이 설정된다"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    with patch("core.feature_builder.get_client", return_value=client):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    assert result["prev_win_odds"].iloc[0] == 15.0
    assert result["training_grade_enc"].iloc[0] == 1.0
    assert result["days_since_last_race"].iloc[0] == 60
    assert result["prev_speed_figure"].iloc[0] == 0.0
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_feature_builder.py -v
```

Expected: `ERROR` (모듈 없음)

- [ ] **Step 3: core/feature_builder.py 구현**

```python
"""DB에서 피처를 수집·계산해 DataFrame으로 반환하는 클래스"""
import numpy as np
import pandas as pd
from Utils.logger import get_logger

logger = get_logger(__name__)


def _training_grade(remk_txt: str) -> int:
    """조교 평가 텍스트 → 수치 등급. 양호=2, 기록없음=1, 불량계열=0"""
    s = str(remk_txt).strip()
    if not s:
        return 1
    if "불량" in s:
        return 0
    if "양호" in s:
        return 2
    return 1


class RaceFeatureBuilder:
    """
    race_entries 기반 피처 수집 클래스.

    - build_for_prediction(race_date): 예측용. DB에서 race_entries 조회 후 피처 전부 붙임.
    - build_for_training(df, end_date): 학습용. 이미 조회된 df에 피처 조인.
    """

    # ──────────────────────────────────
    # Public API
    # ──────────────────────────────────

    def build_for_prediction(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
        from Utils.database import get_client
        client = get_client()

        result = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).eq("race_date", race_date).execute()

        if not result.data:
            logger.warning(f"{race_date} 출전 데이터 없음")
            return pd.DataFrame()

        df = self._flatten_entries(result.data)

        if meet_code is not None:
            df = df[df["meet_code"] == meet_code].reset_index(drop=True)
            if df.empty:
                return df

        df = self._fetch_weights(df, race_date, client)
        df = self._fetch_recent_rates(df, race_date, client)
        df = self._fetch_training_grades(df, race_date, client)
        df = self._fetch_history(df, race_date, client)
        df = self._fetch_last_race_features(df, race_date, client)
        df = self._fetch_prev_odds(df, race_date, client)
        df = self._fetch_speed_features(df, race_date, client)
        df = self._fetch_distance_rates(df, race_date, client)
        df = self._fetch_combo_features(df, race_date, client)
        return df

    def build_for_training(self, df: pd.DataFrame, end_date: str) -> pd.DataFrame:
        from Utils.database import get_client
        client = get_client()

        df = self._fetch_history_for_training(df, end_date, client)
        df = self._fetch_training_grades_for_training(df, client)
        df = self._fetch_prev_odds_for_training(df)
        df = self._fetch_speed_features_for_training(df)
        return df

    # ──────────────────────────────────
    # 공통 헬퍼
    # ──────────────────────────────────

    def _flatten_entries(self, data: list) -> pd.DataFrame:
        rows = []
        for r in data:
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
        return pd.DataFrame(rows)

    # ──────────────────────────────────
    # 예측용 피처 메서드
    # ──────────────────────────────────

    def _fetch_weights(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        if "horse_weight" not in df.columns or "entry_number" not in df.columns:
            return df
        zero_races = df[
            df["horse_weight"].isna() | (df["horse_weight"] == 0)
        ][["meet_code", "race_no"]].drop_duplicates()
        if not zero_races.empty:
            from library.scraper_weights import fetch_race_weights
            for _, row in zero_races.iterrows():
                mc, rn = int(row["meet_code"]), int(row["race_no"])
                weights = fetch_race_weights(mc, rn, race_date)
                for entry_no, wt in weights.items():
                    mask = (
                        (df["meet_code"] == mc)
                        & (df["race_no"] == rn)
                        & (df["entry_number"] == entry_no)
                    )
                    if mask.any():
                        df.loc[mask, "horse_weight"] = wt["horse_weight"]
        return df

    def _fetch_recent_rates(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        if "jk_no" not in df.columns:
            df["jockey_recent_win_rate"] = 0.0
            df["trainer_recent_win_rate"] = 0.0
            return df
        cutoff = (pd.to_datetime(race_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        res = client.table("race_entries")\
            .select("jk_no, trainer_id, final_rank")\
            .gte("race_date", cutoff).lt("race_date", race_date)\
            .not_.is_("final_rank", "null").execute()
        if res.data:
            h = pd.DataFrame(res.data)
            h["_is_win"] = (h["final_rank"] == 1).astype(int)
            jk_rates = h.groupby("jk_no")["_is_win"].mean().to_dict()
            df["jockey_recent_win_rate"] = df["jk_no"].map(jk_rates).fillna(0.0)
            if "trainer_id" in df.columns:
                tr_rates = h.groupby("trainer_id")["_is_win"].mean().to_dict()
                df["trainer_recent_win_rate"] = df["trainer_id"].map(tr_rates).fillna(0.0)
            else:
                df["trainer_recent_win_rate"] = 0.0
        else:
            df["jockey_recent_win_rate"] = 0.0
            df["trainer_recent_win_rate"] = 0.0
        return df

    def _fetch_training_grades(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        if "horse_name" not in df.columns:
            df["training_grade_enc"] = 1.0
            return df
        horse_names = df["horse_name"].dropna().unique().tolist()
        cutoff = (pd.to_datetime(race_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        res = client.table("horse_training_sessions")\
            .select("hrnm, remk_txt")\
            .in_("hrnm", horse_names)\
            .gte("trng_dt", cutoff).lt("trng_dt", race_date)\
            .execute()
        if res.data:
            grade_map: dict = {}
            for row in res.data:
                name = row["hrnm"]
                grade = _training_grade(row.get("remk_txt", ""))
                grade_map[name] = min(grade_map.get(name, 2), grade)
            df["training_grade_enc"] = df["horse_name"].apply(
                lambda n: float(grade_map.get(n, 1))
            )
        else:
            df["training_grade_enc"] = 1.0
        return df

    def _fetch_history(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            return df
        res = client.table("horse_race_history").select(
            "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, "
            "prev_5_avg_rank, prev_total_avg_rank"
        ).in_("horse_id", horse_ids).lt("as_of_date", race_date).execute()
        if res.data:
            hist_df = pd.DataFrame(res.data)
            hist_latest = (
                hist_df.sort_values("as_of_date", ascending=False)
                       .drop_duplicates(subset=["horse_id"], keep="first")
                       .drop(columns=["as_of_date"])
            )
            df = df.merge(hist_latest, on="horse_id", how="left")
        return df

    def _fetch_last_race_features(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            df["days_since_last_race"] = 60
            df["horse_weight_diff"] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, race_date, horse_weight")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).execute()
        if res.data:
            last_df = pd.DataFrame(res.data)
            last_df = (
                last_df.sort_values("race_date", ascending=False)
                       .drop_duplicates(subset=["horse_id"], keep="first")
            )
            race_dt = pd.to_datetime(race_date)
            last_df["_days"] = (race_dt - pd.to_datetime(last_df["race_date"])).dt.days
            days_map = last_df.set_index("horse_id")["_days"].to_dict()
            prev_wt_map = last_df.set_index("horse_id")["horse_weight"].to_dict()
            df["days_since_last_race"] = df["horse_id"].map(days_map).fillna(60)
            df["_prev_weight"] = df["horse_id"].map(prev_wt_map).fillna(0)
            zero_mask = df["horse_weight"].isna() | (df["horse_weight"] == 0)
            df.loc[zero_mask, "horse_weight"] = df.loc[zero_mask, "_prev_weight"]
            df["horse_weight_diff"] = df["horse_weight"].fillna(0) - df["_prev_weight"].fillna(0)
            df = df.drop(columns=["_prev_weight"])
        else:
            df["days_since_last_race"] = 60
            df["horse_weight_diff"] = 0.0
        return df

    def _fetch_prev_odds(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            df["prev_win_odds"] = 15.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, race_date, win_odds")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("win_odds", 0).execute()
        if res.data:
            prev_df = pd.DataFrame(res.data)
            prev_df = (
                prev_df.sort_values("race_date", ascending=False)
                       .drop_duplicates(subset=["horse_id"], keep="first")
            )
            odds_map = prev_df.set_index("horse_id")["win_odds"].to_dict()
            median_odds = prev_df["win_odds"].median()
            df["prev_win_odds"] = df["horse_id"].map(odds_map).fillna(median_odds)
        else:
            df["prev_win_odds"] = 15.0
        return df

    def _fetch_speed_features(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids:
            df["prev_speed_figure"] = 0.0
            df["speed_figure_trend"] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, race_date, finish_time, races(race_distance)")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).not_.is_("finish_time", "null").execute()
        if res.data:
            spd_rows = [{
                "horse_id": r["horse_id"],
                "race_date": r["race_date"],
                "finish_time": r["finish_time"],
                "race_distance": (r.get("races") or {}).get("race_distance"),
            } for r in res.data]
            spd_df = pd.DataFrame(spd_rows).dropna(subset=["finish_time", "race_distance"])
            spd_df = spd_df[spd_df["finish_time"] > 0]
            if not spd_df.empty:
                spd_df["_speed"] = spd_df["race_distance"].astype(float) / spd_df["finish_time"]
                spd_sorted = spd_df.sort_values(["horse_id", "race_date"])
                spd_latest = spd_sorted.drop_duplicates(subset=["horse_id"], keep="last")
                spd_map = spd_latest.set_index("horse_id")["_speed"].to_dict()
                df["prev_speed_figure"] = df["horse_id"].map(spd_map).fillna(0.0)
                trend_map = {}
                for hid, grp in spd_sorted.groupby("horse_id"):
                    vals = grp["_speed"].values[-3:]
                    trend_map[hid] = (
                        float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
                        if len(vals) >= 2 else 0.0
                    )
                df["speed_figure_trend"] = df["horse_id"].map(trend_map).fillna(0.0)
            else:
                df["prev_speed_figure"] = 0.0
                df["speed_figure_trend"] = 0.0
        else:
            df["prev_speed_figure"] = 0.0
            df["speed_figure_trend"] = 0.0
        return df

    def _fetch_distance_rates(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids or "race_distance" not in df.columns:
            df["horse_win_rate_at_distance"] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, final_rank, races(race_distance)")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).execute()
        if res.data:
            dist_rows = [{
                "horse_id": r["horse_id"],
                "final_rank": r["final_rank"],
                "race_distance": (r.get("races") or {}).get("race_distance"),
            } for r in res.data]
            dist_df = pd.DataFrame(dist_rows).dropna(subset=["race_distance"])
            dist_df["_is_win"] = (dist_df["final_rank"] == 1).astype(float)
            dist_stats = dist_df.groupby(["horse_id", "race_distance"]).agg(
                _wins=("_is_win", "sum"), _races=("_is_win", "count")
            ).reset_index()
            dist_stats["horse_win_rate_at_distance"] = (
                dist_stats["_wins"] / (dist_stats["_races"] + 1)
            )
            df = df.merge(
                dist_stats[["horse_id", "race_distance", "horse_win_rate_at_distance"]],
                on=["horse_id", "race_distance"], how="left",
            )
            df["horse_win_rate_at_distance"] = df["horse_win_rate_at_distance"].fillna(0.0)
        else:
            df["horse_win_rate_at_distance"] = 0.0
        return df

    def _fetch_combo_features(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        combo_cols = [
            "horse_track_win_rate", "jockey_horse_win_rate",
            "jockey_track_win_rate", "recent_rank_trend", "weight_trend",
        ]
        horse_ids = df["horse_id"].dropna().unique().tolist()
        if not horse_ids or "meet_code" not in df.columns:
            for col in combo_cols:
                df[col] = 0.0
            return df
        res = client.table("race_entries")\
            .select("horse_id, jk_no, final_rank, race_date, horse_weight, races(meet_code)")\
            .in_("horse_id", horse_ids).lt("race_date", race_date)\
            .gt("final_rank", 0).execute()
        if not res.data:
            for col in combo_cols:
                df[col] = 0.0
            return df

        hist = pd.DataFrame([{
            "horse_id": r["horse_id"],
            "jk_no": r.get("jk_no"),
            "final_rank": r["final_rank"],
            "race_date": r["race_date"],
            "horse_weight": r.get("horse_weight") or 0,
            "meet_code": (r.get("races") or {}).get("meet_code"),
        } for r in res.data])
        hist["_is_win"] = (hist["final_rank"] == 1).astype(float)

        # horse_track_win_rate
        ht = hist.groupby(["horse_id", "meet_code"]).agg(
            _w=("_is_win", "sum"), _r=("_is_win", "count")
        ).reset_index()
        ht["horse_track_win_rate"] = ht["_w"] / (ht["_r"] + 1)
        df = df.merge(ht[["horse_id", "meet_code", "horse_track_win_rate"]],
                      on=["horse_id", "meet_code"], how="left")
        df["horse_track_win_rate"] = df["horse_track_win_rate"].fillna(0.0)

        # jockey_horse_win_rate
        jh = hist.groupby(["horse_id", "jk_no"]).agg(
            _w=("_is_win", "sum"), _r=("_is_win", "count")
        ).reset_index()
        jh["jockey_horse_win_rate"] = jh["_w"] / (jh["_r"] + 1)
        if "jk_no" in df.columns:
            df = df.merge(jh[["horse_id", "jk_no", "jockey_horse_win_rate"]],
                          on=["horse_id", "jk_no"], how="left")
        if "jockey_horse_win_rate" not in df.columns:
            df["jockey_horse_win_rate"] = 0.0
        df["jockey_horse_win_rate"] = df["jockey_horse_win_rate"].fillna(0.0)

        # jockey_track_win_rate
        jt = hist.groupby(["jk_no", "meet_code"]).agg(
            _w=("_is_win", "sum"), _r=("_is_win", "count")
        ).reset_index()
        jt["jockey_track_win_rate"] = jt["_w"] / (jt["_r"] + 1)
        if "jk_no" in df.columns:
            df = df.merge(jt[["jk_no", "meet_code", "jockey_track_win_rate"]],
                          on=["jk_no", "meet_code"], how="left")
        if "jockey_track_win_rate" not in df.columns:
            df["jockey_track_win_rate"] = 0.0
        df["jockey_track_win_rate"] = df["jockey_track_win_rate"].fillna(0.0)

        # recent_rank_trend
        hist_s = hist.sort_values(["horse_id", "race_date"])
        trend_map = {}
        for hid, grp in hist_s.groupby("horse_id"):
            last3 = grp["final_rank"].values[-3:]
            trend_map[hid] = (
                float(np.polyfit(np.arange(len(last3)), last3, 1)[0])
                if len(last3) >= 2 else 0.0
            )
        df["recent_rank_trend"] = df["horse_id"].map(trend_map).fillna(0.0)

        # weight_trend
        wt_map = {}
        for hid, grp in hist_s.groupby("horse_id"):
            wts = grp["horse_weight"].values
            diffs = np.diff(wts.astype(float))
            last3d = diffs[-3:] if len(diffs) >= 1 else np.array([0.0])
            wt_map[hid] = float(last3d.mean())
        df["weight_trend"] = df["horse_id"].map(wt_map).fillna(0.0)

        return df

    # ──────────────────────────────────
    # 학습용 피처 메서드
    # ──────────────────────────────────

    def _fetch_history_for_training(self, df: pd.DataFrame, end_date: str, client) -> pd.DataFrame:
        all_hist = []
        page_size, offset = 1000, 0
        while True:
            res = client.table("horse_race_history").select(
                "horse_id, as_of_date, prev_total_races, prev_wins, prev_top3, "
                "prev_5_avg_rank, prev_total_avg_rank"
            ).lte("as_of_date", end_date)\
             .order("horse_id").order("as_of_date")\
             .range(offset, offset + page_size - 1).execute()
            if not res.data:
                break
            all_hist.extend(res.data)
            if len(res.data) < page_size:
                break
            offset += page_size

        if not all_hist:
            logger.warning("horse_race_history 비어있음 – rebuild 먼저 실행하세요")
            return df

        hist_df = pd.DataFrame(all_hist)
        hist_df["as_of_date"] = pd.to_datetime(hist_df["as_of_date"])
        df["_race_dt"] = pd.to_datetime(df["race_date"])
        merged = pd.merge_asof(
            df.sort_values("_race_dt").reset_index(drop=True),
            hist_df.sort_values("as_of_date").reset_index(drop=True)
                   .rename(columns={"as_of_date": "_race_dt"}),
            on="_race_dt", by="horse_id",
            direction="backward", allow_exact_matches=False,
        )
        df = merged.drop(columns=["_race_dt"])
        logger.info(f"horse_race_history 조인 완료: {len(all_hist)}개 행")
        return df

    def _fetch_training_grades_for_training(self, df: pd.DataFrame, client) -> pd.DataFrame:
        horse_names = df["horse_name"].dropna().unique().tolist() if "horse_name" in df.columns else []
        if not horse_names:
            df["training_grade_enc"] = 1.0
            return df
        tr_all_data = []
        for i in range(0, len(horse_names), 100):
            res = client.table("horse_training_sessions")\
                .select("hrnm, remk_txt, trng_dt")\
                .in_("hrnm", horse_names[i:i + 100]).execute()
            if res.data:
                tr_all_data.extend(res.data)

        if not tr_all_data:
            df["training_grade_enc"] = 1.0
            return df

        tr_df = pd.DataFrame(tr_all_data)
        tr_df["trng_dt"] = pd.to_datetime(tr_df["trng_dt"])
        tr_df["_grade"] = tr_df["remk_txt"].apply(_training_grade)
        df["_race_dt"] = pd.to_datetime(df["race_date"])

        grade_rows = []
        for _, row in df[["horse_name", "_race_dt"]].drop_duplicates().iterrows():
            name, race_dt = row["horse_name"], row["_race_dt"]
            cutoff = race_dt - pd.Timedelta(days=7)
            sub = tr_df[
                (tr_df["hrnm"] == name)
                & (tr_df["trng_dt"] >= cutoff)
                & (tr_df["trng_dt"] < race_dt)
            ]
            grade_rows.append({
                "horse_name": name,
                "_race_dt": race_dt,
                "training_grade_enc": float(int(sub["_grade"].min()) if len(sub) else 1),
            })

        grade_df = pd.DataFrame(grade_rows)
        df = df.merge(grade_df, on=["horse_name", "_race_dt"], how="left")
        df["training_grade_enc"] = df["training_grade_enc"].fillna(1.0)
        df = df.drop(columns=["_race_dt"])
        logger.info(f"조교 등급 조인 완료: {(df['training_grade_enc'] != 1).sum()}건 반영")
        return df

    def _fetch_prev_odds_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        if "win_odds" not in df.columns or "horse_id" not in df.columns:
            df["prev_win_odds"] = 15.0
            return df
        odds_df = df[["horse_id", "race_date", "win_odds"]].copy()
        odds_df["win_odds"] = pd.to_numeric(odds_df["win_odds"], errors="coerce")
        odds_df = odds_df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
        odds_df["prev_win_odds"] = odds_df.groupby("horse_id")["win_odds"].shift(1)
        df = df.merge(
            odds_df[["horse_id", "race_date", "prev_win_odds"]],
            on=["horse_id", "race_date"], how="left",
        )
        median_odds = odds_df["win_odds"].median()
        df["prev_win_odds"] = df["prev_win_odds"].fillna(median_odds)
        logger.info(f"prev_win_odds 조인 완료 (중앙값 폴백={median_odds:.1f})")
        return df

    def _fetch_speed_features_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        if "finish_time" not in df.columns or "race_distance" not in df.columns:
            df["prev_speed_figure"] = 0.0
            df["speed_figure_trend"] = 0.0
            return df
        spd_df = df[["horse_id", "race_date", "finish_time", "race_distance"]].copy()
        spd_df["finish_time"] = pd.to_numeric(spd_df["finish_time"], errors="coerce")
        spd_df["race_distance"] = pd.to_numeric(spd_df["race_distance"], errors="coerce")
        spd_df = spd_df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)
        spd_df["_speed"] = spd_df["race_distance"] / spd_df["finish_time"].replace(0, np.nan)
        spd_df["prev_speed_figure"] = spd_df.groupby("horse_id")["_speed"].shift(1)

        def _speed_slope(s):
            arr = s.values
            out = np.zeros(len(arr))
            for i in range(len(arr)):
                past = arr[max(0, i - 3):i]
                past = past[~np.isnan(past)]
                if len(past) >= 2:
                    out[i] = float(np.polyfit(np.arange(len(past)), past, 1)[0])
            return pd.Series(out, index=s.index)

        spd_df["speed_figure_trend"] = spd_df.groupby("horse_id")["_speed"].transform(_speed_slope)
        df = df.merge(
            spd_df[["horse_id", "race_date", "prev_speed_figure", "speed_figure_trend"]],
            on=["horse_id", "race_date"], how="left",
        )
        df["prev_speed_figure"] = df["prev_speed_figure"].fillna(0.0)
        df["speed_figure_trend"] = df["speed_figure_trend"].fillna(0.0)
        n_valid = (df["prev_speed_figure"] > 0).sum()
        logger.info(f"prev_speed_figure 계산 완료 (유효 {n_valid}건)")
        return df
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/test_feature_builder.py -v
```

Expected: `11 passed`

- [ ] **Step 5: 커밋**

```bash
git add core/feature_builder.py tests/test_feature_builder.py
git commit -m "feat: RaceFeatureBuilder 생성 - DB 피처 수집 로직 분리"
```

---

## Task 4: core/predictor.py — predict_from_db() 교체

**Files:**
- Modify: `core/predictor.py` (lines 1-6 import 추가, lines 81-409 `predict_from_db` 교체)
- `_training_grade` 함수는 `core/feature_builder.py`로 이동했으므로 predictor.py에서 제거

- [ ] **Step 1: 기존 테스트 그린 확인**

```bash
pytest tests/test_predictor.py -v
```

Expected: 전부 통과 (4 passed)

- [ ] **Step 2: predictor.py 상단 import 블록 교체**

기존:
```python
import pandas as pd
import numpy as np
from config import SOFTMAX_TEMPERATURE, LAMBDARANK_BLEND
from Utils.logger import get_logger
```

변경 후 (첫 6줄 전체):
```python
import pandas as pd
import numpy as np
from config import SOFTMAX_TEMPERATURE, LAMBDARANK_BLEND
from Utils.logger import get_logger
from core.feature_builder import RaceFeatureBuilder
```

- [ ] **Step 3: `_training_grade` 함수 삭제**

predictor.py에서 아래 함수 전체 삭제 (lines 24-33):

```python
def _training_grade(remk_txt: str) -> int:
    """조교 평가 텍스트 → 수치 등급. 양호=2, 기록없음=1, 불량계열=0"""
    s = str(remk_txt).strip()
    if not s:
        return 1
    if "불량" in s:
        return 0
    if "양호" in s:
        return 2
    return 1
```

- [ ] **Step 4: `predict_from_db()` 메서드 전체 교체**

기존 `predict_from_db()` (lines 81~409) 전체를 아래로 교체:

```python
def predict_from_db(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
    fb = RaceFeatureBuilder()
    df = fb.build_for_prediction(race_date, meet_code)
    if df.empty:
        return pd.DataFrame()
    df = self.preprocessor.transform(df)
    return self.predict(df)
```

- [ ] **Step 5: 기존 테스트 그린 확인**

```bash
pytest tests/test_predictor.py -v
```

Expected: 전부 통과 (4 passed)

- [ ] **Step 6: 전체 테스트 확인**

```bash
pytest tests/ -v --tb=short
```

Expected: 전부 통과

- [ ] **Step 7: 커밋**

```bash
git add core/predictor.py
git commit -m "refactor: predict_from_db() → RaceFeatureBuilder 위임 (330줄 → 6줄)"
```

---

## Task 5: main.py — cmd_train() FeatureBuilder 재사용으로 교체

**Files:**
- Modify: `main.py` (lines 4-11 import, lines 15-206 `cmd_train` 전체 교체)

- [ ] **Step 1: cmd_train의 race_entries 조회 부분을 private 함수로 분리**

`main.py`에서 `cmd_train()` 함수 **전체**를 아래 두 함수로 교체한다:

```python
def _fetch_race_entries(args) -> "pd.DataFrame":
    """race_entries + 관련 테이블 조회, 컬럼 평탄화 후 DataFrame 반환"""
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder

    client = get_client()
    all_entries = []
    page_size = 1000
    offset = 0
    while True:
        result = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).gte("race_date", args.start_date).lte("race_date", args.end_date)\
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

    logger.info(f"race_entries 수집: {len(all_entries)}건")
    fb = RaceFeatureBuilder()
    return fb._flatten_entries(all_entries)


def cmd_train(args):
    logger.info("=== 모델 학습 시작 ===")
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder

    df = _fetch_race_entries(args)
    if df.empty:
        logger.error("학습 데이터 없음")
        return

    fb = RaceFeatureBuilder()
    df = fb.build_for_training(df, args.end_date)

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

- [ ] **Step 2: main.py 상단 import에서 `_training_grade` 제거 확인**

predictor.py에서 `_training_grade`가 제거되었으므로, `main.py` 안에 `from core.predictor import _training_grade` 가 있다면 삭제한다.

```bash
grep -n "_training_grade" main.py
```

Expected: 출력 없음 (이미 없었거나, 있었다면 해당 줄 삭제)

- [ ] **Step 3: 임포트 정상 확인**

```bash
python -c "from main import cmd_train, cmd_predict, cmd_status, cmd_update_results; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: 전체 테스트 실행**

```bash
pytest tests/ -v --tb=short
```

Expected: 전부 통과

- [ ] **Step 5: 커밋**

```bash
git add main.py
git commit -m "refactor: cmd_train() ETL 중복 제거 - RaceFeatureBuilder.build_for_training() 재사용"
```

---

## 완료 체크리스트

- [ ] 루트에 `debug_*.py` 파일 없음
- [ ] 루트에 `*_result.txt`, `pred_*.txt` 등 임시 파일 없음
- [ ] `rebuild_horse_history.py` 루트에 없음
- [ ] `core/history_rebuilder.py` 존재
- [ ] `core/feature_builder.py` 존재
- [ ] `core/predictor.py:predict_from_db()` 10줄 이하
- [ ] `main.py:cmd_train()` 20줄 이하
- [ ] `main.py:cmd_update_results()` sys.argv 조작 없음
- [ ] `pytest tests/ -v` 전부 통과
