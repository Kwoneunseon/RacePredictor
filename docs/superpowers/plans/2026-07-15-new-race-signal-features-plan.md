# 신규 경주 신호 피처 추가 (부담중량·구간기록·성적·러닝스타일) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 단승/연승 예측 정확도를 올리기 위해 부담중량(kg)·구간기록·말/기수/조교사 성적·코너별 통과순위(러닝스타일) 4가지 신규 신호를 3단계로 나누어 추가하고, 각 단계마다 백테스트로 개선 여부를 검증한다. 부수적으로 `scripts/backtest_win.py`의 인기마 기준선 픽이 데뷔마 동률로 왜곡되는 기존 버그도 고친다.

**Architecture:** 신규 API 2개(`API299/Race_Result_total`, `API37_1/sectionRecord_1`)를 위한 신규 수집기(`library/collector_race_detail.py`, `library/collector_section_records.py`)를 만들어 신규 테이블(`race_entry_detail`, `horse_section_records`)에 적재하고, `core/feature_builder.py`가 이를 조인해 `config.FEATURE_COLUMNS`에 새 피처로 등록한다. 기존 3개 트랙별 경주결과 API·수집기·`race_entries` 테이블은 건드리지 않는다.

**Tech Stack:** Python, pandas, requests, Supabase(supabase-py), 기존 `core/feature_builder.py`/`core/trainer.py`/`core/preprocessor.py`, `scripts/backtest_win.py`/`scripts/backtest_top3.py`

**Spec:** `docs/superpowers/specs/2026-07-15-new-race-signal-features-design.md`

**중요 배경 (구현 중 반드시 참고):**
- `rcBudam`/`budam`(기존 3개 API)은 부담중량이 아니라 부담구분 코드(1:마령,2:별정,3:핸디캡)다 — 절대 이 필드를 파싱하지 말 것. 실제 kg 값은 `API299/Race_Result_total`의 `wgBudam` 필드에서 가져온다.
- `config.FEATURE_COLUMNS`에 이미 `prev_total_races`(말의 과거 출전 횟수, `horse_race_history` 테이블 기반)가 있다 — 이게 사실상 "신마여부"이므로 별도 신마여부 피처는 만들지 않는다. 대신 `scripts/backtest_win.py`의 인기마 기준선 버그만 이 값을 활용해 고친다(Task 0).
- 각 단계는 반드시 백필→재학습→백테스트 검증까지 마친 뒤 다음 단계로 진행한다. 개선이 없으면 해당 단계 피처를 `config.FEATURE_COLUMNS`에서 빼고 원인을 기록한다(과거 `FILTER_MAX_HORSES` 과적합 사고 재발 방지).
- 트레이너(`core/trainer.py`)는 최종적으로 `[feature_cols].fillna(0)`을 적용한다. 0이 비현실적인 값인 신규 피처(부담중량, 구간기록 등)는 반드시 `feature_builder.py` 단계에서 미리 결측치를 채워서 트레이너가 0으로 채우는 일이 없게 한다.

---

## Task 0: 백테스트 인기마 기준선 버그 수정

**Files:**
- Modify: `scripts/_backtest_common.py`
- Modify: `scripts/backtest_win.py`
- Test: 없음 (이 프로젝트의 백테스트 스크립트 컨벤션상 유닛 테스트 없음, 실행 결과로 검증)

**배경:** `scripts/backtest_win.py`의 `best_pick()`은 `prev_win_odds.idxmin()`으로 "인기마"를 뽑는데, `prev_win_odds`는 데뷔마(과거 출전 이력 없음)의 경우 전체 중앙값으로 대체된 값이라, 데뷔마가 여러 마리인 경주에서는 동률 중 임의(행 순서)로 픽이 결정된다. `prev_total_races`(이미 계산되는 값)를 활용해 데뷔마를 인기마 후보에서 제외하도록 고친다.

- [ ] **Step 1: `scripts/_backtest_common.py`의 `run_predictions()`이 `prev_total_races`도 반환하도록 수정**

`scripts/_backtest_common.py`에서 아래 부분을 찾는다:

```python
    final_ranks   = df_feat["final_rank"].copy()
    race_ids      = df_feat["race_id"].copy() if "race_id" in df_feat.columns else pd.Series(range(len(df_feat)))
    race_dates    = df_feat["race_date"].copy() if "race_date" in df_feat.columns else pd.Series([""] * len(df_feat))
    prev_win_odds = df_feat["prev_win_odds"].copy() if "prev_win_odds" in df_feat.columns else pd.Series([0.0] * len(df_feat))
```

바로 아래에 한 줄 추가:

```python
    prev_total_races = df_feat["prev_total_races"].copy() if "prev_total_races" in df_feat.columns else pd.Series([0] * len(df_feat))
```

그리고 `result_df = pd.DataFrame({...})` 안에 `"prev_win_odds": prev_win_odds.values,` 다음 줄에 추가:

```python
        "prev_total_races": prev_total_races.values,
```

docstring의 반환 컬럼 목록에도 `prev_total_races`를 추가한다.

- [ ] **Step 2: import 오류 확인**

Run: `python -c "from scripts._backtest_common import run_predictions; print('ok')"`
Expected: `ok`

- [ ] **Step 3: `scripts/backtest_win.py`의 `best_pick()`이 데뷔마를 인기마 후보에서 제외하도록 수정**

기존:

```python
        def best_pick(grp):
            best = grp.loc[grp["win_prob"].idxmax()]
            favorite = grp.loc[grp["prev_win_odds"].idxmin()]
            return pd.Series({
                "race_date":       best["race_date"],
                "win_prob":        best["win_prob"],
                "is_win":          best["is_win_actual"],
                "n_horses":        best["n_horses"],
                "is_favorite_win": favorite["is_win_actual"],
            })
```

아래로 교체:

```python
        def best_pick(grp):
            best = grp.loc[grp["win_prob"].idxmax()]
            # 데뷔마(prev_total_races==0)는 prev_win_odds가 중앙값으로 대체된 값이라
            # 인기마 후보에서 제외한다 (동률 임의 픽 방지). 전원 데뷔마인 경주만 예외적으로 전체에서 픽.
            non_debut = grp[grp["prev_total_races"] > 0]
            favorite_pool = non_debut if len(non_debut) > 0 else grp
            favorite = favorite_pool.loc[favorite_pool["prev_win_odds"].idxmin()]
            return pd.Series({
                "race_date":       best["race_date"],
                "win_prob":        best["win_prob"],
                "is_win":          best["is_win_actual"],
                "n_horses":        best["n_horses"],
                "is_favorite_win": favorite["is_win_actual"],
            })
```

- [ ] **Step 4: 실행해서 정상 동작 확인 (수치 변화가 있을 수 있음 — 버그 수정이므로 정상)**

Run: `python scripts/backtest_win.py --start 2026-05-01 --end 2026-06-25`
Expected: 오류 없이 완주, "인기마(최저배당) 기준선" 수치가 이전 대비 달라질 수 있음(더 정확해진 것). 총 경주 수(408)는 동일해야 함.

- [ ] **Step 5: Commit**

```bash
git add scripts/_backtest_common.py scripts/backtest_win.py
git commit -m "fix: backtest_win 인기마 기준선이 데뷔마를 제외하도록 수정 (prev_total_races 활용)"
```

---

## Task 1: [1단계] 부담중량 — 라이브 API 검증

**Files:** 없음 (검증 스크립트만 실행, 코드 변경 없음)

**목적:** `API299/Race_Result_total`의 `wgBudam`이 실제로 kg 단위 부담중량인지, `meet`/`hrNo`/`rcDate`/`rcNo` 필드가 기존 `races`/`race_entries` 테이블의 `race_id`/`horse_id`와 정확히 매칭되는지 실제 값으로 확인한다. 여기서 확인한 값이 Task 2~3의 파싱 로직 기준이 된다.

- [ ] **Step 1: 최근 실제 경주일로 API299 호출해서 원본 응답 확인**

Run:
```bash
python -c "
import requests, json
from config import API_KEY

endpoint = 'https://apis.data.go.kr/B551015/API299/Race_Result_total'
params = {
    'serviceKey': API_KEY, 'pageNo': 1, 'numOfRows': 20,
    'rc_date': '20260712', 'meet': '1', '_type': 'json',
}
resp = requests.get(endpoint, params=params, timeout=30)
data = resp.json()
items = data['response']['body']['items']['item']
for it in items[:5]:
    print(it.get('hrName'), it.get('hrNo'), it.get('meet'), it.get('rcDate'), it.get('rcNo'), 'wgBudam=', it.get('wgBudam'), 'wgHr=', it.get('wgHr'))
"
```
Expected: `wgBudam` 값이 50~62 범위(실제 부담중량 kg)로 나와야 한다. `wgHr`(마체중, 400~550대)과 확실히 구분되는 값 범위인지 확인.

- [ ] **Step 2: `meet` 필드 값이 기존 `races.race_id`의 트랙 문자열과 일치하는지 확인**

Run:
```bash
python -c "
from Utils.database import get_client
client = get_client()
res = client.table('races').select('race_id').eq('race_date', '2026-07-12').limit(3).execute()
for r in res.data:
    print(r['race_id'])
"
```
Expected: `race_id`가 `2026-07-12_서울_1` 형태인지 확인하고, Step 1에서 나온 `meet` 필드 값(디코딩된 실제 문자열)이 이 `서울` 부분과 정확히 일치하는지 눈으로 비교한다. **불일치하면(예: API가 "서울" 대신 다른 표기를 쓰면) Task 2에서 `race_id` 조립 시 매핑 테이블을 추가해야 한다 — 이 경우 Task 2 Step 1을 진행하기 전에 정확한 매핑을 확정할 것.**

- [ ] **Step 3: `hrNo` 값이 기존 `race_entries.horse_id`와 동일 포맷인지 확인**

Run:
```bash
python -c "
from Utils.database import get_client
client = get_client()
res = client.table('race_entries').select('horse_id').eq('race_date', '2026-07-12').limit(3).execute()
for r in res.data:
    print(repr(r['horse_id']))
"
```
Expected: Step 1의 `hrNo` 값(예: `'0056052'`)과 동일한 zero-padded 문자열 포맷인지 확인.

- [ ] **Step 4: 발견 사항을 다음 Task 진행 전 확인**

이 3단계에서 불일치가 하나라도 발견되면, Task 2의 `race_id`/`horse_id` 조립 로직에 그 불일치를 반영해서 진행한다(예: `meet` 값 변환 매핑 추가). 전부 일치하면 Task 2를 그대로 진행한다.

---

## Task 2: [1단계] `race_entry_detail` 테이블 및 부담중량 수집기

**Files:**
- Create (DB migration, Supabase SQL 실행): `race_entry_detail` 테이블
- Create: `library/collector_race_detail.py`

- [ ] **Step 1: `race_entry_detail` 테이블 생성**

Supabase SQL 편집기 또는 `mcp__supabase__apply_migration`으로 실행:

```sql
CREATE TABLE IF NOT EXISTS race_entry_detail (
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    assigned_weight NUMERIC,
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (race_id, horse_id)
);
```

- [ ] **Step 2: 생성 확인**

Run:
```bash
python -c "
from Utils.database import get_client
client = get_client()
res = client.table('race_entry_detail').select('*').limit(1).execute()
print('ok', res.data)
"
```
Expected: 오류 없이 `ok []` 출력.

- [ ] **Step 3: `library/collector_race_detail.py` 작성**

Task 1에서 확인한 `meet` 필드 값을 그대로 써도 되는 것으로 가정하고 작성한다(불일치가 있었다면 `_build_race_id` 안에 매핑을 추가할 것):

```python
# -*- coding: utf-8 -*-
"""
경마시행당일 경주결과종합 수집기 — 부담중량·(2단계)성적·(3단계)통과순위
API ID : 15119524
Endpoint: https://apis.data.go.kr/B551015/API299/Race_Result_total
Request : meet(1/2/3), rc_date(YYYYMMDD), pageNo, numOfRows
"""
import requests
import time

from config import API_KEY
from Utils import parse_date, safe_int, safe_float, safe_str
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)

ENDPOINT = "https://apis.data.go.kr/B551015/API299/Race_Result_total"


def fetch_race_detail(meet: int, rc_date: str, page: int = 1, per_page: int = 200):
    """
    meet: 1(서울)/2(제주)/3(부산경남), rc_date: 'YYYYMMDD'
    Returns list of raw item dicts, or None on error.
    """
    params = {
        "serviceKey": API_KEY,
        "meet": meet,
        "rc_date": rc_date,
        "pageNo": page,
        "numOfRows": per_page,
        "_type": "json",
    }
    try:
        resp = requests.get(ENDPOINT, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning("경주결과종합 API 403 Forbidden — data.go.kr 활용신청 승인 확인 필요")
            return None
        if resp.status_code != 200:
            logger.error(f"경주결과종합 API 오류: {resp.status_code}")
            return None
        data = resp.json()
        body = data.get("response", {}).get("body", {})
        items = body.get("items") or {}
        raw = items.get("item", [])
        if isinstance(raw, dict):
            raw = [raw]
        return raw
    except Exception as exc:
        logger.error(f"경주결과종합 API 호출 오류: {exc}")
        return None


def _build_race_id(item: dict) -> str | None:
    race_date = parse_date(item.get("rcDate"))
    meet_str = safe_str(item.get("meet"))
    race_no = safe_int(item.get("rcNo"))
    if not race_date or not meet_str or not race_no:
        return None
    return f"{race_date}_{meet_str}_{race_no}"


def parse_item(item: dict) -> dict | None:
    """1단계: assigned_weight(wgBudam)만 파싱한다. 2·3단계에서 이 함수를 확장한다."""
    race_id = _build_race_id(item)
    horse_id = safe_str(item.get("hrNo"))
    if not race_id or not horse_id:
        return None
    return {
        "race_id": race_id,
        "horse_id": horse_id,
        "assigned_weight": safe_float(item.get("wgBudam"), default=None),
    }


def save_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    client = get_client()
    client.table("race_entry_detail").upsert(
        rows, on_conflict="race_id,horse_id"
    ).execute()
    return len(rows)


def collect_meet_date(meet: int, rc_date: str) -> int:
    """특정 경마장·경주일자의 전체 데이터 수집 (페이지네이션)."""
    all_rows = []
    page = 1
    while True:
        items = fetch_race_detail(meet, rc_date, page=page)
        if items is None:
            return 0
        if not items:
            break
        for it in items:
            parsed = parse_item(it)
            if parsed:
                all_rows.append(parsed)
        if len(items) < 200:
            break
        page += 1
        time.sleep(0.2)

    saved = save_rows(all_rows)
    logger.info(f"경주결과종합 저장: meet={meet} {rc_date} → {saved}건")
    return saved


def collect_range(start_date: str, end_date: str) -> int:
    """start_date, end_date: 'YYYY-MM-DD'. 기간 내 실제 경주가 있었던 (경마장,날짜) 조합만 수집."""
    client = get_client()
    res = client.table("races").select("race_date, meet_code")\
        .gte("race_date", start_date).lte("race_date", end_date).execute()
    combos = sorted({(r["race_date"], r["meet_code"]) for r in (res.data or [])})

    total = 0
    for race_date, meet_code in combos:
        rc_date = race_date.replace("-", "")
        cnt = collect_meet_date(meet_code, rc_date)
        total += cnt
        time.sleep(0.2)
    logger.info(f"경주결과종합 수집 완료: {start_date}~{end_date} → 총 {total}건")
    return total


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python -m library.collector_race_detail START END  (YYYY-MM-DD)")
        sys.exit(1)
    collect_range(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: 단일 날짜로 실제 수집 테스트**

Run: `python -m library.collector_race_detail 2026-07-12 2026-07-12`
Expected: 로그에 `경주결과종합 저장: meet=... 2026-07-12 → N건` 출력, 오류 없음.

- [ ] **Step 5: DB에 실제로 부담중량이 합리적 범위로 들어갔는지 확인**

Run:
```bash
python -c "
from Utils.database import get_client
client = get_client()
res = client.table('race_entry_detail').select('*').eq('race_id', '2026-07-12_서울_1').execute()
for r in res.data:
    print(r)
"
```
Expected: `assigned_weight` 값이 50~62 범위, NULL 아닌 행이 대부분.

- [ ] **Step 6: Commit**

```bash
git add library/collector_race_detail.py
git commit -m "feat: 경주결과종합 API 부담중량 수집기 추가 (race_entry_detail)"
```

---

## Task 3: [1단계] `feature_builder`에 부담중량 피처 추가

**Files:**
- Modify: `core/feature_builder.py`
- Modify: `config.py`
- Modify: `core/preprocessor.py`
- Test: `tests/test_feature_builder.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_feature_builder.py`에 추가:

```python
def test_fetch_assigned_weight_for_training_fills_median_when_missing():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "race_id": ["R1", "R2", "R3"],
        "horse_id": ["H1", "H2", "H3"],
    })
    detail_rows = [
        {"race_id": "R1", "horse_id": "H1", "assigned_weight": 54.0},
        {"race_id": "R2", "horse_id": "H2", "assigned_weight": 58.0},
        # R3/H3는 race_entry_detail에 없음 (수집 실패/미백필 케이스)
    ]
    result = fb._fetch_assigned_weight_for_training(df, detail_rows)
    assert result.loc[result["race_id"] == "R1", "assigned_weight"].iloc[0] == 54.0
    assert result.loc[result["race_id"] == "R2", "assigned_weight"].iloc[0] == 58.0
    # 결측은 중앙값(54, 58의 중앙값=56)으로 대체
    assert result.loc[result["race_id"] == "R3", "assigned_weight"].iloc[0] == 56.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_feature_builder.py::test_fetch_assigned_weight_for_training_fills_median_when_missing -v`
Expected: FAIL — `AttributeError: 'RaceFeatureBuilder' object has no attribute '_fetch_assigned_weight_for_training'`

- [ ] **Step 3: `core/feature_builder.py`에 학습용/예측용 메서드 추가**

`build_for_training` 안에 있는 다음 줄:
```python
        df = self._fetch_speed_features_for_training(df)
        return df
```
를 아래로 교체:
```python
        df = self._fetch_speed_features_for_training(df)
        df = self._fetch_assigned_weight_for_training(df)
        return df
```

`build_for_prediction` 안에 있는 다음 줄:
```python
        df = self._fetch_speed_features(df, race_date, client)
        df = self._fetch_distance_rates(df, race_date, client)
```
사이에 추가:
```python
        df = self._fetch_speed_features(df, race_date, client)
        df = self._fetch_assigned_weight(df, client)
        df = self._fetch_distance_rates(df, race_date, client)
```

파일 끝(`_fetch_speed_features_for_training` 메서드 뒤)에 두 메서드 추가:

```python
    def _fetch_assigned_weight(self, df: pd.DataFrame, client) -> pd.DataFrame:
        if "race_id" not in df.columns or "horse_id" not in df.columns:
            df["assigned_weight"] = 56.0
            return df
        race_ids = df["race_id"].dropna().unique().tolist()
        if not race_ids:
            df["assigned_weight"] = 56.0
            return df
        res = client.table("race_entry_detail")\
            .select("race_id, horse_id, assigned_weight")\
            .in_("race_id", race_ids).execute()
        return self._join_assigned_weight(df, res.data or [])

    def _fetch_assigned_weight_for_training(self, df: pd.DataFrame, detail_rows: list = None) -> pd.DataFrame:
        if detail_rows is None:
            if "race_id" not in df.columns:
                df["assigned_weight"] = 56.0
                return df
            client = get_client()
            race_ids = df["race_id"].dropna().unique().tolist()
            all_rows = []
            for i in range(0, len(race_ids), 500):
                res = client.table("race_entry_detail")\
                    .select("race_id, horse_id, assigned_weight")\
                    .in_("race_id", race_ids[i:i + 500]).execute()
                all_rows.extend(res.data or [])
            detail_rows = all_rows
        return self._join_assigned_weight(df, detail_rows)

    def _join_assigned_weight(self, df: pd.DataFrame, detail_rows: list) -> pd.DataFrame:
        if not detail_rows:
            df["assigned_weight"] = 56.0
            return df
        detail_df = pd.DataFrame(detail_rows)[["race_id", "horse_id", "assigned_weight"]]
        detail_df["assigned_weight"] = pd.to_numeric(detail_df["assigned_weight"], errors="coerce")
        df = df.merge(detail_df, on=["race_id", "horse_id"], how="left")
        median_weight = detail_df["assigned_weight"].median()
        if pd.isna(median_weight):
            median_weight = 56.0
        n_missing = df["assigned_weight"].isna().sum()
        df["assigned_weight"] = df["assigned_weight"].fillna(median_weight)
        logger.info(f"assigned_weight 조인 완료 (결측 {n_missing}건 → 중앙값 {median_weight:.1f}로 대체)")
        return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_feature_builder.py::test_fetch_assigned_weight_for_training_fills_median_when_missing -v`
Expected: PASS

- [ ] **Step 5: `config.py`의 `FEATURE_COLUMNS`에 추가**

`config.py`의 `FEATURE_COLUMNS` 리스트 안, `# 출발 게이트 번호 (안쪽 레인 이점)` 주석 앞에 추가:

```python
    # 부담중량
    "assigned_weight",
```

- [ ] **Step 6: `core/preprocessor.py`의 `transform()`에 방어 코드 추가**

`transform()` 안, 기존 `if "horse_win_rate_at_distance" not in df.columns:` 근처 방어 코드들 옆에 추가:

```python
        if "assigned_weight" not in df.columns:
            df["assigned_weight"] = 56.0
```

- [ ] **Step 7: 전체 feature_builder 테스트 통과 확인**

Run: `python -m pytest tests/test_feature_builder.py -v`
Expected: 모든 테스트 PASS

- [ ] **Step 8: Commit**

```bash
git add core/feature_builder.py config.py core/preprocessor.py tests/test_feature_builder.py
git commit -m "feat: assigned_weight(부담중량) 피처 추가"
```

---

## Task 4: [1단계] 과거 데이터 백필 + 재학습 + 백테스트 체크포인트

**Files:** 없음 (실행만)

- [ ] **Step 1: 최소 백테스트 검증 기간 + 학습 기간 백필**

Run: `python -m library.collector_race_detail 2025-01-01 2026-07-15`
Expected: 오류 없이 완주 (시간이 오래 걸릴 수 있음 — 날짜×경마장 조합 수만큼 API 호출). 진행 중 403이 뜨면 활용신청 승인 상태를 먼저 확인.

- [ ] **Step 2: 재학습**

Run: `python main.py --mode train --start-date 2024-01-01`
Expected: 오류 없이 완료, `assigned_weight`가 `feature_columns`에 포함됐는지 학습 로그로 확인.

- [ ] **Step 3: 백테스트로 개선 여부 확인 (1단계 전/후 비교)**

Run: `python scripts/backtest_win.py --start 2026-05-01 --end 2026-06-25`
Run: `python scripts/backtest_top3.py --start 2026-05-01 --end 2026-06-25`
Expected: 두 스크립트 모두 정상 완주. **Task 0 수정 후의 기준 수치(단승 Precision@1, 연승 적중률)와 비교해서 기록해둔다.** 나아지지 않으면 `config.FEATURE_COLUMNS`에서 `assigned_weight`를 빼고 원인을 메모(예: 결측 비율이 너무 높음, 데이터 품질 문제 등)한 뒤 2단계로 진행한다.

- [ ] **Step 4: 결과를 커밋 메시지 대신 진행 기록으로 남김**

이 태스크는 코드 변경이 없으므로 커밋 없음. 다음 사람(에이전트/사용자)이 볼 수 있도록 1단계 백테스트 수치를 대화/보고에 명시한다.

---

## Task 5: [2단계] 라이브 API 필드 재검증 (성적·구간기록)

**Files:** 없음 (검증만)

- [ ] **Step 1: 말/기수/조교사 성적 필드 중 어느 쪽이 실제 값인지 확인**

Run:
```bash
python -c "
import requests
from config import API_KEY

endpoint = 'https://apis.data.go.kr/B551015/API299/Race_Result_total'
params = {
    'serviceKey': API_KEY, 'pageNo': 1, 'numOfRows': 10,
    'rc_date': '20260712', 'meet': '1', '_type': 'json',
}
resp = requests.get(endpoint, params=params, timeout=30)
items = resp.json()['response']['body']['items']['item']
for it in items[:3]:
    print(it.get('hrName'),
          'hrOrd1CntT=', it.get('hrOrd1CntT'), 'ord1CntT=', it.get('ord1CntT'), 'ord1CntY=', it.get('ord1CntY'),
          'hrRcCntT=', it.get('hrRcCntT'), 'rcCntT=', it.get('rcCntT'), 'rcCntY=', it.get('rcCntY'))
"
```
Expected: `hrOrd1CntT`/`ord1CntT`, `hrRcCntT`/`rcCntT` 중 어느 필드가 서로 값이 다른지, 어느 쪽이 "통산"이고 어느 쪽이 "최근1년"인지 실제 숫자로 판단한다(통산 값 ≥ 최근1년 값이어야 함). **Task 6에서 어느 필드를 쓸지는 이 결과로 확정한다 — 기본 가정은 `ord1CntT`=통산 1착, `ord1CntY`=최근1년 1착, `rcCntT`=통산 출전, `rcCntY`=최근1년 출전, `hrOrd1CntT`/`hrRcCntT`는 사용 안 함(중복/용도 불명 필드)이다.**

- [ ] **Step 2: 구간기록 API가 거리별로 로우를 나누는지, 짧은 거리에서 1C/2C가 0인지 확인**

Run:
```bash
python -c "
import requests
from config import API_KEY

endpoint = 'http://apis.data.go.kr/B551015/API37_1/sectionRecord_1'
params = {
    'serviceKey': API_KEY, 'pageNo': 1, 'numOfRows': 5,
    'rc_date': '20260712', 'meet': '1', '_type': 'json',
}
resp = requests.get(endpoint, params=params, timeout=30)
items = resp.json()['response']['body']['items']['item']
for it in items:
    print(it.get('hrName'), 'dist=', it.get('rcDist'),
          '1c=', it.get('ARcTime_1c'), '2c=', it.get('ARcTime_2c'),
          '3c=', it.get('ARcTime_3c'), '4c=', it.get('ARcTime_4c'))
"
```
Expected: `rcDist`(거리)가 짧을수록(예: 1000m~1200m) `ARcTime_1c`/`ARcTime_2c`가 0으로 나오는 패턴 확인. **Task 7에서 "1C/2C 미해당" 거리 임계값을 이 결과로 정한다 — 0이 나오는 최대 거리를 기준으로 삼는다.**

- [ ] **Step 3: 발견 사항 기록**

이 태스크는 코드 변경 없음. Step 1·2에서 확정한 필드명/임계값을 Task 6·7 구현 시 그대로 반영한다.

---

## Task 6: [2단계] 구간기록·성적 테이블 및 수집기

**Files:**
- Create (DB migration): `horse_section_records` 테이블, `race_entry_detail` 컬럼 추가
- Create: `library/collector_section_records.py`
- Modify: `library/collector_race_detail.py`

- [ ] **Step 1: `horse_section_records` 테이블 생성 + `race_entry_detail` 컬럼 확장**

```sql
CREATE TABLE IF NOT EXISTS horse_section_records (
    horse_id TEXT NOT NULL,
    race_distance INTEGER NOT NULL,
    snapshot_race_date DATE NOT NULL,
    s1f_best NUMERIC, s1f_avg NUMERIC, s1f_worst NUMERIC,
    c1_best NUMERIC, c1_avg NUMERIC, c1_worst NUMERIC,
    c2_best NUMERIC, c2_avg NUMERIC, c2_worst NUMERIC,
    c3_best NUMERIC, c3_avg NUMERIC, c3_worst NUMERIC,
    c4_best NUMERIC, c4_avg NUMERIC, c4_worst NUMERIC,
    g3f_best NUMERIC, g3f_avg NUMERIC, g3f_worst NUMERIC,
    g1f_best NUMERIC, g1f_avg NUMERIC, g1f_worst NUMERIC,
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (horse_id, race_distance, snapshot_race_date)
);

ALTER TABLE race_entry_detail
    ADD COLUMN IF NOT EXISTS horse_career_wins INTEGER,
    ADD COLUMN IF NOT EXISTS horse_career_starts INTEGER,
    ADD COLUMN IF NOT EXISTS horse_recent1y_wins INTEGER,
    ADD COLUMN IF NOT EXISTS horse_recent1y_starts INTEGER,
    ADD COLUMN IF NOT EXISTS jockey_career_wins INTEGER,
    ADD COLUMN IF NOT EXISTS jockey_career_starts INTEGER,
    ADD COLUMN IF NOT EXISTS jockey_recent1y_wins INTEGER,
    ADD COLUMN IF NOT EXISTS jockey_recent1y_starts INTEGER,
    ADD COLUMN IF NOT EXISTS trainer_career_wins INTEGER,
    ADD COLUMN IF NOT EXISTS trainer_career_starts INTEGER,
    ADD COLUMN IF NOT EXISTS trainer_recent1y_wins INTEGER,
    ADD COLUMN IF NOT EXISTS trainer_recent1y_starts INTEGER;
```

- [ ] **Step 2: 생성 확인**

Run:
```bash
python -c "
from Utils.database import get_client
client = get_client()
print(client.table('horse_section_records').select('*').limit(1).execute().data)
print(client.table('race_entry_detail').select('horse_career_wins').limit(1).execute().data)
"
```
Expected: 오류 없이 빈 리스트/None 출력.

- [ ] **Step 3: `library/collector_race_detail.py`의 `parse_item()`에 성적 필드 추가**

Task 5에서 확정한 필드명으로(예시는 `ord1CntT`/`ord1CntY`/`rcCntT`/`rcCntY` 계열 가정, 실제와 다르면 이 필드명만 교체):

```python
def parse_item(item: dict) -> dict | None:
    """1·2단계: 부담중량 + 성적. 3단계에서 통과순위를 추가한다."""
    race_id = _build_race_id(item)
    horse_id = safe_str(item.get("hrNo"))
    if not race_id or not horse_id:
        return None
    return {
        "race_id": race_id,
        "horse_id": horse_id,
        "assigned_weight": safe_float(item.get("wgBudam"), default=None),
        "horse_career_wins": safe_int(item.get("ord1CntT"), default=None),
        "horse_career_starts": safe_int(item.get("rcCntT"), default=None),
        "horse_recent1y_wins": safe_int(item.get("ord1CntY"), default=None),
        "horse_recent1y_starts": safe_int(item.get("rcCntY"), default=None),
        "jockey_career_wins": safe_int(item.get("jkOrd1CntT"), default=None),
        "jockey_career_starts": safe_int(item.get("jkRcCntT"), default=None),
        "jockey_recent1y_wins": safe_int(item.get("jkOrd1CntY"), default=None),
        "jockey_recent1y_starts": safe_int(item.get("jkRcCntY"), default=None),
        "trainer_career_wins": safe_int(item.get("trOrd1CntT"), default=None),
        "trainer_career_starts": safe_int(item.get("trRcCntT"), default=None),
        "trainer_recent1y_wins": safe_int(item.get("trOrd1CntY"), default=None),
        "trainer_recent1y_starts": safe_int(item.get("trRcCntY"), default=None),
    }
```

(`safe_int`가 `default=None`을 지원하지 않으면 — `Utils/function.py`의 `safe_int(value, default=0)` 시그니처 확인 후, `None` 전달 시 그대로 `None`을 반환하도록 `Utils/function.py`의 `safe_int`/`safe_float`를 고치지 말고, 대신 이 파일 내에 로컬 헬퍼를 둔다):

```python
def _safe_int_or_none(value):
    try:
        return int(value) if value not in (None, "", "-") else None
    except (TypeError, ValueError):
        return None
```

그리고 위 `parse_item`에서 `safe_int(..., default=None)` 대신 `_safe_int_or_none(item.get(...))`으로 교체한다.

- [ ] **Step 4: `library/collector_section_records.py` 작성**

```python
# -*- coding: utf-8 -*-
"""
마필 구간별 경주기록 수집기
API ID : 15057859
Endpoint: http://apis.data.go.kr/B551015/API37_1/sectionRecord_1
Request : meet(1/2/3), rc_date(YYYYMMDD), pageNo, numOfRows
Response: 말별·거리별 구간(S1F,1C~4C,G3F,G1F) 최고(F)/평균(A)/최저(S) 기록
"""
import requests
import time

from config import API_KEY
from Utils import parse_date, safe_str
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)

ENDPOINT = "http://apis.data.go.kr/B551015/API37_1/sectionRecord_1"


def _safe_num_or_none(value):
    try:
        return float(value) if value not in (None, "", "-") else None
    except (TypeError, ValueError):
        return None


def fetch_section_records(meet: int, rc_date: str, page: int = 1, per_page: int = 200):
    params = {
        "serviceKey": API_KEY,
        "meet": meet,
        "rc_date": rc_date,
        "pageNo": page,
        "numOfRows": per_page,
        "_type": "json",
    }
    try:
        resp = requests.get(ENDPOINT, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning("구간기록 API 403 Forbidden — data.go.kr 활용신청 승인 확인 필요")
            return None
        if resp.status_code != 200:
            logger.error(f"구간기록 API 오류: {resp.status_code}")
            return None
        data = resp.json()
        body = data.get("response", {}).get("body", {})
        items = body.get("items") or {}
        raw = items.get("item", [])
        if isinstance(raw, dict):
            raw = [raw]
        return raw
    except Exception as exc:
        logger.error(f"구간기록 API 호출 오류: {exc}")
        return None


def parse_item(item: dict, snapshot_race_date: str) -> dict | None:
    horse_id = safe_str(item.get("hrNo"))
    race_distance = item.get("rcDist")
    if not horse_id or not race_distance:
        return None
    return {
        "horse_id": horse_id,
        "race_distance": int(race_distance),
        "snapshot_race_date": snapshot_race_date,
        "s1f_best": _safe_num_or_none(item.get("FRcTimeS1f")),
        "s1f_avg": _safe_num_or_none(item.get("ARcTimeS1f")),
        "s1f_worst": _safe_num_or_none(item.get("SRcTimeS1f")),
        "c1_best": _safe_num_or_none(item.get("FRcTime_1c")),
        "c1_avg": _safe_num_or_none(item.get("ARcTime_1c")),
        "c1_worst": _safe_num_or_none(item.get("SRcTime_1c")),
        "c2_best": _safe_num_or_none(item.get("FRcTime_2c")),
        "c2_avg": _safe_num_or_none(item.get("ARcTime_2c")),
        "c2_worst": _safe_num_or_none(item.get("SRcTime_2c")),
        "c3_best": _safe_num_or_none(item.get("FRcTime_3c")),
        "c3_avg": _safe_num_or_none(item.get("ARcTime_3c")),
        "c3_worst": _safe_num_or_none(item.get("SRcTime_3c")),
        "c4_best": _safe_num_or_none(item.get("FRcTime_4c")),
        "c4_avg": _safe_num_or_none(item.get("ARcTime_4c")),
        "c4_worst": _safe_num_or_none(item.get("SRcTime_4c")),
        "g3f_best": _safe_num_or_none(item.get("FRcTimeG3f")),
        "g3f_avg": _safe_num_or_none(item.get("ARcTimeG3f")),
        "g3f_worst": _safe_num_or_none(item.get("SRcTimeG3f")),
        "g1f_best": _safe_num_or_none(item.get("FRcTimeG1f")),
        "g1f_avg": _safe_num_or_none(item.get("ARcTimeG1f")),
        "g1f_worst": _safe_num_or_none(item.get("SRcTimeG1f")),
    }


def save_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    client = get_client()
    client.table("horse_section_records").upsert(
        rows, on_conflict="horse_id,race_distance,snapshot_race_date"
    ).execute()
    return len(rows)


def collect_meet_date(meet: int, rc_date: str) -> int:
    all_rows = []
    page = 1
    snapshot_date = parse_date(rc_date)
    while True:
        items = fetch_section_records(meet, rc_date, page=page)
        if items is None:
            return 0
        if not items:
            break
        for it in items:
            parsed = parse_item(it, snapshot_date)
            if parsed:
                all_rows.append(parsed)
        if len(items) < 200:
            break
        page += 1
        time.sleep(0.2)

    saved = save_rows(all_rows)
    logger.info(f"구간기록 저장: meet={meet} {rc_date} → {saved}건")
    return saved


def collect_range(start_date: str, end_date: str) -> int:
    client = get_client()
    res = client.table("races").select("race_date, meet_code")\
        .gte("race_date", start_date).lte("race_date", end_date).execute()
    combos = sorted({(r["race_date"], r["meet_code"]) for r in (res.data or [])})

    total = 0
    for race_date, meet_code in combos:
        rc_date = race_date.replace("-", "")
        cnt = collect_meet_date(meet_code, rc_date)
        total += cnt
        time.sleep(0.2)
    logger.info(f"구간기록 수집 완료: {start_date}~{end_date} → 총 {total}건")
    return total


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python -m library.collector_section_records START END  (YYYY-MM-DD)")
        sys.exit(1)
    collect_range(sys.argv[1], sys.argv[2])
```

- [ ] **Step 5: 단일 날짜 실제 수집 테스트 (두 수집기 모두)**

Run: `python -m library.collector_race_detail 2026-07-12 2026-07-12`
Run: `python -m library.collector_section_records 2026-07-12 2026-07-12`
Expected: 둘 다 오류 없이 저장 건수 로그 출력.

- [ ] **Step 6: Commit**

```bash
git add library/collector_race_detail.py library/collector_section_records.py
git commit -m "feat: 구간기록·성적 수집기 추가 (horse_section_records, race_entry_detail 확장)"
```

---

## Task 7: [2단계] `feature_builder`에 구간기록·성적 피처 추가

**Files:**
- Modify: `core/feature_builder.py`
- Modify: `config.py`
- Modify: `core/preprocessor.py`
- Test: `tests/test_feature_builder.py`

**참고:** Task 5 Step 2에서 확정한 "1C/2C 미해당 거리 임계값"을 아래 `_APPLICABLE_DISTANCE_THRESHOLD`에 반영한다. 기본값은 1300(1300m 미만은 1C/2C 미해당)으로 가정하고 작성하되, 실제 값과 다르면 이 상수만 수정한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_feature_builder.py`에 추가:

```python
def test_fetch_section_features_treats_zero_corner_as_not_applicable():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"],
        "race_date": ["2026-07-12"],
        "race_distance": [1000],  # 임계값 미만 → 1C/2C 미해당
    })
    section_rows = [{
        "horse_id": "H1", "race_distance": 1000, "snapshot_race_date": "2026-07-01",
        "c1_avg": 0.0, "c2_avg": 0.0, "c3_avg": 30.0, "c4_avg": 49.0,
        "s1f_avg": 13.9, "g3f_avg": 37.8, "g1f_avg": 13.5,
    }]
    result = fb._join_section_records(df, section_rows)
    # 1000m는 임계값 미만이므로 c1/c2 값(0)을 실제 기록으로 쓰지 않고 결측 취급 → 다른 말 데이터 없어 전체 c1/c2 중앙값 대체값이 들어감(여기선 표본 1개뿐이라 NaN 처리 후 fallback 상수)
    assert result.loc[0, "c3_avg"] == 30.0
    assert result.loc[0, "g1f_avg"] == 13.5


def test_horse_career_win_rate_uses_starts_as_denominator():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "race_id": ["R1"], "horse_id": ["H1"],
    })
    detail_rows = [{
        "race_id": "R1", "horse_id": "H1",
        "horse_career_wins": 3, "horse_career_starts": 10,
        "horse_recent1y_wins": 1, "horse_recent1y_starts": 4,
        "jockey_career_wins": 50, "jockey_career_starts": 500,
        "jockey_recent1y_wins": 5, "jockey_recent1y_starts": 40,
        "trainer_career_wins": 20, "trainer_career_starts": 300,
        "trainer_recent1y_wins": 2, "trainer_recent1y_starts": 30,
        "assigned_weight": 56.0,
    }]
    result = fb._join_career_stats(df, detail_rows)
    assert result.loc[0, "horse_career_win_rate"] == 0.3
    assert result.loc[0, "horse_recent1y_win_rate"] == 0.25
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_feature_builder.py::test_fetch_section_features_treats_zero_corner_as_not_applicable tests/test_feature_builder.py::test_horse_career_win_rate_uses_starts_as_denominator -v`
Expected: FAIL — 메서드 없음

- [ ] **Step 3: `core/feature_builder.py`에 구현 추가**

파일 상단 상수 추가 (import문들 아래):

```python
_SECTION_CORNER_APPLICABLE_DISTANCE = 1300  # 이 값 미만 거리는 1C/2C 데이터 없음 (Task 5 실측으로 확정)
```

파일 끝에 메서드 추가:

```python
    def _join_section_records(self, df: pd.DataFrame, section_rows: list) -> pd.DataFrame:
        section_cols = [
            "s1f_avg", "c1_avg", "c2_avg", "c3_avg", "c4_avg", "g3f_avg", "g1f_avg",
        ]
        if not section_rows or "horse_id" not in df.columns or "race_date" not in df.columns:
            for col in section_cols:
                df[col] = 0.0
            return df

        sec_df = pd.DataFrame(section_rows)
        sec_df["snapshot_race_date"] = pd.to_datetime(sec_df["snapshot_race_date"])
        # 짧은 거리는 1C/2C가 0으로 나옴 → 실제 기록이 아니라 "미해당" 이므로 NaN 처리
        short_mask = sec_df["race_distance"] < _SECTION_CORNER_APPLICABLE_DISTANCE
        sec_df.loc[short_mask, ["c1_avg", "c2_avg"]] = np.nan

        df = df.copy()
        df["_race_dt"] = pd.to_datetime(df["race_date"])
        merged_parts = []
        for hid, grp in df.groupby("horse_id"):
            sub = sec_df[sec_df["horse_id"] == hid].sort_values("snapshot_race_date")
            if sub.empty:
                part = grp.copy()
                for col in section_cols:
                    part[col] = np.nan
                merged_parts.append(part)
                continue
            part = pd.merge_asof(
                grp.sort_values("_race_dt"),
                sub.rename(columns={"snapshot_race_date": "_race_dt"}),
                on="_race_dt", direction="backward", allow_exact_matches=False,
            )
            merged_parts.append(part)
        df = pd.concat(merged_parts).sort_index().drop(columns=["_race_dt"])

        for col in section_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            if n_missing:
                logger.info(f"{col} 결측 {n_missing}건 → 중앙값 {median_val:.2f}로 대체")
        return df

    def _join_career_stats(self, df: pd.DataFrame, detail_rows: list) -> pd.DataFrame:
        rate_specs = [
            ("horse_career_win_rate", "horse_career_wins", "horse_career_starts"),
            ("horse_recent1y_win_rate", "horse_recent1y_wins", "horse_recent1y_starts"),
            ("jockey_career_win_rate", "jockey_career_wins", "jockey_career_starts"),
            ("jockey_recent1y_win_rate", "jockey_recent1y_wins", "jockey_recent1y_starts"),
            ("trainer_career_win_rate", "trainer_career_wins", "trainer_career_starts"),
            ("trainer_recent1y_win_rate", "trainer_recent1y_wins", "trainer_recent1y_starts"),
        ]
        sample_cols = [
            "horse_recent1y_starts", "jockey_recent1y_starts", "trainer_recent1y_starts",
        ]
        if not detail_rows or "race_id" not in df.columns or "horse_id" not in df.columns:
            for rate_col, _, _ in rate_specs:
                df[rate_col] = 0.0
            for col in sample_cols:
                df[col] = 0.0
            return df

        detail_df = pd.DataFrame(detail_rows)
        df = df.merge(
            detail_df[["race_id", "horse_id"] + [c for _, w, s in rate_specs for c in (w, s)]].drop_duplicates(),
            on=["race_id", "horse_id"], how="left",
        )
        for rate_col, win_col, start_col in rate_specs:
            starts = pd.to_numeric(df[start_col], errors="coerce")
            wins = pd.to_numeric(df[win_col], errors="coerce")
            df[rate_col] = (wins / starts.replace(0, np.nan)).fillna(0.0)
        for col in sample_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
```

`build_for_prediction`/`build_for_training`에서 위 메서드를 호출하도록 배선 (예측용, `_fetch_assigned_weight` 다음 줄):

```python
        df = self._fetch_assigned_weight(df, client)
        df = self._fetch_section_and_career_features(df, client)
        df = self._fetch_distance_rates(df, race_date, client)
```

학습용(`_fetch_assigned_weight_for_training` 다음 줄):

```python
        df = self._fetch_assigned_weight_for_training(df)
        df = self._fetch_section_and_career_features_for_training(df)
        return df
```

두 배선 헬퍼 추가:

```python
    def _fetch_section_and_career_features(self, df: pd.DataFrame, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist() if "horse_id" in df.columns else []
        race_ids = df["race_id"].dropna().unique().tolist() if "race_id" in df.columns else []
        sec_rows, detail_rows = [], []
        if horse_ids:
            res = client.table("horse_section_records").select("*").in_("horse_id", horse_ids).execute()
            sec_rows = res.data or []
        if race_ids:
            res = client.table("race_entry_detail").select(
                "race_id, horse_id, horse_career_wins, horse_career_starts, "
                "horse_recent1y_wins, horse_recent1y_starts, jockey_career_wins, jockey_career_starts, "
                "jockey_recent1y_wins, jockey_recent1y_starts, trainer_career_wins, trainer_career_starts, "
                "trainer_recent1y_wins, trainer_recent1y_starts"
            ).in_("race_id", race_ids).execute()
            detail_rows = res.data or []
        df = self._join_section_records(df, sec_rows)
        df = self._join_career_stats(df, detail_rows)
        return df

    def _fetch_section_and_career_features_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        client = get_client()
        return self._fetch_section_and_career_features(df, client)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_feature_builder.py -v`
Expected: 모든 테스트 PASS

- [ ] **Step 5: `config.py`의 `FEATURE_COLUMNS`에 추가**

`"assigned_weight",` 다음 줄에 추가:

```python
    # 구간기록
    "s1f_avg", "c1_avg", "c2_avg", "c3_avg", "c4_avg", "g3f_avg", "g1f_avg",
    # 통산·최근1년 성적
    "horse_career_win_rate", "horse_recent1y_win_rate",
    "jockey_career_win_rate", "jockey_recent1y_win_rate",
    "trainer_career_win_rate", "trainer_recent1y_win_rate",
    "horse_recent1y_starts", "jockey_recent1y_starts", "trainer_recent1y_starts",
```

- [ ] **Step 6: `core/preprocessor.py`의 `transform()`에 방어 코드 추가**

```python
        for col in ["s1f_avg", "c1_avg", "c2_avg", "c3_avg", "c4_avg", "g3f_avg", "g1f_avg",
                    "horse_career_win_rate", "horse_recent1y_win_rate",
                    "jockey_career_win_rate", "jockey_recent1y_win_rate",
                    "trainer_career_win_rate", "trainer_recent1y_win_rate",
                    "horse_recent1y_starts", "jockey_recent1y_starts", "trainer_recent1y_starts"]:
            if col not in df.columns:
                df[col] = 0.0
```

- [ ] **Step 7: Commit**

```bash
git add core/feature_builder.py config.py core/preprocessor.py tests/test_feature_builder.py
git commit -m "feat: 구간기록·통산/최근1년 성적 피처 추가"
```

---

## Task 8: [2단계] 과거 데이터 백필 + 재학습 + 백테스트 체크포인트

**Files:** 없음 (실행만)

- [ ] **Step 1: 구간기록·성적 과거분 백필**

Run: `python -m library.collector_race_detail 2025-01-01 2026-07-15` (Task 4에서 이미 실행했다면, 이번엔 성적 필드까지 포함해서 재실행 — upsert라 안전)
Run: `python -m library.collector_section_records 2025-01-01 2026-07-15`
Expected: 둘 다 오류 없이 완주.

- [ ] **Step 2: 재학습**

Run: `python main.py --mode train --start-date 2024-01-01`

- [ ] **Step 3: 백테스트로 1단계 대비 개선 여부 확인**

Run: `python scripts/backtest_win.py --start 2026-05-01 --end 2026-06-25`
Run: `python scripts/backtest_top3.py --start 2026-05-01 --end 2026-06-25`
Expected: 1단계 체크포인트(Task 4) 수치와 비교해서 개선 여부 기록. 개선 없으면 해당 피처군을 `config.FEATURE_COLUMNS`에서 빼고 원인 메모 후 3단계로 진행.

- [ ] **Step 4: 결과 기록**

코드 변경 없음, 커밋 없음. 수치를 대화/보고에 기록.

---

## Task 9: [3단계] 코너별 통과순위 → 러닝스타일 피처

**Files:**
- Modify: `library/collector_race_detail.py` (통과순위 필드 추가)
- Modify: `core/feature_builder.py`
- Modify: `config.py`
- Modify: `core/preprocessor.py`
- Test: `tests/test_feature_builder.py`

**피처 정의:** 말의 과거 경주들에서 "초반 코너 순위(1C) − 최종 순위(final_rank)"의 평균을 계산한다. 양수가 클수록 초반엔 뒤쳐졌다가 후반에 순위를 끌어올리는 클로저(추입마) 성향, 음수면 초반에 앞서다 처지는 성향으로 해석한다. 짧은 거리(1C 없음)는 `sj_3c_ord`(3코너 순위)로 대체한다.

- [ ] **Step 1: `race_entry_detail`에 통과순위 컬럼 추가**

```sql
ALTER TABLE race_entry_detail
    ADD COLUMN IF NOT EXISTS sj_1c_ord INTEGER,
    ADD COLUMN IF NOT EXISTS sj_2c_ord INTEGER,
    ADD COLUMN IF NOT EXISTS sj_3c_ord INTEGER,
    ADD COLUMN IF NOT EXISTS sj_4c_ord INTEGER;
```

- [ ] **Step 2: `library/collector_race_detail.py`의 `parse_item()`에 통과순위 필드 추가**

`parse_item()` 반환 dict에 추가:

```python
        "sj_1c_ord": _safe_int_or_none(item.get("sj_1cOrd")),
        "sj_2c_ord": _safe_int_or_none(item.get("sj_2cOrd")),
        "sj_3c_ord": _safe_int_or_none(item.get("sj_3cOrd")),
        "sj_4c_ord": _safe_int_or_none(item.get("sj_4cOrd")),
```

- [ ] **Step 3: 실패하는 테스트 작성**

`tests/test_feature_builder.py`에 추가:

```python
def test_running_style_score_positive_means_closer():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"], "race_date": ["2026-07-12"],
    })
    # 과거 2경주: 1코너 5위→최종 2위(3계단 추입), 1코너 6위→최종 1위(5계단 추입)
    past_rows = [
        {"horse_id": "H1", "race_date": "2026-06-01", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 2},
        {"horse_id": "H1", "race_date": "2026-06-15", "sj_1c_ord": 6, "sj_3c_ord": 5, "final_rank": 1},
    ]
    result = fb._compute_running_style(df, past_rows)
    assert result.loc[0, "running_style_score"] == pytest.approx((5 - 2 + 6 - 1) / 2)
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `python -m pytest tests/test_feature_builder.py::test_running_style_score_positive_means_closer -v`
Expected: FAIL

- [ ] **Step 5: `core/feature_builder.py`에 구현 추가**

```python
    def _compute_running_style(self, df: pd.DataFrame, past_rows: list) -> pd.DataFrame:
        if not past_rows or "horse_id" not in df.columns:
            df["running_style_score"] = 0.0
            return df
        past_df = pd.DataFrame(past_rows)
        # 1코너 순위 우선, 없으면(짧은 거리) 3코너 순위로 대체
        past_df["_early_ord"] = past_df["sj_1c_ord"].fillna(past_df["sj_3c_ord"])
        past_df = past_df.dropna(subset=["_early_ord", "final_rank"])
        past_df["_delta"] = pd.to_numeric(past_df["_early_ord"], errors="coerce") - \
            pd.to_numeric(past_df["final_rank"], errors="coerce")
        style_map = past_df.groupby("horse_id")["_delta"].mean().to_dict()
        df["running_style_score"] = df["horse_id"].map(style_map).fillna(0.0)
        return df

    def _fetch_running_style(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
        horse_ids = df["horse_id"].dropna().unique().tolist() if "horse_id" in df.columns else []
        if not horse_ids:
            df["running_style_score"] = 0.0
            return df
        res = client.table("race_entry_detail")\
            .select("horse_id, race_id, sj_1c_ord, sj_3c_ord, races(race_date), race_entries(final_rank)")\
            .in_("horse_id", horse_ids).lt("races.race_date", race_date).execute()
        rows = [{
            "horse_id": r["horse_id"],
            "sj_1c_ord": r.get("sj_1c_ord"),
            "sj_3c_ord": r.get("sj_3c_ord"),
            "final_rank": (r.get("race_entries") or {}).get("final_rank"),
        } for r in (res.data or [])]
        return self._compute_running_style(df, rows)

    def _fetch_running_style_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        if "horse_id" not in df.columns or "race_date" not in df.columns:
            df["running_style_score"] = 0.0
            return df
        client = get_client()
        horse_ids = df["horse_id"].dropna().unique().tolist()
        all_rows = []
        for i in range(0, len(horse_ids), 500):
            res = client.table("race_entry_detail")\
                .select("horse_id, race_id, sj_1c_ord, sj_3c_ord")\
                .in_("horse_id", horse_ids[i:i + 500]).execute()
            all_rows.extend(res.data or [])
        if not all_rows:
            df["running_style_score"] = 0.0
            return df
        detail_df = pd.DataFrame(all_rows)
        race_ids = detail_df["race_id"].unique().tolist()
        rank_rows = []
        for i in range(0, len(race_ids), 500):
            res = client.table("race_entries").select("race_id, horse_id, race_date, final_rank")\
                .in_("race_id", race_ids[i:i + 500]).execute()
            rank_rows.extend(res.data or [])
        rank_df = pd.DataFrame(rank_rows)
        merged = detail_df.merge(rank_df, on=["race_id", "horse_id"], how="inner")

        df = df.copy()
        df["_race_dt"] = pd.to_datetime(df["race_date"])
        merged["_race_dt"] = pd.to_datetime(merged["race_date"])
        parts = []
        for hid, grp in df.groupby("horse_id"):
            past = merged[(merged["horse_id"] == hid)]
            part_scores = []
            for _, row in grp.iterrows():
                past_before = past[past["_race_dt"] < row["_race_dt"]]
                part_scores.append(past_before.to_dict("records"))
            part = grp.copy()
            style_vals = []
            for recs in part_scores:
                if not recs:
                    style_vals.append(0.0)
                    continue
                sub_df = pd.DataFrame(recs)
                sub_df["_early_ord"] = sub_df["sj_1c_ord"].fillna(sub_df["sj_3c_ord"])
                sub_df = sub_df.dropna(subset=["_early_ord", "final_rank"])
                if sub_df.empty:
                    style_vals.append(0.0)
                    continue
                delta = pd.to_numeric(sub_df["_early_ord"], errors="coerce") - \
                    pd.to_numeric(sub_df["final_rank"], errors="coerce")
                style_vals.append(float(delta.mean()))
            part["running_style_score"] = style_vals
            parts.append(part)
        df = pd.concat(parts).sort_index().drop(columns=["_race_dt"])
        return df
```

배선 — 예측용(`_fetch_section_and_career_features` 다음 줄):

```python
        df = self._fetch_section_and_career_features(df, client)
        df = self._fetch_running_style(df, race_date, client)
```

학습용(`_fetch_section_and_career_features_for_training` 다음 줄):

```python
        df = self._fetch_section_and_career_features_for_training(df)
        df = self._fetch_running_style_for_training(df)
        return df
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `python -m pytest tests/test_feature_builder.py -v`
Expected: 모든 테스트 PASS

- [ ] **Step 7: `config.py`의 `FEATURE_COLUMNS`에 추가**

```python
    # 러닝스타일 (초반 코너순위 - 최종순위, 양수=추입형)
    "running_style_score",
```

- [ ] **Step 8: `core/preprocessor.py`의 `transform()`에 방어 코드 추가**

```python
        if "running_style_score" not in df.columns:
            df["running_style_score"] = 0.0
```

- [ ] **Step 9: Commit**

```bash
git add library/collector_race_detail.py core/feature_builder.py config.py core/preprocessor.py tests/test_feature_builder.py
git commit -m "feat: 코너별 통과순위 기반 러닝스타일 피처 추가"
```

---

## Task 10: [3단계] 과거 데이터 백필 + 재학습 + 최종 백테스트 검증 + 결과 보고

**Files:** 없음 (실행 및 보고만)

- [ ] **Step 1: 통과순위 포함 과거분 재백필**

Run: `python -m library.collector_race_detail 2025-01-01 2026-07-15`
Expected: `sj_1c_ord` 등 새 컬럼까지 채워짐 (upsert라 기존 행 갱신).

- [ ] **Step 2: 재학습**

Run: `python main.py --mode train --start-date 2024-01-01`

- [ ] **Step 3: 최종 백테스트 검증**

Run: `python scripts/backtest_win.py --start 2026-05-01 --end 2026-06-25`
Run: `python scripts/backtest_top3.py --start 2026-05-01 --end 2026-06-25`
Expected: 2단계 체크포인트(Task 8) 수치와 비교. 개선 없으면 `running_style_score`를 `config.FEATURE_COLUMNS`에서 빼고 원인 기록.

- [ ] **Step 4: 사용자에게 최종 결과 보고**

이 단계는 코드 변경이 아니라 보고다. 원래 기준선(단승 Precision@1 42.9%/인기마 22.3%, 연승 79.9%@0.65) 대비 1·2·3단계 각각의 전/후 수치를 표로 정리해서 사용자에게 전달한다. 개선되지 않은 단계가 있었다면 어느 단계인지, 왜인지도 함께 보고한다.

---

## Self-Review Notes

- **Spec coverage:** 부담중량(Task 1-4) ✓, 신마여부→백테스트 버그 수정으로 대체(Task 0) ✓, 구간기록(Task 5-8) ✓, 성적(Task 5-8) ✓, 러닝스타일(Task 9-10) ✓, 단계별 백테스트 체크포인트 ✓, imputation을 feature_builder에서 처리(트레이너 fillna(0) 전에) ✓, 라이브 필드 재검증 태스크(Task 1, 5) ✓, 기존 3개 API/수집기 비변경 ✓.
- **Placeholder scan:** 없음 — 모든 스텝에 실행 가능한 코드/명령 포함. Task 5에서 확정되는 필드명(`ord1CntT` 등)은 "기본 가정"으로 명시하고 실제와 다르면 해당 필드명만 교체하도록 안내 — 이는 라이브 API 값에 의존하는 것이지 미완성 설계가 아님.
- **Type/이름 일관성:** `_join_assigned_weight`/`_join_section_records`/`_join_career_stats`/`_compute_running_style` 등 헬퍼 메서드명이 테스트와 배선 코드에서 동일하게 사용됨을 확인. `race_entry_detail`/`horse_section_records` 컬럼명이 수집기·feature_builder·테스트 전체에서 일관됨을 확인.
