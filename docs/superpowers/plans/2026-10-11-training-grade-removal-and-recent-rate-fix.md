# training_grade_enc 제거 + recent_win_rate 학습경로 날짜정합 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 프로덕션 학습 파이프라인에서 ① 수집기가 없어 사실상 상수가 된 `training_grade_enc` 피처를 완전히 제거하고, ② `jockey_recent_win_rate`/`trainer_recent_win_rate`의 학습 경로가 예측 경로와 다른 기준(행 위치 vs 날짜)으로 "과거"를 잘라 생기는 train-serve skew를 수정한다.

**Architecture:** 두 수정 모두 `core/feature_builder.py`/`core/preprocessor.py`/`config.py`(프로덕션 학습·예측 공용 파이프라인)에 직접 적용한다. DB 접근이나 실행이 필요한 작업은 없다 — 전부 코드 수정 + 기존/신규 유닛테스트로 검증한다.

**Tech Stack:** Python 3.11, pandas, pytest. 신규 의존성 없음.

**참고 문서:** `docs/superpowers/specs/2026-10-11-training-grade-removal-and-recent-rate-fix-design.md`

---

### 사전 확인 사항 (구현자가 알아야 할 것)

- **이번 작업은 프로덕션 학습/예측 공용 파이프라인(`core/feature_builder.py`, `core/preprocessor.py`, `config.py`, `core/feature_importance_report.py`)을 직접 수정합니다.** 지금까지의 `scripts/diagnose_*.py` 진단 스크립트 작업보다 영향 범위가 큽니다 — `main.py`(학습/예측 CLI), `scripts/_backtest_common.py`(모든 백테스트 스크립트)가 이 파일들을 그대로 씁니다.
- **그래서 매 Task 끝에 `tests/` 특정 파일이 아니라 전체 테스트 스위트(`python -m pytest`, 인자 없이)를 돌려서 전부 통과하는지 반드시 확인합니다.** 특정 파일만 돌리고 넘어가면 다른 파일의 숨은 참조를 놓칠 수 있습니다.
- 저장소에는 이 계획과 무관한 사용자의 미커밋 변경이 있을 수 있습니다(지금 세션에서 사용자가 `deleted` 컬럼 연결 작업을 직접 했다고 했으니, `core/feature_builder.py`/`core/preprocessor.py` 자체에 이미 미커밋 변경이 있을 가능성이 높습니다). **작업 시작 전 `git status`/`git diff`로 현재 상태를 먼저 확인하고, 이 계획이 수정하는 부분과 겹치지 않는 기존 변경은 절대 되돌리거나 건드리지 마세요.** 이번 계획의 코드 스니펫(아래 "현재 코드")이 실제 파일과 다르면(예: `deleted` 관련 코드가 이미 추가돼 있어서), 그 차이를 무시하고 덮어쓰지 말고 실제 파일 내용을 보고 같은 의도로 수정을 적용하세요 — 안 되면 BLOCKED로 보고하세요.
- `git add`는 항상 이번에 수정하는 파일만 명시적으로 지정하세요(`git add -A`/`git add .` 금지).
- 테스트 실행: `python -m pytest` (리포지토리 루트에서, 전체 스위트). 특정 파일만 볼 때는 `python -m pytest tests/test_feature_builder.py tests/test_preprocessor.py tests/test_trainer.py -v`.

---

### Task 1: `training_grade_enc` 완전 제거

**Files:**
- Modify: `config.py`
- Modify: `core/feature_builder.py`
- Modify: `core/preprocessor.py`
- Modify: `core/feature_importance_report.py`
- Modify: `tests/test_feature_builder.py`
- Modify: `tests/test_trainer.py`

- [ ] **Step 1: `config.py`에서 FEATURE_COLUMNS 항목 제거**

다음을 찾아:
```python
    "jockey_recent_win_rate", "trainer_recent_win_rate",
    # 조교 상태
    "training_grade_enc",
    # 컨디션/체중
    "days_since_last_race", "horse_weight_diff",
```

다음으로 교체:
```python
    "jockey_recent_win_rate", "trainer_recent_win_rate",
    # 컨디션/체중
    "days_since_last_race", "horse_weight_diff",
```

- [ ] **Step 2: `core/feature_builder.py`에서 `_training_grade` 함수 삭제**

다음을 찾아:
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


class RaceFeatureBuilder:
```

다음으로 교체:
```python
class RaceFeatureBuilder:
```

- [ ] **Step 3: `build_for_prediction`에서 호출 줄 삭제**

다음을 찾아:
```python
        df = self._fetch_weights(df, race_date, client)
        df = self._fetch_recent_rates(df, race_date, client)
        df = self._fetch_training_grades(df, race_date, client)
        df = self._fetch_history(df, race_date, client)
```

다음으로 교체:
```python
        df = self._fetch_weights(df, race_date, client)
        df = self._fetch_recent_rates(df, race_date, client)
        df = self._fetch_history(df, race_date, client)
```

- [ ] **Step 4: `build_for_training`에서 호출 줄 삭제**

다음을 찾아:
```python
        df = self._fetch_history_for_training(df, end_date, client)
        df = self._fetch_training_grades_for_training(df, client)
        df = self._fetch_prev_odds_for_training(df)
```

다음으로 교체:
```python
        df = self._fetch_history_for_training(df, end_date, client)
        df = self._fetch_prev_odds_for_training(df)
```

- [ ] **Step 5: `_fetch_training_grades` 메서드 삭제**

다음을 찾아:
```python
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
```

다음으로 교체:
```python
    def _fetch_history(self, df: pd.DataFrame, race_date: str, client) -> pd.DataFrame:
```

- [ ] **Step 6: `_fetch_training_grades_for_training` 메서드 삭제**

다음을 찾아:
```python
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
```

다음으로 교체:
```python
    def _fetch_prev_odds_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
```

- [ ] **Step 7: `core/preprocessor.py`의 `fit_transform`에서 기본값 채우기 삭제**

다음을 찾아:
```python
        df = self._add_rolling_recent_rates(df)
        df = self._add_historical_performance_features(df)
        df = self._add_new_features(df)
        if "training_grade_enc" not in df.columns:
            df["training_grade_enc"] = 0.0
        df = self._add_race_relative_features(df)
```

다음으로 교체:
```python
        df = self._add_rolling_recent_rates(df)
        df = self._add_historical_performance_features(df)
        df = self._add_new_features(df)
        df = self._add_race_relative_features(df)
```

- [ ] **Step 8: `transform`에서도 기본값 채우기 삭제**

다음을 찾아:
```python
        if "jockey_recent_win_rate" not in df.columns:
            df["jockey_recent_win_rate"] = 0.0
        if "trainer_recent_win_rate" not in df.columns:
            df["trainer_recent_win_rate"] = 0.0
        if "training_grade_enc" not in df.columns:
            df["training_grade_enc"] = 0.0
        if "days_since_last_race" not in df.columns:
```

다음으로 교체:
```python
        if "jockey_recent_win_rate" not in df.columns:
            df["jockey_recent_win_rate"] = 0.0
        if "trainer_recent_win_rate" not in df.columns:
            df["trainer_recent_win_rate"] = 0.0
        if "days_since_last_race" not in df.columns:
```

- [ ] **Step 9: `core/feature_importance_report.py`에서 라벨 매핑 삭제**

다음을 찾아:
```python
    "jockey_recent_win_rate": "기수 최근 30일 승률",
    "trainer_recent_win_rate": "조교사 최근 30일 승률",
    "training_grade_enc": "조교 등급(인코딩)",
    "days_since_last_race": "직전 경주 이후 경과일",
```

다음으로 교체:
```python
    "jockey_recent_win_rate": "기수 최근 30일 승률",
    "trainer_recent_win_rate": "조교사 최근 30일 승률",
    "days_since_last_race": "직전 경주 이후 경과일",
```

- [ ] **Step 10: `tests/test_feature_builder.py`에서 `_training_grade` 단위테스트 4개 삭제**

다음을 찾아:
```python
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
# get_flatten_entries 단위 테스트 (DB 불필요)
```

다음으로 교체:
```python
# ──────────────────────────────────────────────
# get_flatten_entries 단위 테스트 (DB 불필요)
```

- [ ] **Step 11: 같은 파일에서 `training_grade_enc`를 참조하는 다른 테스트 2곳 수정**

다음을 찾아:
```python
    required = [
        "horse_id", "horse_name", "meet_code", "race_distance",
        "prev_win_odds", "current_win_odds", "training_grade_enc",
        "days_since_last_race", "prev_speed_figure", "speed_figure_trend",
        "assigned_weight",
    ]
```

다음으로 교체:
```python
    required = [
        "horse_id", "horse_name", "meet_code", "race_distance",
        "prev_win_odds", "current_win_odds",
        "days_since_last_race", "prev_speed_figure", "speed_figure_trend",
        "assigned_weight",
    ]
```

다음을 찾아:
```python
    assert result["prev_win_odds"].iloc[0] == 15.0
    # 스크래핑 실패 시 current_win_odds = prev_win_odds
    assert result["current_win_odds"].iloc[0] == 15.0
    assert result["training_grade_enc"].iloc[0] == 1.0
    assert result["days_since_last_race"].iloc[0] == 60
```

다음으로 교체:
```python
    assert result["prev_win_odds"].iloc[0] == 15.0
    # 스크래핑 실패 시 current_win_odds = prev_win_odds
    assert result["current_win_odds"].iloc[0] == 15.0
    assert result["days_since_last_race"].iloc[0] == 60
```

- [ ] **Step 12: `tests/test_trainer.py`의 합성 데이터에서 `training_grade_enc` 줄 삭제(3곳 동일)**

다음을 찾아(파일에 동일한 3곳이 있습니다 — 반드시 **셋 다** 지우세요):
```python
        "trainer_recent_win_rate": np.random.uniform(0, 0.3, n),
        "training_grade_enc": np.random.uniform(0, 3, n),
        "days_since_last_race": np.random.randint(7, 60, n),
```

다음으로 교체(3곳 전부):
```python
        "trainer_recent_win_rate": np.random.uniform(0, 0.3, n),
        "days_since_last_race": np.random.randint(7, 60, n),
```

- [ ] **Step 13: `training_grade_enc`가 코드에 더 이상 남아있지 않은지 확인**

Run: `grep -rn "training_grade" config.py core/ tests/`
Expected: 아무 결과도 나오지 않음(완전히 제거됨). 결과가 나오면 빠뜨린 참조이니 같은 방식으로 제거하세요.

- [ ] **Step 14: 전체 테스트 스위트 실행**

Run: `python -m pytest`
Expected: 전부 PASS(실패 0건). 1개라도 실패하면 원인을 찾아 고치세요(놓친 참조일 가능성이 높습니다) — 테스트를 건너뛰거나 억지로 통과시키지 마세요.

- [ ] **Step 15: 커밋**

```bash
git add config.py core/feature_builder.py core/preprocessor.py core/feature_importance_report.py tests/test_feature_builder.py tests/test_trainer.py
git commit -m "fix: training_grade_enc 피처 완전 제거 - 수집기 부재로 사실상 상수

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

### Task 2: `_add_rolling_recent_rates` 날짜 정합 수정 + 회귀 테스트

**Files:**
- Modify: `core/preprocessor.py`
- Modify: `tests/test_preprocessor.py`

- [ ] **Step 1: 실패하는(수정 전 코드로는 실패할) 회귀 테스트 추가**

`tests/test_preprocessor.py`에서 다음을 찾아(Task 1에서 이 파일은 안
건드렸으므로 그대로 있을 것입니다):

```python
    # R3(Feb 20): R1(Jan05)은 30일 밖, R2(Jan10)은 30일 밖 → 0.0
    r3 = result[result["race_id"] == "R3"].iloc[0]
    assert r3["jockey_recent_win_rate"] == pytest.approx(0.0)


def test_preprocessor_rolling_missing_columns(sample_df):
```

다음으로 교체:

```python
    # R3(Feb 20): R1(Jan05)은 30일 밖, R2(Jan10)은 30일 밖 → 0.0
    r3 = result[result["race_id"] == "R3"].iloc[0]
    assert r3["jockey_recent_win_rate"] == pytest.approx(0.0)


def test_preprocessor_rolling_recent_win_rate_excludes_same_day_races():
    """같은 날 여러 경주가 있어도, 그날의 다른 경주는 '과거'로 치지 않는다
    (날짜 기준 비교 - 행 위치 기준으로 자르면 같은 날 먼저 정렬된 경주의
    승패가 섞여 들어간다)."""
    df = pd.DataFrame({
        "horse_id": ["H1", "H2", "H3"],
        "race_id": ["R1", "R2", "R3"],
        "race_date": ["2025-01-05", "2025-01-05", "2025-01-05"],
        "final_rank": [1, 2, 1],
        "horse_age": [4, 4, 4],
        "gender": ["수컷", "수컷", "수컷"],
        "horse_class": ["국5", "국5", "국5"],
        "race_distance": [1200, 1200, 1200],
        "total_horses": [8, 8, 8],
        "horse_weight": [480, 480, 480],
        "race_grade": ["일반", "일반", "일반"],
        "track_condition": ["양호", "양호", "양호"],
        "weather": ["맑음", "맑음", "맑음"],
        "win_odds": [5.0, 5.0, 5.0],
        "jk_no": ["J1", "J1", "J1"],
        "trainer_id": ["T1", "T1", "T1"],
    })
    p = RacePreprocessor()
    result = p.fit_transform(df)
    # 전부 같은 날짜라 어느 경주도 같은 날의 다른 경주를 과거로 쓸 수 없다 -> 전부 0.0
    assert (result["jockey_recent_win_rate"] == 0.0).all()
    assert (result["trainer_recent_win_rate"] == 0.0).all()


def test_preprocessor_rolling_missing_columns(sample_df):
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

Run: `python -m pytest tests/test_preprocessor.py -v -k excludes_same_day`
Expected: FAIL(수정 전 코드는 `wins[:i]`로 같은 날 먼저 정렬된 R1의
승리를 R2/R3의 "과거"로 포함시켜 0.0이 아닌 값이 나옵니다 — 정확히
어떤 값이 나오는지는 pandas groupby 내부 정렬 순서에 따라 달라질 수
있으니 "0.0이 아니면 실패"라는 사실만 확인하면 됩니다)

- [ ] **Step 3: `_add_rolling_recent_rates` 수정**

`core/preprocessor.py`에서 다음을 찾아:

```python
    def _add_rolling_recent_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["race_date", "jk_no", "final_rank"]
        if not all(c in df.columns for c in needed):
            df["jockey_recent_win_rate"] = 0.0
            df["trainer_recent_win_rate"] = 0.0
            return df

        df = df.copy()
        df["_rdt"] = pd.to_datetime(df["race_date"])
        df["_is_win"] = (df["final_rank"] == 1).astype(float)

        def _rolling_rate(id_col):
            rates = pd.Series(0.0, index=df.index)
            for gid, grp in df.groupby(id_col):
                grp = grp.sort_values("_rdt")
                dates = grp["_rdt"].values
                wins = grp["_is_win"].values
                vals = []
                for i in range(len(wins)):
                    cutoff = dates[i] - np.timedelta64(30, "D")
                    past = wins[:i][dates[:i] >= cutoff]
                    vals.append(float(past.mean()) if len(past) > 0 else 0.0)
                rates[grp.index] = vals
            return rates

        df["jockey_recent_win_rate"] = _rolling_rate("jk_no")
        if "trainer_id" in df.columns:
            df["trainer_recent_win_rate"] = _rolling_rate("trainer_id")
        else:
            df["trainer_recent_win_rate"] = 0.0

        df = df.drop(columns=["_rdt", "_is_win"])
        return df
```

다음으로 교체(루프 내부만 변경 — 행 위치 슬라이싱 `wins[:i]`/`dates[:i]`
를 날짜 비교 `dates < current_date`로 교체):

```python
    def _add_rolling_recent_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = ["race_date", "jk_no", "final_rank"]
        if not all(c in df.columns for c in needed):
            df["jockey_recent_win_rate"] = 0.0
            df["trainer_recent_win_rate"] = 0.0
            return df

        df = df.copy()
        df["_rdt"] = pd.to_datetime(df["race_date"])
        df["_is_win"] = (df["final_rank"] == 1).astype(float)

        def _rolling_rate(id_col):
            rates = pd.Series(0.0, index=df.index)
            for gid, grp in df.groupby(id_col):
                grp = grp.sort_values("_rdt")
                dates = grp["_rdt"].values
                wins = grp["_is_win"].values
                vals = []
                for i in range(len(wins)):
                    current_date = dates[i]
                    cutoff = current_date - np.timedelta64(30, "D")
                    mask = (dates < current_date) & (dates >= cutoff)
                    past = wins[mask]
                    vals.append(float(past.mean()) if len(past) > 0 else 0.0)
                rates[grp.index] = vals
            return rates

        df["jockey_recent_win_rate"] = _rolling_rate("jk_no")
        if "trainer_id" in df.columns:
            df["trainer_recent_win_rate"] = _rolling_rate("trainer_id")
        else:
            df["trainer_recent_win_rate"] = 0.0

        df = df.drop(columns=["_rdt", "_is_win"])
        return df
```

- [ ] **Step 4: 새 테스트 통과 확인**

Run: `python -m pytest tests/test_preprocessor.py -v -k excludes_same_day`
Expected: PASS

- [ ] **Step 5: 기존 `test_preprocessor_rolling_recent_win_rate`가 여전히 통과하는지 확인**

Run: `python -m pytest tests/test_preprocessor.py -v -k rolling_recent_win_rate`
Expected: 둘 다(기존 테스트 + 신규 테스트) PASS — 기존 테스트는 날짜가
전부 다른 시나리오라 이번 수정으로 결과가 바뀌지 않아야 합니다.

- [ ] **Step 6: 전체 테스트 스위트 실행**

Run: `python -m pytest`
Expected: 전부 PASS(실패 0건).

- [ ] **Step 7: 커밋**

```bash
git add core/preprocessor.py tests/test_preprocessor.py
git commit -m "fix: jockey/trainer_recent_win_rate 학습경로가 같은 날 경주를 과거로 포함하던 문제 수정

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GGZ7zmtPoVo5Mu5FF5ouzG"
```

---

## Self-Review 체크리스트

- **스펙 커버리지**: ① training_grade_enc 제거(함수·호출부·기본값
  채우기·라벨 매핑·테스트 전부) → Task 1. ② `_add_rolling_recent_rates`
  날짜 정합 수정 + 회귀 테스트 → Task 2. 범위 외(재학습,
  `_fetch_recent_rates` 자체 수정, 신규 수집기) 전부 이 계획에 없음 —
  스펙과 일치.
- **플레이스홀더 스캔**: 모든 Step에 실제 코드/명령어 포함, "적절히
  처리" 식 모호한 지시 없음.
- **일관성**: Task 1 Step 13의 `grep` 확인이 Task 2에서 추가하는
  `training_grade_enc` 무관 코드(날짜 비교 로직)에 영향 없음을 확인 —
  둘은 서로 다른 함수/파일 영역을 건드리므로 충돌 없음. Task 2의 새
  테스트명(`test_preprocessor_rolling_recent_win_rate_excludes_same_day_races`)
  이 기존 테스트명(`test_preprocessor_rolling_recent_win_rate`)과
  겹치지 않음을 확인.
- **리스크 관리**: 두 Task 모두 끝에 "전체 테스트 스위트 실행" 단계를
  명시해 프로덕션 코드 변경의 파급 범위를 매번 확인하도록 함. 사전
  확인 사항에 `deleted` 컬럼 관련 기존 미커밋 변경을 건드리지 말라는
  경고를 포함함.
