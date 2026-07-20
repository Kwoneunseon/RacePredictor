# 학습 파이프라인 버그 수정 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 학습 파이프라인에서 발견된 4가지 문제를 수정해 모델 예측 성능을 개선한다.

**Background:** 2026-06-07 기준 단승 0/4, 연승 1/4로 성능이 저조. 코드 분석 결과 아래 4가지 문제를 확인함.

**수정 파일:** `config.py`, `core/trainer.py`, `core/preprocessor.py` (신규 파일 없음)

---

## 문제 1: 튜닝-학습 불일치 (가장 심각)

### 현상
`tune()`은 `XGBClassifier`(objective=logloss)로 최적 하이퍼파라미터를 찾고,
`train()`은 그 파라미터를 `XGBRanker`(objective=rank:pairwise)에 적용한다.
분류기 최적 파라미터 ≠ 랭커 최적 파라미터이므로 튜닝이 실질적으로 효과 없음.

### 위치
- `core/trainer.py:95-156` — `tune()` 내부 `xgb_objective`, `lgb_objective`

### 수정 방법
`tune()` 내부의 `xgb_objective`와 `lgb_objective` 함수를 분류기 대신 랭커로 교체한다.

**현재 코드 (`core/trainer.py:104-111`):**
```python
m = XGBClassifier(
    **params, scale_pos_weight=pos_weight,
    random_state=MODEL_CONFIG["random_state"],
    eval_metric="logloss", verbosity=0,
)
m.fit(X_train, y_train)
return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
```

**수정 후:**
```python
def make_label(rank: int) -> int:
    if rank == 1: return 4
    if rank == 2: return 3
    if rank == 3: return 2
    return 1

# 랭커용 race-level 그룹 분할 (tune 함수 시작부에서 준비)
# df 대신 race-level split이 필요하므로 아래 준비 코드도 추가해야 함
race_dates = df_sorted.groupby("race_id")["race_date"].first().sort_values()
split_idx_r = int(len(race_dates) * (1 - MODEL_CONFIG["test_size"]))
train_races = set(race_dates.index[:split_idx_r])
test_races  = set(race_dates.index[split_idx_r:])
df_tr = df_sorted[df_sorted["race_id"].isin(train_races)].sort_values("race_id")
df_te = df_sorted[df_sorted["race_id"].isin(test_races)].sort_values("race_id")
X_tr = df_tr[feature_cols].fillna(0)
X_te = df_te[feature_cols].fillna(0)
y_tr = df_tr["final_rank"].apply(make_label)
tr_groups = df_tr.groupby("race_id", sort=False).size().values
qid_tr = np.repeat(np.arange(len(tr_groups)), tr_groups)

def xgb_objective(trial):
    params = { ... }  # 기존 파라미터 탐색 범위 유지
    m = XGBRanker(
        objective="rank:pairwise",
        **params,
        random_state=MODEL_CONFIG["random_state"],
        verbosity=0,
    )
    m.fit(X_tr, y_tr, qid=qid_tr)
    scores = m.predict(X_te)
    # NDCG@3으로 평가 (랭킹 품질 기준)
    return _ndcg_for_tune(df_te, scores)
```

`lgb_objective`도 동일하게 `LGBMRanker(objective="lambdarank")`로 교체.

평가 지표를 위한 헬퍼 함수 `_ndcg_for_tune(df_te, scores) -> float`를 추가한다 (기존 `_eval_ndcg`를 참고해 tune 내부에서 사용 가능한 형태로 작성).

### 주의
- `tune()` 함수 시그니처 변경 없음 (외부 호출부 `main.py:cmd_tune`에서 `df_processed` 전달 중인데, race_id와 final_rank가 있어야 하므로 현재대로 유지)
- `df_sorted`에 `race_id`가 없을 경우 기존처럼 classifier fallback 유지

---

## 문제 2: 계산하지만 모델에 입력되지 않는 피처들

### 현상
`preprocessor.py`에서 아래 피처들을 계산하지만 `config.py`의 `FEATURE_COLUMNS`에 없어서
`train()`과 `predict()`에서 모두 무시된다.

| 피처명 | 계산 위치 | 의미 |
|---|---|---|
| `recent_form` | `preprocessor.py:100` | `6 - prev_5_avg_rank` (최근 폼) |
| `experience_score` | `preprocessor.py:99` | `log1p(prev_total_races)` (경험치) |
| `track_condition_enc` | `preprocessor.py:276` | 마장 상태 인코딩 |
| `weather_enc` | `preprocessor.py:276` | 날씨 인코딩 |
| `is_male` | `preprocessor.py:81` | 성별 (수컷 여부) |
| `implied_prob_vs_mean` | `preprocessor.py:266` | 경주 내 배당 확률 대비 평균 비율 |
| `race_implied_prob_rank` | `preprocessor.py:265` | 경주 내 배당 확률 순위 |

### 위치
- `config.py:53-80` — `FEATURE_COLUMNS` 리스트

### 수정 방법
`config.py`의 `FEATURE_COLUMNS`에 아래를 추가:

```python
FEATURE_COLUMNS = [
    # ... 기존 피처들 유지 ...

    # 아래 항목들 추가
    "recent_form",             # 최근 5경주 평균순위 기반 폼 지표
    "experience_score",        # 누적 레이스 수 log 변환
    "track_condition_enc",     # 마장 상태 (양호/불량 등)
    "weather_enc",             # 날씨 인코딩
    "is_male",                 # 성별 (수컷=1)
    "implied_prob_vs_mean",    # 경주 내 배당 확률 / 평균 배당 확률
    "race_implied_prob_rank",  # 경주 내 배당 확률 순위 (1=가장 낮은 배당=인기마)
]
```

### 주의
- `transform()` (`preprocessor.py:32-62`)에서 `recent_form`, `experience_score`, `is_male`에 대한 기본값 fallback이 없음. 예측 시 이 컬럼들이 없으면 KeyError 발생 가능.
- `transform()` 내에 아래 fallback 추가 필요:
  ```python
  if "recent_form" not in df.columns:
      df["recent_form"] = 0.0
  if "experience_score" not in df.columns:
      df["experience_score"] = 0.0
  if "is_male" not in df.columns:
      df["is_male"] = 0
  if "implied_prob_vs_mean" not in df.columns:
      df["implied_prob_vs_mean"] = 1.0
  if "race_implied_prob_rank" not in df.columns:
      df["race_implied_prob_rank"] = 1
  ```

---

## 문제 3: Softmax temperature 과도하게 낮음

### 현상
`SOFTMAX_TEMPERATURE = 0.3` (`config.py:49`)으로 설정되어 있어 1위 예측마에 확률이 극단적으로 집중된다.
1위 예측이 틀렸을 때 2~3위 실제 우승마의 확률도 매우 낮아져 연승 추천을 놓침.

### 확인 방법
temperature=0.3으로 5두 경주에서 softmax를 계산하면:
- 입력 [1.0, 0.8, 0.6, 0.4, 0.2] → 1위 확률 약 90% 이상 집중

### 수정 방법
`config.py:49` 수정:
```python
# 수정 전
SOFTMAX_TEMPERATURE = 0.3

# 수정 후
SOFTMAX_TEMPERATURE = 0.6  # 확률 분포를 더 완만하게 (연승 포착 가능성 증가)
```

### 근거
- 0.3은 top1이 틀렸을 때 top3 내 다른 말들의 확률이 너무 낮아짐
- 0.6은 top3 내 말들에 유의미한 확률을 분산시켜 연승 포착 가능
- 1.0은 원래 softmax와 동일 (너무 균등)
- 0.5~0.7 범위가 실험적으로 적절

---

## 문제 4: implied_prob 피처의 학습/예측 배당 불일치

### 현상
`preprocessor.py:246-251`에서:
- 학습 시: `win_odds` (경주 결과 확정 후 배당) 사용
- 예측 시: `current_win_odds` (경주 전 실시간 배당) 사용

경마 배당은 경주 시작 직전까지 변동하므로 두 값의 분포가 다를 수 있음.
모델이 `win_odds` 기반 `implied_prob`을 학습했는데 예측에서는 `current_win_odds` 기반 값이 입력됨.

### 위치
`core/preprocessor.py:244-267` — `_add_race_relative_features`

### 수정 방법 (보수적 접근)
`win_odds`와 `current_win_odds` 중 어느 것을 쓰든 분포가 크게 다르지 않을 수 있으므로,
우선 예측 시 `current_win_odds`가 없는 경우 `win_odds`로 fallback하되,
두 컬럼을 명시적으로 통합한 단일 컬럼 `_odds_for_model`을 만들어 사용:

```python
def _add_race_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # current_win_odds 우선, 없으면 win_odds, 둘 다 없으면 NaN
    if "current_win_odds" in df.columns:
        odds_src = pd.to_numeric(df["current_win_odds"], errors="coerce").replace(0, np.nan)
    elif "win_odds" in df.columns:
        odds_src = pd.to_numeric(df["win_odds"], errors="coerce").replace(0, np.nan)
    else:
        odds_src = pd.Series(np.nan, index=df.index)

    df["implied_prob"] = 1.0 / odds_src
    # ... 이하 동일
```

현재 코드와 로직은 동일하나, 주석을 명확히 해서 학습 시에도 `current_win_odds`를 수집해 저장하는 방향으로 장기적으로 개선을 유도.

**장기 개선 방향 (이번 태스크에서 구현 안 해도 됨):**
- 학습 데이터 수집 시 경주 직전 배당(`current_win_odds`)도 DB에 저장
- `win_odds`(결과 배당)와 `current_win_odds`(예측 배당)를 분리해서 관리

---

## 수정 우선순위 요약

| 순위 | 파일 | 수정 내용 | 난이도 |
|---|---|---|---|
| 1 | `config.py:49` | `SOFTMAX_TEMPERATURE` 0.3 → 0.6 | 쉬움 (1줄) |
| 2 | `config.py:53-80` | `FEATURE_COLUMNS`에 7개 피처 추가 | 쉬움 (7줄 추가) |
| 3 | `core/preprocessor.py:32-62` | `transform()`에 누락 피처 fallback 추가 | 쉬움 (5줄 추가) |
| 4 | `core/trainer.py:74-156` | `tune()` 내부를 XGBRanker+NDCG 기반으로 교체 | 어려움 (함수 전면 수정) |

**권장 실행 순서:** 1→2→3 먼저 적용해 재학습 후 성능 확인, 이후 4 진행.
