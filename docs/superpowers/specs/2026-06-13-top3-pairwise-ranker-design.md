# 연승(top3) Pairwise Ranker 추가 설계

## 배경

### 현재 상태
- 단승 모델: XGBRanker(pairwise) + LGBMRanker(lambdarank) + RandomForest + LogisticRegression
- 연승 모델: RandomForest + LogisticRegression **만** (ranker 없음)

### 측정된 성능 (백테스트 13일, 215경주)
- 단승 Precision@1: 52.1% (랜덤 기준 ~10%)
- 연승 Precision@1: 65.1% (랜덤 기준 ~30%)
- top3_prob >= 0.65 기준 실제 top3 적중률: 66.0% (262건 / 159경주)

### 목표
연승 모델에 Pairwise Ranker를 추가해 top3_probability 품질을 높인다.  
임계값은 모델 개선 후 재조정한다.

---

## 설계

### 수정 파일
| 파일 | 변경 내용 |
|------|----------|
| `core/trainer.py` | `top3_ranker_models` dict 추가, `_train_pairwise_rankers_top3()` 신규, `predict_rank_score_top3()` 신규 |
| `core/predictor.py` | top3 ranker score를 RF+LR 결과에 블렌딩 |
| `core/model_manager.py` | `top3_ranker_models` save/load |

신규 파일 없음.

---

### 1. trainer.py

#### 1-1. `top3_ranker_models` 추가
```python
class RaceModelTrainer:
    def __init__(self):
        self.models: dict = {}              # 단승 모델 (분류 + ranker)
        self.top3_models: dict = {}         # 연승 분류 모델 (RF, LR)
        self.top3_ranker_models: dict = {}  # 연승 ranker 모델 (XGBRanker, LGBMRanker) ← 신규
        self.scaler = StandardScaler()
        self.feature_columns: list = []
```

#### 1-2. `train()` 에서 호출 추가
```python
ranker_top3_metrics = self._train_pairwise_rankers_top3(df_sorted, feature_cols, best_params)
```

#### 1-3. top3 ranker 레이블 (graded)
```
rank == 1      → 2  (1등도 top3이며, 가장 높은 relevance)
rank 2 또는 3  → 1  (top3)
rank 4+        → 0  (탈락)
```
단순 이진(0/1)이 아닌 graded label로 ranker가 순위를 더 세밀하게 학습.

#### 1-4. `_train_pairwise_rankers_top3()` 구조
`_train_pairwise_rankers()`와 동일 패턴:
- race_date 기준 train/test split (경주 단위)
- XGBRanker(rank:pairwise) + LGBMRanker(lambdarank) 학습
- 결과를 `self.top3_ranker_models`에 저장
- 평가: Precision@3, NDCG@3

**Precision@3 정의:**  
각 경주에서 top3_ranker_score 상위 3마리 중 실제 final_rank <= 3인 말의 비율 평균.

#### 1-5. `predict_rank_score_top3()` 신규
`predict_rank_score()`와 동일 구조, `top3_ranker_models`를 사용.

---

### 2. predictor.py

현재 win_probability 블렌딩 코드:
```python
# LAMBDARANK_BLEND * ranker_score + (1-LAMBDARANK_BLEND) * clf_proba
# → 경주 내 softmax 정규화 후 합산
result["win_probability"] = result.groupby("race_id").apply(_blend_race)
```

동일 패턴을 top3_probability에 적용:
```python
top3_rank_scores = self.trainer.predict_rank_score_top3(X)
if top3_rank_scores is not None:
    # win 블렌딩과 동일한 _blend_race 로직으로 top3_probability 계산
    result["top3_probability"] = ...blend(top3_rank_scores, clf_probs_top3)...
else:
    result["top3_probability"] = clf_probs_top3
```

블렌딩 비율은 기존 `LAMBDARANK_BLEND` 상수를 재사용 (재학습 후 성능 보고 조정 가능).

> **구현 주의:** 현재 `_blend_race`는 `predict()` 내부 중첩 함수. top3 블렌딩에도 동일 로직이 필요하므로 `_blend_within_race(rank_scores, clf_probs, blend_ratio)` 형태의 내부 메서드로 추출한 뒤 win/top3 양쪽에서 호출.

---

### 3. model_manager.py

save/load 시 `top3_ranker_models` 키 추가:
```python
# save
data = {
    "models": trainer.models,
    "top3_models": trainer.top3_models,
    "top3_ranker_models": trainer.top3_ranker_models,  # ← 추가
    ...
}

# load
trainer.top3_ranker_models = data.get("top3_ranker_models", {})  # ← 추가
```

---

## 평가 계획

학습 후 동일 백테스트 스크립트(13일, 215경주)로 before/after 비교:

| 지표 | 현재 | 목표 |
|------|------|------|
| 연승 Precision@1 | 65.1% | 68%+ |
| top3_prob >= 0.65 적중률 | 66.0% | 69%+ |
| Precision@3 (신규) | 측정 전 | 측정 후 기준선 설정 |

---

## 범위 외 (이번 작업에 포함하지 않음)
- 피처 추가 (연승 특화 피처)
- 임계값 재조정
- Stacking / 메타 모델
- Optuna 튜닝 확장 (top3 ranker 파라미터)
