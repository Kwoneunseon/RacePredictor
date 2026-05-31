# 예측 정확도 개선 설계 — 피처 보강 + LambdaRank

**작성일:** 2026-05-31  
**목적:** win_probability 적중률 및 신뢰도 향상

---

## 현황 및 문제

- 데이터: 12,828건 race_entries, 2025-06 ~ 2026-05 (1년)
- 현재 모델: RandomForest / XGBoost / LightGBM / LogisticRegression (이진 분류)
- AUC: ~0.878, 백테스트 1위 적중률 54% (훈련 데이터 내)
- 문제 A: 경주 내 확률 합계가 100%를 초과 (독립 이진 분류의 구조적 한계)
- 문제 B: 경주별 확률 편차가 극심 (부산 6R 79.5% vs 서울 3R 10.9%)
- 근본 원인: 경주 내 상대 비교 없이 각 말을 독립적으로 분류

---

## 개선 방향

### 1. 피처 보강 (B안)

아래 5개 피처를 추가한다. 모두 `race_date < 당일` 데이터로만 계산해 데이터 리키지 없음.

| 피처명 | 계산 방법 | 근거 |
|---|---|---|
| `horse_track_win_rate` | 말의 해당 meet_code별 과거 승률 | 경마장별 성적 편차가 실제로 큼 |
| `jockey_horse_win_rate` | (jk_no, horse_id) 조합의 과거 승률 | 기수-말 호흡 효과 |
| `recent_rank_trend` | 최근 3경주 순위의 선형 기울기 (음수=개선) | 컨디션 상승/하락세 |
| `class_change` | 직전 경주 대비 horse_class_num 차이 | 클래스 적응도 |
| `weight_trend` | 최근 3경주 horse_weight_diff 평균 | 체중 추세 패턴 |

### 2. LambdaRank 랭킹 모델 (C안)

이진 분류 대신 경주 단위 그룹 랭킹 학습.

**레이블 변환:**
```
final_rank 1위 → relevance 4
final_rank 2위 → relevance 3
final_rank 3위 → relevance 2
4위 이하       → relevance 1
```

**모델:** `LGBMRanker(objective='lambdarank', n_estimators=300)`  
**그룹:** race_id 기준으로 그룹 크기 배열 생성  
**평가 지표:** NDCG@3 (상위 3위 예측 품질)

기존 classifier 모델(RF/XGBoost/LightGBM/LR)은 유지하고 LambdaRank를 앙상블에 추가.  
예측 시 LambdaRank score를 `win_probability` 컬럼에 저장해 기존 파이프라인과 호환 유지.

---

## 변경 파일 범위

| 파일 | 변경 내용 |
|---|---|
| `config.py` | `FEATURE_COLUMNS`에 피처 5개 추가 |
| `core/preprocessor.py` | `_add_new_features()` 메서드 추가 (fit_transform, transform 양쪽) |
| `core/trainer.py` | `LGBMRanker` 학습 로직 추가, NDCG 평가 추가 |
| `core/predictor.py` | LambdaRank score를 `win_probability`로 매핑 |
| `core/model_manager.py` | LambdaRank 모델 저장/로드 처리 |

---

## 데이터 흐름

```
학습 시:
race_entries + races + jockeys + trainers
  → preprocessor.fit_transform()
      → 기존 피처 + 5개 신규 피처
  → trainer.train()
      → classifier 모델들 (기존)
      → LGBMRanker (신규, race_id 그룹 기반)
  → model_manager.save()

예측 시:
predict_from_db(date)
  → preprocessor.transform()
  → LambdaRank.predict() → score
  → win_probability = score (경주 내 상대 점수)
  → 기존 출력 형식 유지
```

---

## 성공 기준

- 백테스트(훈련 외 날짜) 1위 적중률 > 30% (무작위 ~10%)
- NDCG@3 > 0.70
- 경주별 top-1 score 변별력: 경주 내 1위 score가 2위보다 명확히 높음
