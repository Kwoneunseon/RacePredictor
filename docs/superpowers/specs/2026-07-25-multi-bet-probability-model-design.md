# 다중 베팅 종류 확률 모델 설계 (단승/연승/복연승)

## 배경

### 계기
네이버 블로그(Henry, "클로드코드를 이용한 경마/경륜 알고리즘 자동 베팅 후기")를
검토하며 현재 코드베이스와 대조한 결과 아래를 확인함([[project_blog_review_edge_gaps]]):
- `core/ev_calculator.py`의 EV(`model_prob × odds - 1`) + 켈리 로직은 이미
  블로그가 설명하는 벤터식 "엣지" 방법론과 동일
- 다만 연승(place) 배당 데이터가 DB에 없어 `core/investor.py`·
  `main.py`의 연승 EV 계산이 단승 배당을 재사용 중(부정확 가능성)
- EV 임계값에 세금(원천징수 22~33%) 미반영

### 현재 모델 구조의 한계
`core/trainer.py`는 단승(`is_winner`)과 연승(top3)을 **독립적으로 학습된
별개 이진분류기 쌍**으로 다룬다(`models`/`top3_models`/`top3_ranker_models`).
이 때문에:
- 경주 내 연승확률의 합이 3이 되도록 보장되지 않음
- 단승모델과 연승모델의 말 순위 판단이 서로 어긋날 수 있음
- 신규 베팅 종류(복연승 등)를 추가하려면 매번 새 이진분류기를 처음부터
  학습해야 함

### 목표
"각 등수(1등~출전마수등)에 말이 정확히 하나씩만 배정되는" 제약을 만족하는
단일 확률모델을 만들어, 그로부터 단승/연승/복연승 확률을 모두 수학적으로
유도한다. 다만 이미 408경주 백테스트로 검증된 연승 모델([[project_next_task]],
[[project_top3_no_blend]])은 보수적으로만 교체한다.

### 소유 에이전트
이 설계 이후의 구현·백테스트·향후 유사 작업은 `quant-model-expert`
서브에이전트(`.claude/agents/quant-model-expert.md`, 2026-07-25 신설)가
전담한다.

---

## 결정된 원칙 (2026-07-25 논의로 확정)

1. **베팅 여부/금액 결정은 항상 결정론적으로.** LLM은 베팅 결정(금액·여부)에
   관여하지 않는다. 필요하면 리포트용 자연어 설명 생성에만 선택적으로 사용.
2. **강도 점수는 기존 `win_probability`를 재사용.** 처음부터 순위 우도로
   재학습(정식 Plackett-Luce MLE)하지 않는다 — 이미 검증/캘리브레이션된
   자산을 버리지 않기 위함.
3. **보수적 롤아웃.** 연승은 기존 raw classifier가 이미 검증되어 있으므로,
   Harville 유도 버전은 **동일 백테스트 기준으로 같거나 나을 때만** 교체.
   복연승은 기존 검증된 모델이 없는 신규 항목이므로 Harville 유도 방식을
   바로 적용.

---

## Phase 1: 피처 안정성 리포트

### 목적
`FEATURE_COLUMNS`(config.py)의 각 피처가 특정 시기에만 유효한 건지, 12개월간
꾸준히 중요한지 확인. Harville 강도 점수의 기반이 되는 `win_probability`
모델의 신뢰도를 보강하는 사전 진단.

### 데이터 흐름
1. `RaceFeatureBuilder`로 뽑은 학습 데이터(서울/제주/부산 통합)를
   `race_date` 기준 최근 12개 캘린더월로 분할
2. 월별 데이터로 `is_winner`/top3 타깃 각각에 대해 RF·LR·XGBClassifier·
   LGBMClassifier를 개별 학습(Optuna 튜닝·랭커·캘리브레이션은 생략 —
   진단용 경량 학습)
3. 월 표본 30경주 미만이면 "데이터 부족"으로 표시하고 스킵
4. 월별 정규화 중요도 벡터 → [피처 × 월] 행렬(히트맵용)
5. 피처별 12개월 평균/표준편차로 변동계수(CV) 산출 → 안정(CV<0.3)/
   보통(0.3~0.5)/불안정(CV>0.5) 분류

### 신규 파일
| 파일 | 내용 |
|------|------|
| `scripts/feature_stability_report.py` | 월별 재학습 + 히트맵/안정성 HTML 리포트 생성 |
| `reports/feature_stability.html` | 산출물 (기존 `feature_importance_report.py`의 라이트/다크 카드 스타일 재사용) |

기존 `scripts/feature_importance_report.py`(최신 모델 1개 기준 정적 중요도)는
그대로 유지 — 데이터 형태가 달라 하나로 합치지 않는다.

### 사용 원칙
불안정 피처가 발견돼도 즉시 제거하지 않는다. 참고 자료로 두고, 제거 여부는
항상 백테스트로 검증한다([[project_new_signal_features_rollout]]과 동일 원칙).

---

## Phase 2: Harville 기반 복연승 확률 유도

### 방법론
1. 경주 내 각 말의 `win_probability`(합=1, 이미 캘리브레이션됨)를 강도
   점수로 사용
2. 순차 소거 방식으로 전체 순위 확률분포 유도:
   - 1등 확률 = 강도 / 전체 강도합 (= 기존 `win_probability`)
   - k등 확률(앞 순위 확정 후) = 강도 / (남은 말들 강도합)
3. 이 분포로부터:
   - 연승(top3) 확률: 특정 말이 1~3등 중 어디든 들어가는 모든 경우의 합
   - 복연승 확률: 특정 두 말이 동시에 1~3등에 들어가는 모든 경우의 합
   - 출전마 수(5~14두)가 작아 직접 열거 또는 DP로 계산 비용 무시할 만함
4. Harville 공식의 알려진 편향(연승/복연승 쪽) 보정: 유도된 확률을 실제
   결과 대비 isotonic regression으로 보정(기존 `win_calibrator`/
   `top3_calibrator` 패턴 재사용)

### 신규/수정 파일
| 파일 | 변경 내용 |
|------|----------|
| `core/rank_distribution.py` (신규) | 강도 점수 → 순차 소거 순위분포 → top3/페어 확률 유도 함수 |
| `core/trainer.py` | 복연승용 isotonic calibrator(`pair_top3_calibrator`) 추가, Harville 유도값 vs 기존 연승 classifier 비교용 백테스트 훅 |
| `core/predictor.py` | `pair_top3_probability`(말 쌍별) 출력 추가 |
| `core/ev_calculator.py` | 베팅 종류별(단승/연승/복연승) EV 계산 일반화 |
| `core/investor.py` | 복연승 EV/켈리 판단 추가 (odds_map 확장 필요, 아래 참고) |
| `scripts/backtest_pair_top3.py` (신규) | 기존 `backtest_top3.py` 패턴으로 복연승 백테스트 |

### 알려진 블로커: 복연승 배당 데이터
`library/scraper_odds.py`는 단승 배당만 수집하고, DB 스키마
(`migrations/001_init_schema.sql`)에도 연승/복연승 배당 컬럼이 없다
([[project_blog_review_edge_gaps]]). 복연승 EV를 정확히 계산하려면 실제
복연승 배당 수집이 선행돼야 한다 — 스크래핑 가능 여부 조사 및 스키마
추가가 필요하면 `db-expert`와 협업. **이 조사가 끝나기 전까지는 복연승
확률 유도/백테스트는 배당 없이 "모델 확률 자체의 정확도"만 검증하고,
실제 EV 기반 베팅 추천은 배당 데이터 확보 후 활성화한다.**

### 롤아웃 절차
1. `core/rank_distribution.py` 구현 + 단위테스트(확률 합=1 검증, 소규모
   출전마수 케이스로 직접 열거와 대조)
2. 기존 408경주 백테스트 기준(0.65 임계값)으로 Harville 유도 연승확률과
   기존 raw classifier 연승확률을 나란히 비교
   - Harville가 같거나 우수 → 연승을 Harville 유도로 교체
   - Harville가 열등 → 연승은 기존 classifier 유지, 복연승에만 Harville 적용
3. 복연승 확률 자체 정확도를 별도 백테스트로 검증(배당 무관, 순수 적중률)
4. 복연승 배당 데이터 확보 후 EV/켈리 로직 활성화

---

## 평가 계획

| 항목 | 기준 |
|------|------|
| Phase 1 산출물 | `reports/feature_stability.html` 생성, 안정/보통/불안정 피처 분류 결과 보고 |
| Harville 연승 vs 기존 연승 | 동일 408경주 백테스트, 적중률/커버리지 비교 |
| 복연승 확률 자체 정확도 | 신규 백테스트(배당 무관), 예측 적중률만 우선 확인 |
| 복연승 EV/켈리 | 배당 데이터 확보 후 별도 평가 |

---

## 범위 외 (이번 작업에 포함하지 않음)
- 정식 Plackett-Luce MLE 재학습(강도 점수를 처음부터 다시 최적화)
- 삼복승/쌍승 등 그 외 베팅 종류
- EV 임계값의 세금 반영 재설계 (별도 작업, [[project_blog_review_edge_gaps]])
- 배당 변화(시계열) 피처 추가 (별도 작업, [[project_blog_review_edge_gaps]])
- LLM 기반 리포트 설명문 생성 (결정에 관여하지 않는 부가 기능, 필요 시 후속 작업)
