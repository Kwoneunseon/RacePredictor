# 예측 성능 개선 설계 문서

**날짜**: 2026-05-28  
**상태**: 승인됨

---

## 문제 진단

| 문제 | 현황 | 영향 |
|------|------|------|
| `horse_race_history` 비어있음 | DB 테이블 존재하나 0 rows | 말 과거 성적 특성 6개 전부 0으로 학습 |
| `win_odds` 특성 미사용 | DB에 있지만 FEATURE_COLUMNS 미포함 | 경마 최강 예측 신호 완전 무시 |
| `prev_total_avg_rank` 버그 | FEATURE_COLUMNS에 있으나 전처리기에서 미계산 | 항상 0 |
| 말 등급 서수 인코딩 오류 | LabelEncoder → 의미 손실 | "국5"→5, "제4"→4 의 서수 관계 무시 |

---

## 개선 범위

### 수정 1: horse_race_history SQL 채우기

`race_entries` 자체 조인으로 말별·날짜별 점진적 과거 성적 집계.  
Supabase SQL Editor에서 1회 실행 후, 새 경주 수집 시 upsert.

계산 항목:
- `prev_total_races`: 해당 날짜 이전 총 출전 수
- `prev_wins`: 1위 횟수
- `prev_top3`: 3위 이내 횟수
- `prev_5_avg_rank`: 최근 5경주 평균 순위 (없으면 6)
- `prev_total_avg_rank`: 전체 평균 순위 (없으면 6)

### 수정 2: 학습/예측 파이프라인 horse_race_history 연결

**`main.py` `cmd_train`**: 학습 데이터 추출 시 각 (horse_id, race_date)에 맞는 horse_race_history 조인.  
**`core/predictor.py` `predict_from_db`**: 예측 날짜 기준으로 horse_race_history 일괄 조회 후 merge.

### 수정 3: win_odds + implied_prob 특성 추가

**`config.py`**: FEATURE_COLUMNS에 `win_odds`, `implied_prob` 추가.

**`core/preprocessor.py` `_add_derived_features`**:
- `implied_prob = 1 / max(win_odds, 1.01)` (배당이 높을수록 낮은 확률)
- `prev_total_avg_rank` 계산 추가: horse_race_history에서 채워진 컬럼 활용
- win_odds 결측 처리: 폴백 배당(거리별 중앙값) 사용 (기존 로직 유지)

### 수정 4: 말 등급 서수 인코딩

**`core/preprocessor.py` `_add_derived_features`**:
- `horse_class` ("국5", "제4", "외6") → 숫자 추출 → `horse_class_num` (int 1~6)
- 숫자 없으면 3 (중간값 default)
- `CATEGORICAL_COLS`에서 `horse_class` 제거, `horse_class_num` 직접 사용

**`config.py`**: `horse_class_enc` → `horse_class_num` 교체.

---

## 제약 조건

- 예측 시 win_odds는 실시간 배당 미구현 → 폴백 배당(DB 중앙값) 사용 (수용됨)
- 학습 데이터의 win_odds = 경주 종료 후 확정 배당 → 사용 가능
- 기존 모델 파일은 재학습으로 교체

---

## 예상 효과

- horse_race_history 채움: 과거 성적 6개 특성 실데이터로 대체
- win_odds/implied_prob 추가: AUC 5~15% 향상 기대 (경마 예측의 핵심 특성)
- 버그 수정: prev_total_avg_rank 0→실제값
- 등급 서수화: 등급 간 순서 관계 모델에 전달

---

## 향후 개선 (이번 스코프 외)

- `15140265` 조교현황 API 수집 → 경주 전 컨디션 특성 추가
- Playwright 실시간 배당 스크래핑 구현
- 기수/조교사 최근 30일 승률 (누적 대신 최신)
