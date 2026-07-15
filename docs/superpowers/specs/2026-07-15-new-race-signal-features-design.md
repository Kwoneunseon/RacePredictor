# 신규 경주 신호 피처 추가 설계 (부담중량·신마여부·구간기록·성적·러닝스타일)

## 배경

`scripts/backtest_win.py` / `scripts/backtest_top3.py`로 측정한 현재 기준선(408경주, 2026-05-01~06-25): 단승 모델 Precision@1 42.9% (인기마 기준선 22.3%), 연승 모델 79.9%(0.65 임계값, 69.6% 커버리지). 정확도를 더 올리기 위해, 지금 모델이 쓰지 않는 신규 신호를 추가한다.

라이브 API 조사 결과 다음을 확인했다:
- 기존에 연동된 3개 트랙별 경주결과 API(서울 `API186_1`/제주 `API4_2`/부산경남 `API4_3`)의 `rcBudam`/`budam` 필드는 **부담중량(kg)이 아니라 부담구분 코드(1:마령, 2:별정, 3:핸디캡)**다 (`_info/테이블 구조.txt:89` 및 라이브 응답 실측값 `rcBudam=2` 로 확인). 이 API들에는 서울 기준 실제 kg 값을 담은 필드가 없다.
- 새로 활용신청·승인받은 통합 API **`API299/Race_Result_total`** (data.go.kr 15119524, "경마시행당일_경주결과종합")가 3개 트랙 전체를 통일된 스키마로 커버하며, 실제 부담중량 kg(`wgBudam`), 코너별 통과순위(`sj_1cOrd`~`sj_4cOrd` 등), 구간별 누적기록(`buG1fAccTime`~`buG8fAccTime`, `se_1cAccTime`~`se_4cAccTime` 등), 말/기수/조교사 통산·최근1년 성적(`ord1CntT/Y`, `jkOrd1CntT/Y`, `trOrd1CntT/Y`, `rcCntT/Y` 등)을 모두 제공한다. `rc_date`로 과거 날짜 조회가 가능해 과거분 소급 조회(백필)도 이 API 하나로 가능하다.
- 새로 승인받은 **`API37_1/sectionRecord_1`** (data.go.kr 15057859, "마필 구간별 경주기록")는 말별로 거리 구간(S1F, 1C~4C, G3F, G1F)마다 KRA가 이미 계산해둔 최고(F)/평균(A)/최저(S) 기록을 제공한다. 짧은 거리 경주는 1C/2C 값이 0으로 나오는데, 이는 결측이 아니라 "해당 코너 없음"이다.

## 스코프

이번 설계는 다음 4가지 신호를 **3단계로 나누어** 순차 도입하고, 각 단계 후 반드시 백테스트로 개선 여부를 확인한다:

- **1단계:** 부담중량(kg), 신마여부(과거 출전 횟수 count로)
- **2단계:** 구간기록(최고/평균/최저), 말·기수·조교사 통산/최근1년 성적
- **3단계:** 코너별 통과순위 기반 러닝스타일 피처 (가장 새로운 신호이자 가장 구현 리스크가 큰 것 — 앞 단계가 안정적으로 검증된 뒤 진행)

**스코프 밖:** 기존 3개 트랙별 경주결과 API(`collector_races.py`)를 이 새 API로 대체하는 리팩터링은 하지 않는다(안정적으로 동작 중인 기존 파이프라인 리스크 회피). `top3_probability`는 계속 raw classifier 확률만 사용하며 이번 피처 추가와 무관하게 이 규칙은 유지한다([[project_top3_no_blend]] 참고).

## 데이터 소스 및 필드 매핑

| 신호 | 소스 | 필드 | 비고 |
|---|---|---|---|
| 부담중량(kg) | `API299/Race_Result_total` | `wgBudam` | 기존 3개 API의 `rcBudam`/`budam`(카테고리 코드)은 사용 안 함 |
| 신마여부 | 없음 — 우리 DB 자체 | `race_entries`에서 해당 horse_id의 `race_date` 이전 출전 횟수 | count로 저장(0/1 boolean 아님). DB 수집 시작 이전 데뷔마는 이 값이 실제보다 작게 나올 수 있음(좌측 절단) — 알려진 한계로 문서화만 하고 넘어감 |
| 구간기록(S1F,1C~4C,G3F,G1F 최고/평균/최저) | `API37_1/sectionRecord_1` | `FRcTime*`(최고), `ARcTime*`(평균), `SRcTime*`(최저) — 거리(`rcDist`)별로 로우가 나뉨 | 1C/2C 값 0은 "해당 코너 없음" — 결측과 구분해서 처리 |
| 말/기수/조교사 통산·최근1년 성적 | `API299/Race_Result_total` | 말: `ord1CntT/Y`, `ord2CntT/Y`, `rcCntT/Y` / 기수: `jkOrd1CntT/Y`, `jkOrd2CntT/Y`, `jkRcCntT/Y`, `jkCareer` / 조교사: `trOrd1CntT/Y`, `trOrd2CntT/Y`, `trRcCntT/Y`, `trCareer` | `hrOrd1CntT` 등 유사 이름의 중복 필드가 있어 구현 착수 시 실제 값 비교로 어느 쪽이 맞는지 재확인 필요 |
| 코너별 통과순위(러닝스타일) | `API299/Race_Result_total` | `sj_1cOrd`~`sj_4cOrd`, `sjS1fOrd`, `sjG1fOrd`, `sjG3fOrd` | 3단계에서만 사용. 짧은 거리는 해당 코너 자체가 없어 0/결측 구분 필요 |

## 컴포넌트 및 데이터 흐름

```
[API299/Race_Result_total]  → library/collector_race_detail.py (신규)
                                  → race_entry_detail 테이블 (신규)
[API37_1/sectionRecord_1]   → library/collector_section_records.py (신규)
                                  → horse_section_records 테이블 (신규)
[기존 race_entries 이력]     → core/feature_builder.py 내 계산 (신마여부, 신규 API 불필요)
                                  ↓
                        core/feature_builder.py (신규 피처 조인·계산, 단계별 fillna 처리)
                                  ↓
                        python main.py --mode train (단계별 재학습)
                                  ↓
                        scripts/backtest_win.py / scripts/backtest_top3.py (단계별 검증)
```

기존 `collector_races.py`(서울/제주/부산경남 3개 API)와 `race_entries`의 기존 컬럼(`horse_weight`, `win_odds`, `final_rank` 등)은 변경하지 않는다.

### 신규 테이블

**`race_entry_detail`** (API299 기반, races/race_entries의 보충 테이블)
- PK: `(race_id, horse_id)`
- 컬럼: `assigned_weight`(wgBudam), `sj_1c_ord`, `sj_2c_ord`, `sj_3c_ord`, `sj_4c_ord`, `sj_s1f_ord`, `sj_g1f_ord`, `sj_g3f_ord`, `horse_career_wins`, `horse_career_places`, `horse_career_starts`, `horse_recent1y_wins`, `horse_recent1y_places`, `horse_recent1y_starts`, `jockey_career_wins`, `jockey_career_places`, `jockey_career_starts`, `jockey_recent1y_wins`, `jockey_recent1y_places`, `jockey_recent1y_starts`, `trainer_career_wins`, `trainer_career_places`, `trainer_career_starts`, `trainer_recent1y_wins`, `trainer_recent1y_places`, `trainer_recent1y_starts`, `updated_at`
- 1단계 구현 시 `assigned_weight`만 채우고, 2단계에서 성적 컬럼, 3단계에서 통과순위 컬럼을 채우는 식으로 같은 테이블을 단계적으로 확장한다 (매 단계 새 컬럼만 추가하는 마이그레이션)

**`horse_section_records`**
- PK: `(horse_id, race_distance, snapshot_race_date)`
- 컬럼: `s1f_best`, `s1f_avg`, `s1f_worst`, `c1_best`, `c1_avg`, `c1_worst`, `c2_*`, `c3_*`, `c4_*`, `g3f_*`, `g1f_*`, `updated_at`
- `c1`/`c2` 값이 0인 경우 별도 플래그 컬럼(`c1_applicable BOOLEAN`, `c2_applicable BOOLEAN`) 없이, feature_builder에서 "짧은 거리(예: 900~1200m)는 1C/2C 미해당"이라는 규칙으로 판단한다 — 정확한 거리 임계값은 구현 시 실제 데이터 분포로 확정한다.

### 과거 데이터 백필

`API299`/`API37_1` 모두 `rc_date` 기반 과거 조회가 가능하므로, 기존 3개 트랙별 수집기를 건드리지 않고 **날짜 범위를 순회하며 두 신규 API를 호출해 신규 테이블만 채우는 별도 백필 스크립트**를 작성한다 (기존 `[백필 1]/[백필 2]` VSCode launch-config 패턴과 동일한 형태). 최소 기존 백테스트 검증 기간(2026-05-01~06-25, 408경주)과 학습 데이터 기간을 커버해야 재학습 시 새 피처가 실제로 반영된다.

## 피처 엔지니어링 (`core/feature_builder.py`)

트레이너(`core/trainer.py`)는 최종적으로 `[feature_cols].fillna(0)`을 적용하므로, 0이 물리적으로 말이 안 되는 값(부담중량 0kg, 구간기록 0초)인 피처는 **feature_builder 단계에서 미리 적절한 값으로 채워서 트레이너에 0-결측치를 넘기지 않는다.**

- `assigned_weight`: 트랙·거리별 중앙값으로 결측 대체 (로그 남김), `config.FEATURE_COLUMNS`에 추가
- `prior_starts_count`(신마여부 대체): 해당 말의 과거 출전 횟수. 0이면 데뷔마. boolean이 아닌 count 자체를 피처로 사용
- 구간기록 피처: `race_date` 이전 가장 최근 `snapshot_race_date` 스냅샷 조인 (`prev_win_odds`와 동일하게 `race_date <` 조건으로 리키지 방지). 1C/2C 미해당 구간은 결측 대체값을 다른 구간과 다르게 처리(중앙값 대체 시 "미해당" 표본은 집계에서 제외)
- 성적 피처: 원시 횟수를 그대로 넣지 않고 **비율로 변환** (`recent1y_win_rate = recent1y_wins / recent1y_starts`, 0으로 나누기 방지), 표본수(`recent1y_starts`) 자체도 별도 피처로 유지해 신뢰도 정보를 보존
- 3단계 러닝스타일 피처: 말의 과거 경주들에서 "초반 코너 순위 - 최종 순위" 델타의 평균 등으로 계산 (구체적 피처 정의는 3단계 구현 착수 시 세부 설계)

각 신규 계산 함수는 기존 `_fetch_prev_odds_for_training` 패턴을 따라 `_fetch_*_for_training(df, ...)` 형태로 작성한다.

## 단계별 롤아웃 및 검증

각 단계는 반드시: 백필 → 재학습(`python main.py --mode train`) → 기존 백테스트 검증 기간(2026-05-01~06-25)으로 `backtest_win.py`/`backtest_top3.py` 재실행 → 이전 단계 대비 Precision@1/적중률 변화 기록. 개선이 없거나 악화되면 해당 단계 피처를 `config.FEATURE_COLUMNS`에서 제외하고 원인을 기록한 뒤 다음 단계로 넘어간다 (한 번에 여러 단계를 묶어서 판단하지 않는다 — 과거 `FILTER_MAX_HORSES` 과적합 사고 재발 방지).

## 에러 처리 정책

- 신규 collector가 API 호출에 실패(403/네트워크 오류 등)하면 조용히 스킵하지 않고 로그를 남긴 뒤 해당 배치만 건너뛴다 (`collector_training.py`의 기존 403 처리와 동일 패턴, 단 이번엔 이미 승인된 API라 403은 예외 상황으로 취급)
- feature_builder의 결측치 대체는 모두 로그로 남긴다 (몇 건이 대체됐는지) — 기존 `prev_win_odds`의 조용한 중앙값 대체 문제를 반복하지 않는다
- 구간기록의 "코너 해당없음"(값 0)과 "데이터 없음"(값 자체가 없음)은 서로 다른 처리 경로를 탄다

## 테스트 정책

- `feature_builder.py`의 신규 계산 함수(`prior_starts_count`, 구간기록 조인, 성적 비율 변환 등)는 기존 컨벤션대로 `tests/test_feature_builder.py`에 유닛 테스트 추가
- 신규 collector 모듈(`collector_race_detail.py`, `collector_section_records.py`)은 기존 `collector_training.py` 등과 동일하게 유닛 테스트 없음 (실제 API 연동 위주, 이 프로젝트의 기존 컨벤션)
- 최종 판단 기준은 테스트 통과가 아니라 단계별 백테스트 수치 변화

## 완료 기준

- 1·2·3단계 각각 백필 → 재학습 → 백테스트 재실행까지 완료되고, 각 단계의 전/후 Precision@1(단승) 및 적중률(연승) 수치가 기록됨
- 3단계까지 마친 시점의 최종 수치가 현재 기준선(단승 42.9%, 연승 79.9%@0.65)과 비교되어 사용자에게 보고됨
- 개선되지 않은 단계가 있다면 어느 단계인지, 왜인지가 문서화됨 (다음 개선 방향 판단 근거로 사용)
