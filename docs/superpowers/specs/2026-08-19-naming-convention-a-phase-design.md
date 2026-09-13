# 구조 정리 A단계 설계: 네이밍 정리

## 배경

구조 정리 3단계(C 문서화 → B 경계 재정립 → A 네이밍 정리) 중 마지막
단계. C, B단계는 이미 완료됐다(`docs/architecture.html`이 B단계 완료
상태를 반영하고 있음). 이번 A단계는 `core/`, `library/`의 파일명을
현재의 명사형(`feature_builder.py`, `collector_horse.py` 등)에서
**동사형**으로 전환한다. 이 방향은 B단계 진행 중 `core/select_odds.py`
(구 `odds_collector.py`)를 개명하며 이미 사용자가 확정했다
([[project_a_phase_naming_convention]] 메모리 참고).

architecture-expert가 전체 파일을 조사해 개명안을 제시했고
([[project_a_phase_naming_convention]]에 이어 기록 예정), 브레인스토밍
중 다음을 확정했다:

- **범위는 파일명만.** 클래스명(`OddsCollector`, `RaceFeatureBuilder`
  등)과 메서드명(`build_for_prediction`, `predict_from_db` 등)은
  이번 A단계에서 건드리지 않는다. 근거: `core/`의 클래스 9개가 전부
  일관되게 "행위자명사(-er/-or)" 패턴이라 그 자체로 이미 하나의 관례이고,
  파일명(동작 축)과 클래스명(행위자 축)을 의도적으로 분리하는 것으로
  정리한다. 메서드명은 이미 대부분 동사형이라 손댈 곳이 적고, 범위를
  넓히면 회귀 위험만 커진다.
- **5단계(A-1~A-5) 전부를 하나의 계획으로 진행.** 고위험 4개 파일
  (A-4)을 별도 작업으로 미루지 않고 이번에 함께 처리한다.

## 네이밍 규칙 확인

기존에 확정된 접두어 의미 규칙([[project_a_phase_naming_convention]]):
- `select_`: 조회한다(SQL SELECT 개념)
- `choose_`: 여러 옵션 중 하나를 선택한다
- `get_`: 이미 가진 데이터를 변환/가공해 반환(넓은 의미)

이번 A단계 대상 파일들은 위 세 접두어보다는 각 파일의 실제 동작을
그대로 나타내는 동사(`collect_`, `scrape_`, `build_`, `train_`,
`filter_`, `rebuild_`, `calculate_`, `predict_`, `decide_`, `manage_`,
`preprocess_`, `report_`)를 쓴다. architecture-expert 조사 결과 현재
이름 대부분이 `[명사]_[행위자명사]` 형태라, 전환은 대체로 어순을
`[동사]_[명사]`로 뒤집는 기계적 작업이 된다(예: `feature_builder.py`
→ `build_features.py`).

## 범위: 개명 대상 파일 목록

### library/ (9개)

| 현재 이름 | 새 이름 | 위험도(import 사이트 수) |
|---|---|---|
| `collector_horse.py` | `collect_horse.py` | 중간 |
| `collector_jockeys.py` | `collect_jockeys.py` | 중간 |
| `collector_trainers.py` | `collect_trainers.py` | 중간 |
| `collector_races.py` | `collect_races.py` | 중간~높음(`library/scraper_score.py`가 직접 import, CLI docstring 있음) |
| `collector_race_detail.py` | `collect_race_detail.py` | 적음 |
| `collector_section_records.py` | `collect_section_records.py` | 적음 |
| `scraper_odds.py` | `scrape_odds.py` | 중간(`core/select_odds.py`, `core/feature_builder.py`가 별칭 import로 사용) |
| `scraper_score.py` | `scrape_score.py` | 중간 |
| `scraper_weights.py` | `scrape_weights.py` | 적음(`core/feature_builder.py`, `scheduler.py`) |

### core/ (12개, 위험도 순)

| 현재 이름 | 새 이름 | 위험도 |
|---|---|---|
| `rank_distribution.py` | `calculate_rank_probabilities.py` | 매우 적음(파이프라인 미연결, 테스트만 import) — 파일럿 |
| `investor.py` | `decide_bets.py` | 적음(2곳: `main.py`, 테스트) |
| `prediction_report.py` | `report_predictions.py` | 적음(2곳) |
| `feature_importance_report.py` | `report_feature_importance.py` | 적음(2곳) |
| `race_filter.py` | `filter_races.py` | 적음~중간(3곳) |
| `history_rebuilder.py` | `rebuild_history.py` | 적음~중간(`.vscode/launch.json` 디버그 설정 포함 3곳) |
| `ev_calculator.py` | `calculate_ev.py` | 중간(4곳) |
| `predictor.py` | `predict_races.py` | 중간(`main.py`, `scheduler.py`, 테스트 3곳) |
| `preprocessor.py` | `preprocess_features.py` | 높음(6곳 이상) |
| `feature_builder.py` | `build_features.py` | 높음(6곳 이상, 858줄 최대 모듈) |
| `trainer.py` | `train_models.py` | 높음(6곳 이상) |
| `model_manager.py` | `manage_models.py` | 높음(6곳 이상) |

### 개명하지 않는 것

- `Utils/database.py`, `Utils/logger.py`, `Utils/kakaotalk.py`,
  `library/parsing.py` — 파이프라인 "단계"가 아니라 인프라/공용 유틸
  모듈이라 대상에서 제외.
- `scripts/*.py` — 이미 전부 동사형(`backtest_top3.py` 등), `_backtest_common.py`는
  private 헬퍼 관례상 예외, `kakao_auth.py`는 고유명사라 그대로.
- `main.py`, `scheduler.py`, `config.py` — CLAUDE.md에 이미 진입점/설정
  관례로 명시됨.
- `core/select_odds.py` — B단계에서 이미 완료.
- 클래스명, 메서드명, 인스턴스 변수명 — 위 "범위" 절 참고.

## 각 파일 개명 시 수정 대상

파일 이동은 `git mv`로 하고(히스토리 보존), 다음을 함께 수정한다:

1. 그 파일을 import하는 모든 소비처의 import 경로
2. 대응하는 테스트 파일명(`tests/`가 core/library 구조를 1:1 미러링하는
   관례이므로 `tests/test_feature_builder.py` → `tests/test_build_features.py`
   식으로 함께 리네임)
3. `.vscode/launch.json`처럼 파일 경로를 직접 참조하는 설정 파일
4. 모듈 docstring 안의 실행 예시(`python -m library.collector_races` 같은
   안내가 있는 파일)

클래스/함수 정의 자체, 메서드 이름, 로직은 전혀 건드리지 않는다 —
파일 이름과 그 파일을 가리키는 경로 문자열만 바뀐다.

## 위험도 기반 실행 순서

architecture-expert가 제안한 5단계를 그대로 따른다:

- **A-1 (파일럿)**: `core/rank_distribution.py` — 프로덕션 import가
  전무해 리네임 절차 자체를 검증하기 좋음.
- **A-2 (library/, 낮음~중간)**: `collector_*`/`scraper_*` 8개 파일.
  `collector_races.py`와 `scraper_odds.py`는 상호 참조가 있어 주의.
- **A-3 (core/, 중간)**: `investor.py`, `prediction_report.py`,
  `feature_importance_report.py`, `race_filter.py`, `history_rebuilder.py`,
  `ev_calculator.py`, `predictor.py`.
- **A-4 (core/, 높음)**: `preprocessor.py`, `feature_builder.py`,
  `trainer.py`, `model_manager.py` — 서로 강하게 얽힌 조합이라 반드시
  하나씩 순차로 진행하고, 매 파일 리네임 후 전체 테스트를 돌린다.
- **A-5**: 이번 계획에서는 A-3에 이미 포함(`prediction_report.py`,
  `feature_importance_report.py`는 A-3으로 통합했다 — 원래 architecture-expert
  제안의 A-3/A-5 구분은 "리포트 모듈은 우선순위 낮음"이라는 의미였지,
  별도 계획으로 쪼개라는 뜻은 아니었으므로 이번 스펙에서는 위험도
  순서 안에 자연스럽게 배치했다).

## 검증 방법

- 각 파일 리네임 후 `python -c "import <새 경로>"`로 즉시 임포트 확인.
- 각 파일 리네임 후 관련 테스트 파일만 먼저 실행, 이어서 전체
  `pytest` 실행.
- A-4(고위험 4개) 완료 후에는 추가로 `main.py --mode predict` 스모크
  테스트(architecture-expert 권고) — 단, 이 프로젝트는 로컬 Supabase
  없이 실제 운영 DB에 붙으므로, 이 스모크 테스트는 **읽기 전용 모드
  확인이 가능한 범위에서만** 수행하고, 실제 베팅 실행(`investor.decide()`
  결과로 인한 부작용)이 없는지 먼저 코드를 읽고 확인한 뒤 진행한다.
- 전체 작업 완료 후 `grep -rn "collector_horse\|collector_jockeys\|collector_trainers\|collector_races\|collector_race_detail\|collector_section_records\|scraper_odds\|scraper_score\|scraper_weights\|rank_distribution\|race_filter\|history_rebuilder\|ev_calculator\.py\|core\.predictor\|core\.preprocessor\|core\.feature_builder\|core\.trainer\|core\.model_manager\|core\.investor\|prediction_report\.py\|feature_importance_report\.py" --include=*.py .`
  형태로 옛 모듈명이 남아있지 않은지 전수 확인(단, `investor.py`처럼
  클래스명이나 변수명에 같은 단어가 남는 것은 정상 — 파일 경로
  import 문맥에서만 걸러서 판단).

## 이번 단계에서 하지 않는 것

- 클래스명 변경(`OddsCollector`, `RaceFeatureBuilder`, `ModelManager`,
  `EVCalculator`, `Investor`, `RacePredictor`, `RacePreprocessor`,
  `RaceFilter` 등 9개 전부 유지).
- 메서드/함수 레벨 네이밍 정리(단, `library/collector_races.py`의
  `collecting_race_results` 같은 동명사형 함수명은 발견됐지만 별도
  후속 작업으로 미룸 — 이번 스펙 범위 아님).
- `core/select_odds.py`/`core/feature_builder.py`의 배당 조회 로직
  중복 통합(B단계에서 이미 별도 이슈로 분리됨).
- `library/collector_trainers.py`의 무가드 `int()` 데이터 유실 버그
  수정(B단계 리뷰에서 발견, 별도 이슈).
- `docs/architecture.html`, `.claude/agents/architecture-expert.md`
  갱신은 이번 스펙에 포함하지 않고, 구현 계획의 마지막 태스크로
  별도 처리한다(B단계와 동일한 패턴).

## 다음 단계

이 스펙 승인 후 `writing-plans` 스킬로 구현 계획을 만든다. 완료 후
`docs/architecture.html`을 새 파일명으로 갱신한다.
