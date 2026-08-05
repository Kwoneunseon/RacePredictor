# 프로젝트 문서 재작성 설계 (구조 정리 1단계: 문서화)

## 배경

사용자가 "현재 상태 개선, 구조 정리"를 요청했다. 목적은 이 프로젝트의 이력을
모르는 사람(또는 새 세션의 에이전트)이 왔을 때도 구조를 쉽게 파악할 수 있게
하는 것. 브레인스토밍을 통해 이 작업이 세 개의 독립적인 하위 프로젝트로
나뉜다는 데 합의했다:

- **C. 전체 데이터/실행 흐름 문서화** ← 이 스펙의 범위
- **B. `core`/`library`/`Utils`/`scripts` 경계 재정립** (파일 이동 포함)
- **A. 모호한 파일/함수/변수명 개선**

진행 순서는 C → B → A로 확정했다. 문서화(C)를 먼저 해서 현재 구조의 문제점을
명확히 드러내고, 그 진단을 B의 입력 자료로 쓴다. 성능/효율 개선은 이번
작업의 목표가 아니며, 구조 정리 과정에서 자연스럽게 따라오는 정도만 반영한다.

기존 `README.md`는 실제 코드와 무관한 가상의 예시(`HorseRacing1stPlacePredictor`
클래스, 가짜 이메일/디스코드 링크, 가짜 논문 인용 등)로 채워져 있었다 —
한 번도 실제 코드와 연결된 적이 없는 템플릿으로 확인됨. 전면 재작성 대상.

## 범위

1. `README.md` 재작성 — 표준적인 README 형식(프로젝트 설명 + 사용법)
2. `docs/architecture.html` 신규 작성 — 상세 구조/파이프라인 문서(HTML)

DB 스키마 상세는 다루지 않는다(`migrations/`가 source of truth, 파일 목록만
한 줄씩 언급). 성능 프로파일링, 실제 코드 이동/리네이밍은 이 단계에서
하지 않는다(각각 B, A 단계 소관).

## 1. `README.md`

표준 오픈소스 README 관례를 따른다 — "이 프로젝트가 뭔지, 어떻게 쓰는지"만
간결하게. 상세 아키텍처는 다루지 않고 `docs/architecture.html`로 링크한다.

섹션 구성:

1. 제목 + 한 줄 설명
2. 핵심 기능 요약 (데이터 수집 → 학습 → 예측 → EV/켈리 기준 베팅 → 자동 스케줄링)
3. 요구사항 (Python 3.11, Supabase 계정)
4. 설치
   - `pip install -r requirements.txt`
   - `.env` 설정: `API_KEY`(공공데이터포털), `SUPABASE_URL`, `SUPABASE_KEY`
     (필수) / `KAKAO_REST_API_KEY`, `KAKAO_ACCESS_TOKEN`(선택 — 카카오톡
     알림용, `Utils/kakao_auth.py`로 최초 발급)
5. 사용법 — 실제 `main.py` CLI 그대로:
   ```
   python main.py --mode train --start-date 2024-01-01 --end-date 2024-12-31
   python main.py --mode predict --date 2026-08-05
   python main.py --mode update-results --start-date ... --end-date ...
   python main.py --mode tune
   python scheduler.py   # 자동 수집(시간별/평일 06:00) + 매일 10:00 예측·카카오톡 전송
   ```
6. 테스트: `pytest`
7. 프로젝트 구조 — 한 문단 요약("`library`가 수집, `core`가
   전처리/학습/예측/베팅, `Utils`가 공용 레이어, `main.py`/`scheduler.py`가
   진입점") + "자세한 구조는 [`docs/architecture.html`](docs/architecture.html)
   참고" 링크
8. 면책조항 — 과장 없이 짧게(교육/연구 목적, 실제 베팅 손실 책임 없음)

가짜 성과 수치, 가짜 연락처, 가짜 참고문헌, 존재하지 않는 API 예시 코드는
전부 제거한다.

## 2. `docs/architecture.html`

정적 HTML 파일, `docs/architecture.html`에 저장. 인터넷 연결 없이 로컬에서
브라우저로 바로 열어볼 수 있어야 한다 — CSS/폰트 전부 인라인, 외부 CDN
의존성 없음. 다이어그램은 mermaid.js 같은 외부 JS 라이브러리 대신 순수
HTML/CSS(박스 + 화살표, flexbox/grid)로 그린다. 라이트/다크 모드
(`prefers-color-scheme`) 모두 지원.

섹션 구성:

1. **파이프라인 개요 다이어그램**
   `library/`(수집) → `core/`(전처리→학습→예측→베팅→리포트) →
   `main.py`/`scheduler.py`(진입점). `Utils/`는 전 레이어를 가로지르는
   공용 레이어로 별도 표시. `scripts/`는 원칙적으로 `core`/`library`를
   소비하는 독립 브랜치이나, `feature_importance_report.py`는
   `main.py:cmd_train()`이 직접 import해서 쓰는 예외로 점선 표시.

2. **단계별 설명** (아래 "모듈 인벤토리" 내용을 서사적으로 재구성)
   - 데이터 수집: `library/collector_*.py`(말/기수/조교사/경주/경주결과/
     구간기록, 공공데이터포털 API), `library/scraper_*.py`(배당/마체중,
     KRA 웹 스크래핑)
   - 전처리/피처엔지니어링: `core/preprocessor.py`(라벨인코딩 등),
     `core/feature_builder.py`(DB→DataFrame), `core/history_rebuilder.py`
     (`horse_race_history` 재구성), `core/race_filter.py`(베팅 대상 경주
     필터)
   - 학습/캘리브레이션: `core/trainer.py`(XGBoost/LightGBM top3
     분류기+ranker, isotonic 캘리브레이션), `core/model_manager.py`
     (모델 저장/로드)
   - 확률 모델: `core/rank_distribution.py`(Harville 방식 top3/복연승
     확률 유도)
   - 예측: `core/predictor.py`(오늘자 경주 예측, 카카오톡 메시지 포맷)
   - 베팅: `core/ev_calculator.py`(EV/켈리 계산), `core/odds_collector.py`
     (실시간 배당 조회), `core/investor.py`(베팅 실행)
   - 리포트: `core/prediction_report.py`(예측결과 HTML),
     `scripts/feature_importance_report.py`(변수중요도 HTML, 학습 시마다
     갱신)
   - 실행 진입점: `main.py`(train/predict/update-results/tune CLI),
     `scheduler.py`(APScheduler — 시간별/평일 06:00 수집, 매일 10:00
     예측+카카오톡)

3. **모듈 인벤토리 표** — `library/`, `core/`, `Utils/`, `scripts/` 각각
   파일명 + 한 줄 책임. 실제 docstring/코드에서 확인한 내용만 사용(추측
   금지). Utils는 4개(`function.py` 공용 헬퍼, `logger.py`, `database.py`
   Supabase 싱글톤+upsert, `kakaotalk.py`/`kakao_auth.py` 카카오 알림).

4. **`migrations/` 이력 요약** — 파일명 + 한 줄(스키마 상세 생략).

5. **알려진 구조적 이슈** (B단계 입력 자료) — architecture-expert 진단
   기반:
   - `scripts/feature_importance_report.py`를 `main.py`가 직접 import —
     "scripts는 core/library를 소비만 한다"는 레이어 원칙과 반대 방향
   - `Utils/function.py` — 이름만으로 내용을 알 수 없는 그랩백 유틸리티
   - `core/odds_collector.py`가 `library/scraper_odds.py`와 이름이
     겹치면서 역할이 다름(전자는 core 레이어의 배당 조회 클래스, 후자는
     library 레이어의 스크래핑 함수) — 새로 온 사람이 혼동하기 쉬움
   - 이 섹션은 문서 작성 중 실제 코드를 더 살펴보며 발견되는 대로 추가

## 검증 방법

- README의 모든 CLI 예시는 `main.py`/`scheduler.py` 실제 인자와 1:1
  대조해서 작성(이미 확인 완료: `--mode {train,predict,update-results,tune}`,
  `--start-date`/`--end-date`/`--date`/`--meet`)
- `.env` 변수는 `config.py`의 `os.getenv(...)` 호출과 대조해서 작성(이미
  확인 완료: `API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`,
  `KAKAO_REST_API_KEY`, `KAKAO_ACCESS_TOKEN`)
- 모듈 인벤토리 표의 각 항목은 실제 파일의 docstring 또는 최상단 코드를
  읽고 작성(이미 1차 조사 완료, 작성 시 재확인)
- 작성 후 `docs/architecture.html`을 브라우저로 열어 다크/라이트 모드
  전환, 레이아웃 깨짐 여부를 육안 확인

## 유지보수 메모

이 문서(특히 `docs/architecture.html`의 다이어그램·모듈 인벤토리 표)는
**현재(B/A 이전) 구조의 스냅샷**이다. B(경계 재정립)에서 파일이 실제로
이동하면 경로가 달라지므로, B/A 각 단계의 마지막에 이 문서를 새 구조에
맞게 갱신하는 작업을 해당 단계의 스펙/계획에 포함시킨다.

## 다음 단계

이 스펙 승인 후 `writing-plans` 스킬로 구현 계획을 만든다. 이 작업 완료
후에는 B(경계 재정립), A(네이밍 정리)를 별도 스펙으로 이어가며, 각 단계
마지막에 위 유지보수 메모에 따라 이 문서를 갱신한다.
