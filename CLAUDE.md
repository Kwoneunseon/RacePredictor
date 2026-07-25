# CLAUDE.md

## 프로젝트 개요

경마 1위/연승 예측 및 자동 투자 봇. Python 3.11, Supabase(Postgres, ORM 없이
`supabase-py`로 PostgREST 직접 호출), XGBoost/LightGBM 기반 ML 파이프라인.

- `library/` — 데이터 수집(공공 API/스크레이핑)
- `core/` — 전처리/피처엔지니어링/학습/예측/베팅 파이프라인
- `Utils/` — DB 싱글톤, 로거, 알림 (전 레이어 공용)
- `main.py`/`scheduler.py` — CLI 진입점/자동 스케줄링
- `migrations/` — 번호 붙은 SQL 스키마 파일 (정식 마이그레이션 툴 없음)
- `scripts/` — 백테스트/분석 1회성 스크립트

## 서브에이전트 운영 규칙

이 프로젝트에는 다음 전문 서브에이전트가 있다:

- **db-expert** (`.claude/agents/db-expert.md`) — 스키마/쿼리/마이그레이션/
  인덱싱/데이터 무결성/운영 DB write
- **architecture-expert** (`.claude/agents/architecture-expert.md`) — 모듈
  의존성/레이어 분리/순환 참조/확장성 (읽기 전용)
- **quant-model-expert** (`.claude/agents/quant-model-expert.md`) — 확률/랭킹
  모델 설계, 피처 엔지니어링/중요도/안정성, 학습·캘리브레이션·백테스트,
  EV/켈리 베팅 로직, 신규 베팅 종류 확률 유도

작업 요청이 들어오면 다음 순서로 판단한다:

1. **범위 판단**: 요청이 db-expert 또는 architecture-expert의 범위에
   해당하면, 이름이 언급되지 않아도 자동으로 해당 에이전트에 위임한다.
2. **복합 작업**: DB와 구조 양쪽에 걸친 작업이면 db-expert와
   architecture-expert를 필요에 따라 병렬 또는 순차로 위임하고, 두 결과를
   종합해서 하나의 실행 계획(무엇을 왜 어떻게 바꿀지)으로 정리해 보고한다.
3. **새로운 전문 영역**: 요청이 위 두 에이전트 범위를 벗어나는데, 앞으로도
   반복될 것 같은 성격의 작업이면(예: 프론트엔드 성능, 보안/인증,
   인프라/배포, 알림 채널 확장 등) — 바로 처리하지 말고 먼저 "이런 전문
   서브에이전트를 새로 만드는 게 좋겠다"고 사용자에게 제안한다. 사용자가
   승인하면 `.claude/agents/`에 `memory: project` 설정을 포함한 새 전문가를
   만들고, 이후 유사 작업에 자동 위임한다.
4. **일회성 작업**: 반복 가능성이 낮은 일회성 작업이면 새 에이전트를 만들지
   않고 내장 `general-purpose` 서브에이전트로 바로 처리한다.
