# 경마 자동 투자 봇

경마 1위/연승 예측 및 자동 투자 봇. 공공데이터포털 API와 KRA 웹 데이터를
수집해 XGBoost/LightGBM 기반 모델로 순위를 예측하고, EV(기대값)/켈리
비율 기준으로 베팅 여부를 판단합니다.

## 핵심 기능

- 말/기수/조교사/경주결과/배당 데이터 자동 수집
- top3(연승) 분류기 + ranker 학습, 단승 확률 isotonic 캘리브레이션(연승은 raw 유지)
- EV/켈리 비율 기반 베팅 로직(단승/연승)
- `scheduler.py`를 통한 자동 수집 + 매일 10:00 예측 결과 카카오톡 전송

`core/rank_distribution.py`에 Harville 방식 확률 모델(단승/연승/복연승)이
구현·테스트돼 있지만, 아직 베팅 파이프라인에는 연결되지 않았습니다.

자세한 코드 구조와 데이터 흐름은 브라우저로 [`docs/architecture.html`](docs/architecture.html)을
열어 참고하세요(GitHub에서는 소스로만 보입니다).

## 요구사항

- Python 3.11
- Supabase 프로젝트(Postgres)
- 공공데이터포털 API 키

## 설치

```bash
pip install -r requirements.txt
```

`.env.example`을 복사해 `.env`를 만들고 값을 채우세요:

```bash
cp .env.example .env
```

`API_KEY`/`SUPABASE_URL`/`SUPABASE_KEY`는 필수입니다. `KAKAO_REST_API_KEY`/
`KAKAO_ACCESS_TOKEN`/`KAKAO_REFRESH_TOKEN`은 선택(예측 결과 카카오톡
알림용)이며, 비워두면 로그에 경고만 남고 나머지 파이프라인은 정상
진행됩니다.

카카오톡 알림을 쓰려면 `python -m scripts.kakao_auth`로 토큰을 최초 1회
발급받으세요(access/refresh 토큰이 `.env`에 저장되고, 이후 만료 시
`kakaotalk.py`가 자동 갱신합니다).

## 사용법

```bash
# 모델 학습
python main.py --mode train --start-date 2024-01-01 --end-date 2024-12-31

# 특정 날짜 경주 예측(--mode train으로 학습된 모델이 먼저 있어야 함)
python main.py --mode predict --date 2026-08-05

# 경주 결과 반영(정산)
python main.py --mode update-results --start-date 2026-08-01 --end-date 2026-08-05

# 하이퍼파라미터 튜닝
python main.py --mode tune

# 자동 스케줄러 실행(데이터 수집 + 매일 10:00 예측/카카오톡 전송)
python scheduler.py
```

`--meet`은 `--mode predict`에서만 유효합니다(1=서울, 2=제주, 3=부산경남).
생략하면 예측 결과는 전체 경마장을 보여주지만, 베팅 판단(EV 계산)은
서울(1) 기준으로 계산됩니다.

## 테스트

```bash
pytest
```

## 프로젝트 구조

- `library/` — 데이터 수집(공공 API/스크레이핑)
- `core/` — 전처리/피처엔지니어링/학습/예측/베팅 파이프라인
- `Utils/` — DB 싱글톤, 로거, 카카오톡 알림(전 레이어 공용)
- `main.py` / `scheduler.py` — CLI 진입점 / 자동 스케줄링
- `migrations/` — 번호 붙은 SQL 스키마 파일
- `scripts/` — 백테스트/분석 스크립트
- `tests/` — pytest 테스트
- `config.py` — 환경변수 로드 + 베팅/학습 임계값 설정

파이프라인 다이어그램, 모듈 인벤토리 표, migrations 이력, 알려진 구조적
이슈까지 자세히 보려면 [`docs/architecture.html`](docs/architecture.html)을
브라우저로 여세요.

## 면책조항

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 실제 베팅에 따른
손실에 대해 책임지지 않습니다. 합법적인 경마장에서만, 책임감 있게
베팅하시기 바랍니다.
