# 경마 자동 투자 봇

경마 1위/연승 예측 및 자동 투자 봇. 공공데이터포털 API와 KRA 웹 데이터를
수집해 XGBoost/LightGBM 기반 모델로 순위를 예측하고, EV(기대값)/켈리
비율 기준으로 베팅 여부를 판단합니다.

## 핵심 기능

- 말/기수/조교사/경주결과/배당 데이터 자동 수집
- top3(연승) 분류기 + ranker 학습, isotonic 확률 캘리브레이션
- Harville 방식 확률 모델(단승/연승/복연승)
- EV/켈리 비율 기반 베팅 로직
- `scheduler.py`를 통한 자동 수집 + 매일 10:00 예측 결과 카카오톡 전송

자세한 코드 구조와 데이터 흐름은 [`docs/architecture.html`](docs/architecture.html)을
참고하세요.

## 요구사항

- Python 3.11
- Supabase 프로젝트(Postgres)
- 공공데이터포털 API 키

## 설치

```bash
pip install -r requirements.txt
```

`.env` 파일을 프로젝트 루트에 생성:

```
API_KEY=공공데이터포털_API_키
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# 선택: 예측 결과 카카오톡 알림
KAKAO_REST_API_KEY=
KAKAO_ACCESS_TOKEN=
```

카카오톡 알림을 쓰려면 `python Utils/kakao_auth.py`로 토큰을 최초 1회
발급받으세요(이후 자동 갱신됩니다).

## 사용법

```bash
# 모델 학습
python main.py --mode train --start-date 2024-01-01 --end-date 2024-12-31

# 특정 날짜 경주 예측
python main.py --mode predict --date 2026-08-05

# 경주 결과 반영(정산)
python main.py --mode update-results --start-date 2026-08-01 --end-date 2026-08-05

# 하이퍼파라미터 튜닝
python main.py --mode tune

# 자동 스케줄러 실행(데이터 수집 + 매일 10:00 예측/카카오톡 전송)
python scheduler.py
```

`--meet`으로 경마장을 지정할 수 있습니다(1=서울, 2=제주, 3=부산경남,
생략 시 전체).

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

구조와 데이터 흐름을 그림으로 보려면 [`docs/architecture.html`](docs/architecture.html)을
여세요.

## 면책조항

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 실제 베팅에 따른
손실에 대해 책임지지 않습니다. 합법적인 경마장에서만, 책임감 있게
베팅하시기 바랍니다.
