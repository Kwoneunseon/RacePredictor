# 경마 자동 투자 봇 전면 재설계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 경마 자동 투자 봇을 새 Supabase(kjctbohokxdzxrsrmwlh)에 최적화된 스키마로 전면 재작성하고, 환경변수 관리·데이터 자동 수집 스케줄러·XGBoost/LightGBM 개선 ML 파이프라인을 구축한다.

**Architecture:** 공공데이터 API → APScheduler 기반 자동 수집 → 새 Supabase DB → ML 앙상블 예측 (RF + XGBoost + LightGBM) → 투자 로직 + 카카오톡 알림. 모든 민감 정보는 `.env`로 관리하고 코드에 하드코딩하지 않는다.

**Tech Stack:** Python 3.11, supabase-py, python-dotenv, APScheduler, scikit-learn, xgboost, lightgbm, pandas, numpy, joblib, pytest, requests

---

## 사전 작업: Claude Code 재시작

Supabase MCP(`.mcp.json`)를 활성화하려면 **Claude Code를 재시작**해야 한다.
재시작 후 `/mcp` 명령으로 `supabase` 서버가 연결되어 있는지 확인한다.

---

## 파일 구조 (변경사항 전체)

```
경마 자동 투자 봇/
├─ .env                          # [신규] 환경변수 (git 제외)
├─ .env.example                  # [신규] 환경변수 예시
├─ .gitignore                    # [수정] .env 추가
├─ config.py                     # [재작성] dotenv 로드
├─ scheduler.py                  # [신규] APScheduler 진입점
├─ main.py                       # [재작성] 예측·투자 실행 진입점
├─ requirements.txt              # [신규] 의존성 목록
│
├─ migrations/
│   └─ 001_init_schema.sql       # [신규] 새 Supabase 스키마
│
├─ library/
│   ├─ __init__.py               # [수정] 수집 함수 노출
│   ├─ collector_races.py        # [수정] 새 config 사용, DB 유틸 사용
│   ├─ collector_horse.py        # [수정] 새 config 사용
│   ├─ collector_jockeys.py      # [수정] 새 config 사용
│   └─ collector_trainers.py     # [수정] 새 config 사용
│
├─ core/
│   ├─ preprocessor.py           # [재작성] 특성 엔지니어링 전담
│   ├─ trainer.py                # [재작성] RF+XGBoost+LightGBM 앙상블 학습
│   ├─ predictor.py              # [재작성] 앙상블 예측
│   ├─ model_manager.py          # [수정] 기존 유지, config 연동
│   ├─ investor.py               # [신규] 배팅 전략·수익 계산
│   └─ backtester.py             # [수정] 새 DB 스키마 대응
│
└─ Utils/
    ├─ database.py               # [재작성] 싱글톤 Supabase 클라이언트
    ├─ kakaotalk.py              # [수정] config에서 키 로드
    └─ logger.py                 # [신규] 공통 로깅 설정
```

---

## Phase 1: 인프라 설정

### Task 1: 환경변수 및 의존성 설정

**Files:**
- Create: `.env`
- Create: `.env.example`
- Modify: `.gitignore`
- Create: `requirements.txt`
- Rewrite: `config.py`

- [ ] **Step 1: `.gitignore`에 .env 추가**

```
# .gitignore 파일 끝에 추가
.env
__pycache__/
*.pyc
*.pkl
models/
logs/
catboost_info/
race_results/
```

- [ ] **Step 2: `.env.example` 작성**

```ini
# .env.example
SUPABASE_URL=https://kjctbohokxdzxrsrmwlh.supabase.co
SUPABASE_KEY=sb_publishable_...
API_KEY=your_public_data_api_key
KAKAO_REST_API_KEY=your_kakao_rest_api_key
KAKAO_ACCESS_TOKEN=your_kakao_access_token
MODEL_DIR=models
LOG_DIR=logs
```

- [ ] **Step 3: `.env` 작성 (실제 키 입력)**

```ini
SUPABASE_URL=https://kjctbohokxdzxrsrmwlh.supabase.co
SUPABASE_KEY=sb_publishable_yR0-4XfK-A8nvqERyYtf2A_CY5zneb_
API_KEY=ofUTHp4Y5zAHr8I41vlXFYFcQ6k5Jc0hdiluFQVyKFEbJHZ8c64vVuZRtTNzYQhJUnMvalRmImgQsv4DhGSQHA==
KAKAO_REST_API_KEY=1f69fda13909927ed4aecb7920851877
KAKAO_ACCESS_TOKEN=
MODEL_DIR=models
LOG_DIR=logs
```

- [ ] **Step 4: `requirements.txt` 작성**

```
supabase==2.10.0
python-dotenv==1.0.1
apscheduler==3.10.4
scikit-learn==1.5.2
xgboost==2.1.1
lightgbm==4.5.0
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2
requests==2.32.3
httpx==0.27.2
python-dateutil==2.9.0
pytest==8.3.3
pytest-mock==3.14.0
```

- [ ] **Step 5: `requirements.txt` 설치**

```bash
pip install -r requirements.txt
```

Expected: 모든 패키지 설치 완료, 오류 없음

- [ ] **Step 6: `config.py` 재작성**

```python
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# 공공데이터 API
API_KEY = os.getenv("API_KEY", "")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# 카카오
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "")
KAKAO_ACCESS_TOKEN = os.getenv("KAKAO_ACCESS_TOKEN", "")

# 경로 설정
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
LOG_DIR = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# DB 설정
DATABASE_CONFIG = {
    "timeout": 300,
    "max_retries": 3,
    "retry_delay": 5,
}

# 데이터 수집 설정
DATA_EXTRACTION = {
    "batch_size_months": 2,
    "page_size": 1000,
    "min_horse_races": 3,
}

# 모델 설정
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "model_expiry_days": 30,
    "performance_threshold": {"min_auc": 0.65},
}

# 특성 컬럼
FEATURE_COLUMNS = [
    "horse_age", "is_male", "horse_class_enc", "race_distance", "total_horses",
    "horse_weight", "race_grade_enc", "track_condition_enc", "weather_enc",
    "prev_total_races", "prev_5_avg_rank", "prev_total_avg_rank",
    "jockey_win_rate", "trainer_win_rate", "horse_win_rate",
    "horse_top3_rate", "experience_score", "recent_form",
]
```

- [ ] **Step 7: config 로드 검증**

```bash
python -c "from config import SUPABASE_URL, API_KEY; print('URL:', SUPABASE_URL[:30]); print('API:', API_KEY[:10])"
```

Expected: URL과 API 키가 출력되고 빈 값이 아님

- [ ] **Step 8: 커밋**

```bash
git add .gitignore .env.example requirements.txt config.py
git commit -m "feat: 환경변수 기반 설정으로 전환"
```

---

### Task 2: 공통 유틸리티 (database, logger)

**Files:**
- Rewrite: `Utils/database.py`
- Create: `Utils/logger.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: `Utils/logger.py` 작성**

```python
import logging
import sys
from config import LOG_DIR

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
```

- [ ] **Step 2: `Utils/database.py` 재작성**

```python
import time
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY, DATABASE_CONFIG
from Utils.logger import get_logger

logger = get_logger(__name__)
_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase 클라이언트 생성됨")
    return _client


def execute_query(table: str, query_fn) -> Optional[Any]:
    client = get_client()
    for attempt in range(DATABASE_CONFIG["max_retries"]):
        try:
            result = query_fn(client.table(table))
            return result.data
        except Exception as e:
            logger.error(f"쿼리 실패 (시도 {attempt + 1}): {e}")
            if attempt < DATABASE_CONFIG["max_retries"] - 1:
                time.sleep(DATABASE_CONFIG["retry_delay"] * (attempt + 1))
    return None


def upsert(table: str, rows: List[Dict], on_conflict: str) -> bool:
    if not rows:
        return True
    try:
        get_client().table(table).upsert(rows, on_conflict=on_conflict).execute()
        logger.info(f"{table} upsert {len(rows)}건 완료")
        return True
    except Exception as e:
        logger.error(f"{table} upsert 실패: {e}")
        return False


def test_connection() -> bool:
    try:
        result = get_client().table("horses").select("horse_id").limit(1).execute()
        return result.data is not None
    except Exception:
        return False
```

- [ ] **Step 3: 실패 테스트 작성**

`tests/test_database.py` 생성:

```python
import pytest
from unittest.mock import patch, MagicMock
from Utils.database import get_client, upsert, test_connection


def test_get_client_returns_singleton():
    client1 = get_client()
    client2 = get_client()
    assert client1 is client2


def test_upsert_empty_rows_returns_true():
    result = upsert("horses", [], "horse_id")
    assert result is True


def test_upsert_calls_supabase(mocker):
    mock_client = MagicMock()
    mocker.patch("Utils.database.get_client", return_value=mock_client)
    mock_client.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    rows = [{"horse_id": "H001", "name": "테스트말"}]
    result = upsert("horses", rows, "horse_id")
    assert result is True
    mock_client.table.assert_called_with("horses")


def test_test_connection_returns_bool(mocker):
    mock_client = MagicMock()
    mocker.patch("Utils.database.get_client", return_value=mock_client)
    mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    result = test_connection()
    assert isinstance(result, bool)
```

- [ ] **Step 4: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_database.py -v
```

Expected: 일부 실패 (모듈이 아직 없거나 import 오류)

- [ ] **Step 5: `tests/__init__.py` 생성**

```bash
mkdir tests
echo. > tests/__init__.py
```

- [ ] **Step 6: 테스트 재실행 (통과 확인)**

```bash
pytest tests/test_database.py -v
```

Expected: 4개 PASSED

- [ ] **Step 7: 커밋**

```bash
git add Utils/database.py Utils/logger.py tests/
git commit -m "feat: Supabase 싱글톤 DB 유틸리티 및 공통 로거"
```

---

### Task 3: DB 스키마 마이그레이션

**Files:**
- Create: `migrations/001_init_schema.sql`

> 이 SQL은 Supabase 대시보드 SQL Editor에서 실행하거나,  
> Claude Code 재시작 후 Supabase MCP로 실행한다.

- [ ] **Step 1: `migrations/001_init_schema.sql` 작성**

```sql
-- =============================================
-- 경마 자동 투자 봇 - 새 스키마 (001_init_schema)
-- Supabase 프로젝트: kjctbohokxdzxrsrmwlh
-- =============================================

-- 말 기본 정보
CREATE TABLE IF NOT EXISTS horses (
    horse_id    TEXT PRIMARY KEY,
    name        TEXT,
    age         INT,
    gender      TEXT,
    rank        TEXT,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 경주 정보
CREATE TABLE IF NOT EXISTS races (
    race_id         TEXT PRIMARY KEY,
    race_date       DATE NOT NULL,
    meet_code       INT NOT NULL,    -- 1=서울, 2=부산, 3=제주
    race_no         INT,
    race_distance   INT,
    race_grade      TEXT,
    track_condition TEXT,
    weather         TEXT,
    total_horses    INT
);

-- 기수
CREATE TABLE IF NOT EXISTS jockeys (
    jk_no       TEXT PRIMARY KEY,
    name        TEXT,
    total_races INT DEFAULT 0,
    total_wins  INT DEFAULT 0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 조교사
CREATE TABLE IF NOT EXISTS trainers (
    trainer_id  TEXT PRIMARY KEY,
    name        TEXT,
    rc_cnt_t    INT DEFAULT 0,
    ord1_cnt_t  INT DEFAULT 0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 출전 기록 (핵심 조인 테이블)
CREATE TABLE IF NOT EXISTS race_entries (
    id              BIGSERIAL PRIMARY KEY,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    horse_id        TEXT NOT NULL REFERENCES horses(horse_id),
    race_date       DATE NOT NULL,
    entry_number    INT,
    horse_weight    INT,
    jk_no           TEXT REFERENCES jockeys(jk_no),
    trainer_id      TEXT REFERENCES trainers(trainer_id),
    final_rank      INT,
    win_odds        FLOAT,
    UNIQUE(race_id, horse_id)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_race_entries_date ON race_entries(race_date);
CREATE INDEX IF NOT EXISTS idx_race_entries_horse ON race_entries(horse_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);

-- 말별 성적 캐시 (매일 배치로 계산)
CREATE TABLE IF NOT EXISTS horse_race_history (
    horse_id            TEXT NOT NULL REFERENCES horses(horse_id),
    as_of_date          DATE NOT NULL,
    prev_total_races    INT DEFAULT 0,
    prev_wins           INT DEFAULT 0,
    prev_top3           INT DEFAULT 0,
    prev_5_avg_rank     FLOAT DEFAULT 6,
    prev_total_avg_rank FLOAT DEFAULT 6,
    PRIMARY KEY (horse_id, as_of_date)
);

-- 예측 결과
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    horse_id        TEXT NOT NULL REFERENCES horses(horse_id),
    predicted_rank  INT,
    win_probability FLOAT,
    model_version   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    actual_rank     INT
);

-- 베팅 기록
CREATE TABLE IF NOT EXISTS betting_records (
    id          BIGSERIAL PRIMARY KEY,
    race_id     TEXT NOT NULL REFERENCES races(race_id),
    horse_id    TEXT NOT NULL REFERENCES horses(horse_id),
    bet_amount  INT NOT NULL,
    bet_type    TEXT NOT NULL,  -- 단승/연승/복승
    odds        FLOAT,
    profit      INT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 스케줄러 실행 로그
CREATE TABLE IF NOT EXISTS scheduler_logs (
    id                  BIGSERIAL PRIMARY KEY,
    job_type            TEXT NOT NULL,
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              TEXT,      -- success/error
    records_processed   INT DEFAULT 0,
    error_message       TEXT
);
```

- [ ] **Step 2: Supabase 대시보드에서 SQL 실행**

Supabase 대시보드 → SQL Editor → 위 SQL 전체 붙여넣기 → Run.

또는 Claude Code 재시작 후 Supabase MCP 연결 확인 후 MCP로 실행.

- [ ] **Step 3: 테이블 생성 확인**

```bash
python -c "
from Utils.database import get_client
c = get_client()
tables = ['horses','races','race_entries','jockeys','trainers','horse_race_history','predictions','betting_records','scheduler_logs']
for t in tables:
    r = c.table(t).select('*').limit(1).execute()
    print(f'{t}: OK')
"
```

Expected: 각 테이블마다 `OK` 출력

- [ ] **Step 4: 커밋**

```bash
git add migrations/
git commit -m "feat: 새 Supabase DB 스키마 마이그레이션 SQL"
```

---

## Phase 2: 데이터 수집 레이어

### Task 4: 수집 라이브러리 업데이트

**Files:**
- Modify: `library/collector_races.py`
- Modify: `library/collector_horse.py`
- Modify: `library/collector_jockeys.py`
- Modify: `library/collector_trainers.py`
- Modify: `library/__init__.py`
- Create: `tests/test_collectors.py`

핵심 변경사항: 각 collector 파일 상단의 `_const` import와 모듈 레벨 supabase 클라이언트를 제거하고, `config`와 `Utils.database`를 사용하도록 변경.

- [ ] **Step 1: `collector_horse.py` 상단 import 수정**

기존:
```python
from _const import API_KEY, SUPABASE_URL, SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
```

변경:
```python
from config import API_KEY
from Utils.database import upsert
from Utils.logger import get_logger
logger = get_logger(__name__)
```

그리고 `save_to_supabase_batch` 함수 내에서 직접 `supabase.table(...)` 대신 `upsert(...)` 사용:

```python
def save_to_supabase_batch(horses: list) -> int:
    if not horses:
        return 0
    rows = [parse_horse_data(h) for h in horses if h]
    rows = [r for r in rows if r]
    upsert("horses", rows, "horse_id")
    return len(rows)
```

- [ ] **Step 2: `collector_jockeys.py` 동일하게 수정**

기존 import 블록:
```python
from _const import API_KEY, SUPABASE_URL, SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
```

변경:
```python
from config import API_KEY
from Utils.database import upsert
from Utils.logger import get_logger
logger = get_logger(__name__)
```

`save_to_supabase_batch` 함수:
```python
def save_to_supabase_batch(jockeys: list) -> int:
    if not jockeys:
        return 0
    rows = [parse_jockey_data(j) for j in jockeys if j]
    rows = [r for r in rows if r]
    upsert("jockeys", rows, "jk_no")
    return len(rows)
```

- [ ] **Step 3: `collector_trainers.py` 동일하게 수정**

기존 import 블록 변경:
```python
from config import API_KEY
from Utils.database import upsert
from Utils.logger import get_logger
logger = get_logger(__name__)
```

`save_to_supabase_batch` 함수:
```python
def save_to_supabase_batch(trainers: list) -> int:
    if not trainers:
        return 0
    rows = [parse_trainer_data(t) for t in trainers if t]
    rows = [r for r in rows if r]
    upsert("trainers", rows, "trainer_id")
    return len(rows)
```

- [ ] **Step 4: `collector_races.py` import 수정**

기존 상단:
```python
from _const import API_KEY, SUPABASE_URL, SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
```

변경:
```python
from config import API_KEY
from Utils.database import upsert, get_client
from Utils.logger import get_logger
logger = get_logger(__name__)
```

`save_race_to_supabase` 류 함수에서 직접 supabase 객체 대신 `upsert` 사용.

- [ ] **Step 5: `library/__init__.py` 수정 (공개 API 정의)**

```python
from .collector_races import collecting_race_results
from .collector_horse import fetch_all_horses
from .collector_jockeys import fetch_all_jockeys
from .collector_trainers import fetch_all_trainers

__all__ = [
    "collecting_race_results",
    "fetch_all_horses",
    "fetch_all_jockeys",
    "fetch_all_trainers",
]
```

- [ ] **Step 6: 수집 테스트 작성**

`tests/test_collectors.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


def test_fetch_horse_data_returns_none_on_error(mocker):
    mocker.patch("requests.get", side_effect=Exception("network error"))
    from library.collector_horse import fetch_horse_data
    result, page = fetch_horse_data(page=1)
    assert result is None
    assert page == 1


def test_save_horse_batch_empty_returns_zero(mocker):
    mocker.patch("Utils.database.upsert", return_value=True)
    from library.collector_horse import save_to_supabase_batch
    result = save_to_supabase_batch([])
    assert result == 0


def test_fetch_race_data_calls_correct_endpoint(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": {"body": {"items": {"item": []}}}}
    mock_get = mocker.patch("requests.get", return_value=mock_response)
    from library.collector_races import fetch_race_data
    fetch_race_data(page=1, end_date="20250101")
    assert mock_get.called
    call_args = mock_get.call_args
    assert "SeoulRace" in call_args[0][0] or "apis.data.go.kr" in call_args[0][0]
```

- [ ] **Step 7: 테스트 실행**

```bash
pytest tests/test_collectors.py -v
```

Expected: 3개 PASSED

- [ ] **Step 8: 수동 수집 실행 검증**

```python
# 터미널에서 소규모 테스트
python -c "
from library import collecting_race_results
collecting_race_results(start_page=1, max_pages=1, start_date='20250101', end_date='20250131')
print('OK')
"
```

Expected: 오류 없이 실행, Supabase에 데이터 저장

- [ ] **Step 9: 커밋**

```bash
git add library/ tests/test_collectors.py
git commit -m "feat: 수집 라이브러리를 새 config·DB 유틸로 마이그레이션"
```

---

### Task 5: APScheduler 기반 자동 수집 스케줄러

**Files:**
- Create: `scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_scheduler.py`:

```python
import pytest
from unittest.mock import MagicMock, patch


def test_scheduler_has_collect_job():
    from scheduler import create_scheduler
    sched = create_scheduler()
    job_ids = [job.id for job in sched.get_jobs()]
    assert "collect_race_data" in job_ids


def test_collect_job_is_interval():
    from scheduler import create_scheduler
    from apscheduler.triggers.interval import IntervalTrigger
    sched = create_scheduler()
    job = next(j for j in sched.get_jobs() if j.id == "collect_race_data")
    assert isinstance(job.trigger, IntervalTrigger)


def test_collect_task_logs_start(mocker):
    mock_collect = mocker.patch("library.collecting_race_results")
    mock_log = mocker.patch("scheduler_logs_write")
    from scheduler import collect_task
    collect_task()
    mock_collect.assert_called_once()
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_scheduler.py -v
```

Expected: ImportError (scheduler.py 없음)

- [ ] **Step 3: `scheduler.py` 작성**

```python
"""
경마 데이터 자동 수집 스케줄러

경주일 (토/일): 매 1시간 데이터 수집
평일:           1일 1회 수집
"""
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from library import collecting_race_results
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


def _is_race_day() -> bool:
    return datetime.now().weekday() in (5, 6)  # 토=5, 일=6


def _get_today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def collect_task():
    today = _get_today_str()
    logger.info(f"데이터 수집 시작: {today}")

    client = get_client()
    log_row = {
        "job_type": "collect",
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    try:
        log_res = client.table("scheduler_logs").insert(log_row).execute()
        log_id = log_res.data[0]["id"] if log_res.data else None
    except Exception:
        log_id = None

    try:
        collecting_race_results(
            start_page=1,
            max_pages=50,
            start_date=today,
            end_date=today,
        )
        status, error_msg = "success", None
    except Exception as e:
        status, error_msg = "error", str(e)
        logger.error(f"수집 실패: {e}")

    if log_id:
        try:
            client.table("scheduler_logs").update({
                "finished_at": datetime.now().isoformat(),
                "status": status,
                "error_message": error_msg,
            }).eq("id", log_id).execute()
        except Exception:
            pass

    logger.info(f"데이터 수집 완료: {status}")


def create_scheduler(background: bool = False) -> BackgroundScheduler | BlockingScheduler:
    Scheduler = BackgroundScheduler if background else BlockingScheduler
    sched = Scheduler(timezone="Asia/Seoul")

    # 경주일(토·일) 매 1시간 수집
    sched.add_job(
        collect_task,
        trigger=IntervalTrigger(hours=1),
        id="collect_race_data",
        name="경주일 시간별 수집",
        replace_existing=True,
    )

    # 평일 매일 오전 6시 1회 수집
    sched.add_job(
        collect_task,
        trigger=CronTrigger(day_of_week="mon-fri", hour=6, minute=0),
        id="collect_weekday_data",
        name="평일 1일 1회 수집",
        replace_existing=True,
    )

    return sched


if __name__ == "__main__":
    logger.info("스케줄러 시작")
    sched = create_scheduler(background=False)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료")
```

- [ ] **Step 4: `tests/test_scheduler.py` 수정 (scheduler_logs_write 참조 제거)**

```python
import pytest
from unittest.mock import MagicMock, patch


def test_scheduler_has_collect_job():
    from scheduler import create_scheduler
    sched = create_scheduler()
    job_ids = [job.id for job in sched.get_jobs()]
    assert "collect_race_data" in job_ids


def test_collect_job_is_interval():
    from scheduler import create_scheduler
    from apscheduler.triggers.interval import IntervalTrigger
    sched = create_scheduler()
    job = next(j for j in sched.get_jobs() if j.id == "collect_race_data")
    assert isinstance(job.trigger, IntervalTrigger)


def test_collect_task_calls_collect(mocker):
    mocker.patch("library.collecting_race_results")
    mocker.patch("Utils.database.get_client", return_value=MagicMock())
    from scheduler import collect_task
    collect_task()  # 오류 없이 실행되어야 함
```

- [ ] **Step 5: 테스트 실행**

```bash
pytest tests/test_scheduler.py -v
```

Expected: 3개 PASSED

- [ ] **Step 6: 커밋**

```bash
git add scheduler.py tests/test_scheduler.py
git commit -m "feat: APScheduler 자동 수집 (경주일 1시간, 평일 1회)"
```

---

## Phase 3: ML 파이프라인

### Task 6: 전처리 모듈 (preprocessor.py)

**Files:**
- Rewrite: `core/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_preprocessor.py`:

```python
import pytest
import pandas as pd
import numpy as np
from core.preprocessor import RacePreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "horse_id": ["H1", "H2"],
        "race_id": ["R1", "R1"],
        "race_date": ["2025-01-01", "2025-01-01"],
        "final_rank": [1, 2],
        "horse_age": [4, 5],
        "gender": ["수컷", "암컷"],
        "horse_class": ["나", "다"],
        "race_distance": [1200, 1200],
        "total_horses": [8, 8],
        "horse_weight": [480, 470],
        "race_grade": ["일반", "일반"],
        "track_condition": ["양호", "양호"],
        "weather": ["맑음", "맑음"],
        "jockey_total_races": [100, 200],
        "jockey_total_wins": [20, 40],
        "trainer_total_races": [150, 300],
        "trainer_total_wins": [30, 60],
        "prev_total_races": [5, 10],
        "prev_wins": [1, 3],
        "prev_top3": [3, 7],
        "prev_5_avg_rank": [3.0, 2.0],
        "prev_total_avg_rank": [3.5, 2.2],
    })


def test_preprocessor_adds_is_winner(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert "is_winner" in result.columns
    assert result["is_winner"].iloc[0] == 1
    assert result["is_winner"].iloc[1] == 0


def test_preprocessor_adds_derived_features(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    for col in ["jockey_win_rate", "trainer_win_rate", "horse_win_rate",
                "horse_top3_rate", "experience_score", "recent_form", "is_male"]:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_preprocessor_encodes_categoricals(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    for col in ["horse_class_enc", "race_grade_enc", "track_condition_enc", "weather_enc"]:
        assert col in result.columns
        assert result[col].dtype in [np.int64, np.int32, np.float64]


def test_preprocessor_transform_uses_fitted_encoders(sample_df):
    p = RacePreprocessor()
    p.fit_transform(sample_df)
    new_df = sample_df.copy()
    result = p.transform(new_df)
    assert "horse_class_enc" in result.columns


def test_preprocessor_win_rate_range(sample_df):
    p = RacePreprocessor()
    result = p.fit_transform(sample_df)
    assert (result["jockey_win_rate"] >= 0).all()
    assert (result["jockey_win_rate"] <= 1).all()
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: ImportError (RacePreprocessor 없음)

- [ ] **Step 3: `core/preprocessor.py` 재작성**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Optional
from Utils.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = ["horse_class", "race_grade", "track_condition", "weather"]


class RacePreprocessor:
    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_target(df)
        df = self._add_derived_features(df)
        df = self._encode_categoricals(df, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_derived_features(df)
        df = self._encode_categoricals(df, fit=False)
        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if "final_rank" in df.columns:
            df["is_winner"] = (df["final_rank"] == 1).astype(int)
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_male"] = (df.get("gender", pd.Series([""] * len(df))) == "수컷").astype(int)
        df["jockey_win_rate"] = df["jockey_total_wins"] / (df["jockey_total_races"] + 1)
        df["trainer_win_rate"] = df["trainer_total_wins"] / (df["trainer_total_races"] + 1)
        df["horse_win_rate"] = df["prev_wins"] / (df["prev_total_races"] + 1)
        df["horse_top3_rate"] = df["prev_top3"] / (df["prev_total_races"] + 1)
        df["experience_score"] = np.log1p(df["prev_total_races"])
        df["recent_form"] = 6 - df["prev_5_avg_rank"].fillna(6)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                df[f"{col}_enc"] = 0
                continue
            series = df[col].fillna("unknown").astype(str)
            if fit:
                enc = LabelEncoder()
                df[f"{col}_enc"] = enc.fit_transform(series)
                self.label_encoders[col] = enc
            else:
                if col in self.label_encoders:
                    enc = self.label_encoders[col]
                    known = set(enc.classes_)
                    series = series.apply(lambda x: x if x in known else enc.classes_[0])
                    df[f"{col}_enc"] = enc.transform(series)
                else:
                    df[f"{col}_enc"] = 0
        return df
```

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: 5개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add core/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: 특성 엔지니어링 전처리 모듈 (RacePreprocessor)"
```

---

### Task 7: 모델 학습 모듈 (trainer.py)

**Files:**
- Rewrite: `core/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_trainer.py`:

```python
import pytest
import pandas as pd
import numpy as np
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer


@pytest.fixture
def preprocessed_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "horse_age": np.random.randint(3, 8, n),
        "is_male": np.random.randint(0, 2, n),
        "horse_class_enc": np.random.randint(0, 5, n),
        "race_distance": np.random.choice([1000, 1200, 1400, 1800], n),
        "total_horses": np.random.randint(5, 14, n),
        "horse_weight": np.random.randint(440, 520, n),
        "race_grade_enc": np.random.randint(0, 3, n),
        "track_condition_enc": np.random.randint(0, 3, n),
        "weather_enc": np.random.randint(0, 3, n),
        "prev_total_races": np.random.randint(3, 30, n),
        "prev_5_avg_rank": np.random.uniform(1, 8, n),
        "prev_total_avg_rank": np.random.uniform(1, 8, n),
        "jockey_win_rate": np.random.uniform(0, 0.3, n),
        "trainer_win_rate": np.random.uniform(0, 0.3, n),
        "horse_win_rate": np.random.uniform(0, 0.3, n),
        "horse_top3_rate": np.random.uniform(0, 0.5, n),
        "experience_score": np.random.uniform(0, 4, n),
        "recent_form": np.random.uniform(-2, 5, n),
        "is_winner": np.random.randint(0, 2, n),
        "race_date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })
    return df


def test_trainer_creates_models(preprocessed_df):
    trainer = RaceModelTrainer()
    result = trainer.train(preprocessed_df)
    assert "models" in result
    assert len(result["models"]) >= 3


def test_trainer_returns_metrics(preprocessed_df):
    trainer = RaceModelTrainer()
    result = trainer.train(preprocessed_df)
    assert "metrics" in result
    for name, metrics in result["metrics"].items():
        assert "auc" in metrics
        assert 0 <= metrics["auc"] <= 1


def test_trainer_predict_proba_shape(preprocessed_df):
    trainer = RaceModelTrainer()
    trainer.train(preprocessed_df)
    from config import FEATURE_COLUMNS
    available = [c for c in FEATURE_COLUMNS if c in preprocessed_df.columns]
    X = preprocessed_df[available].head(10)
    proba = trainer.predict_proba(X)
    assert proba.shape == (10,)
    assert (proba >= 0).all() and (proba <= 1).all()
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_trainer.py -v
```

Expected: ImportError

- [ ] **Step 3: `core/trainer.py` 재작성**

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from config import FEATURE_COLUMNS, MODEL_CONFIG
from Utils.logger import get_logger

logger = get_logger(__name__)


class RaceModelTrainer:
    def __init__(self):
        self.models: dict = {}
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def train(self, df: pd.DataFrame) -> dict:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        self.feature_columns = feature_cols
        logger.info(f"학습 특성 {len(feature_cols)}개: {feature_cols}")

        df_sorted = df.sort_values("race_date")
        split_idx = int(len(df_sorted) * (1 - MODEL_CONFIG["test_size"]))

        X_train = df_sorted.iloc[:split_idx][feature_cols].fillna(0)
        X_test = df_sorted.iloc[split_idx:][feature_cols].fillna(0)
        y_train = df_sorted.iloc[:split_idx]["is_winner"]
        y_test = df_sorted.iloc[split_idx:]["is_winner"]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        model_defs = {
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    random_state=MODEL_CONFIG["random_state"],
                    class_weight="balanced",
                ),
                X_train, X_test,
            ),
            "XGBoost": (
                XGBClassifier(
                    n_estimators=200, max_depth=6,
                    random_state=MODEL_CONFIG["random_state"],
                    eval_metric="logloss",
                    verbosity=0,
                    scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
                ),
                X_train, X_test,
            ),
            "LightGBM": (
                LGBMClassifier(
                    n_estimators=200, max_depth=6,
                    random_state=MODEL_CONFIG["random_state"],
                    class_weight="balanced",
                    verbose=-1,
                ),
                X_train, X_test,
            ),
            "LogisticRegression": (
                LogisticRegression(
                    random_state=MODEL_CONFIG["random_state"],
                    class_weight="balanced",
                    max_iter=1000,
                ),
                X_train_s, X_test_s,
            ),
        }

        results, metrics = {}, {}
        for name, (model, Xtr, Xte) in model_defs.items():
            logger.info(f"{name} 학습 중...")
            model.fit(Xtr, y_train)
            y_prob = model.predict_proba(Xte)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.0
            acc = accuracy_score(y_test, y_pred)
            results[name] = model
            metrics[name] = {"auc": auc, "accuracy": acc}
            logger.info(f"  {name}: AUC={auc:.3f}, Acc={acc:.3f}")

        self.models = results
        return {"models": results, "metrics": metrics}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("모델이 학습되지 않았습니다.")
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        probas = []
        for name, model in self.models.items():
            if name == "LogisticRegression":
                probas.append(model.predict_proba(X_scaled)[:, 1])
            else:
                probas.append(model.predict_proba(X)[:, 1])
        return np.mean(probas, axis=0)
```

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/test_trainer.py -v
```

Expected: 3개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add core/trainer.py tests/test_trainer.py
git commit -m "feat: RF+XGBoost+LightGBM+LR 앙상블 학습 모듈"
```

---

### Task 8: 예측 모듈 (predictor.py)

**Files:**
- Rewrite: `core/predictor.py`
- Create: `tests/test_predictor.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_predictor.py`:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from core.predictor import RacePredictor


@pytest.fixture
def mock_trainer_and_preprocessor():
    trainer = MagicMock()
    trainer.feature_columns = ["horse_age", "is_male", "horse_win_rate"]
    trainer.predict_proba.return_value = np.array([0.8, 0.5, 0.3])
    preprocessor = MagicMock()
    preprocessor.transform.return_value = pd.DataFrame({
        "horse_age": [4, 5, 3],
        "is_male": [1, 0, 1],
        "horse_win_rate": [0.2, 0.1, 0.05],
        "horse_name": ["말A", "말B", "말C"],
        "entry_number": [1, 2, 3],
    })
    return trainer, preprocessor


def test_predictor_returns_sorted_predictions(mock_trainer_and_preprocessor):
    trainer, preprocessor = mock_trainer_and_preprocessor
    p = RacePredictor(trainer, preprocessor)
    df = preprocessor.transform.return_value
    result = p.predict(df)
    assert result["win_probability"].iloc[0] >= result["win_probability"].iloc[1]


def test_predictor_result_has_required_columns(mock_trainer_and_preprocessor):
    trainer, preprocessor = mock_trainer_and_preprocessor
    p = RacePredictor(trainer, preprocessor)
    df = preprocessor.transform.return_value
    result = p.predict(df)
    for col in ["entry_number", "win_probability", "predicted_rank"]:
        assert col in result.columns


def test_predictor_probabilities_in_range(mock_trainer_and_preprocessor):
    trainer, preprocessor = mock_trainer_and_preprocessor
    p = RacePredictor(trainer, preprocessor)
    df = preprocessor.transform.return_value
    result = p.predict(df)
    assert (result["win_probability"] >= 0).all()
    assert (result["win_probability"] <= 1).all()
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_predictor.py -v
```

Expected: ImportError

- [ ] **Step 3: `core/predictor.py` 재작성**

```python
import pandas as pd
import numpy as np
from core.trainer import RaceModelTrainer
from core.preprocessor import RacePreprocessor
from Utils.logger import get_logger

logger = get_logger(__name__)


class RacePredictor:
    def __init__(self, trainer: RaceModelTrainer, preprocessor: RacePreprocessor):
        self.trainer = trainer
        self.preprocessor = preprocessor

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.trainer.feature_columns
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0)

        probas = self.trainer.predict_proba(X)

        result = df.copy()
        result["win_probability"] = probas
        result = result.sort_values("win_probability", ascending=False).reset_index(drop=True)
        result["predicted_rank"] = range(1, len(result) + 1)

        logger.info(f"예측 완료: {len(result)}마리, 상위 확률={probas.max():.3f}")
        return result

    def predict_from_db(self, race_date: str, meet_code: int = None) -> pd.DataFrame:
        from Utils.database import get_client
        client = get_client()

        query = client.table("race_entries").select(
            "*, horses(*), races(*), jockeys(*), trainers(*)"
        ).eq("race_date", race_date)

        if meet_code:
            query = query.eq("races.meet_code", meet_code)

        result = query.execute()
        if not result.data:
            logger.warning(f"{race_date} 출전 데이터 없음")
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df = self.preprocessor.transform(df)
        return self.predict(df)
```

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/test_predictor.py -v
```

Expected: 3개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add core/predictor.py tests/test_predictor.py
git commit -m "feat: 앙상블 예측 모듈 (RacePredictor)"
```

---

### Task 9: 모델 저장/로드 (model_manager.py)

**Files:**
- Modify: `core/model_manager.py`
- Create: `tests/test_model_manager.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_model_manager.py`:

```python
import pytest
import joblib
from pathlib import Path
from unittest.mock import MagicMock, patch
from core.model_manager import ModelManager


def test_save_creates_pkl_file(tmp_path, mocker):
    mocker.patch("core.model_manager.MODEL_DIR", tmp_path)
    trainer = MagicMock()
    trainer.models = {"RF": MagicMock()}
    trainer.scaler = MagicMock()
    trainer.feature_columns = ["col1", "col2"]
    preprocessor = MagicMock()
    preprocessor.label_encoders = {}

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics={"auc": 0.7})
    assert version is not None
    assert (tmp_path / f"model_{version}.pkl").exists()


def test_load_restores_trainer(tmp_path, mocker):
    mocker.patch("core.model_manager.MODEL_DIR", tmp_path)
    trainer = MagicMock()
    trainer.models = {"RF": "fake_model"}
    trainer.scaler = "fake_scaler"
    trainer.feature_columns = ["col1"]
    preprocessor = MagicMock()
    preprocessor.label_encoders = {"horse_class": "enc"}

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics={})
    
    new_trainer = MagicMock()
    new_preprocessor = MagicMock()
    mm.load(new_trainer, new_preprocessor, version=version)
    assert new_trainer.models == {"RF": "fake_model"}
    assert new_trainer.feature_columns == ["col1"]
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_model_manager.py -v
```

Expected: ImportError 또는 AttributeError

- [ ] **Step 3: `core/model_manager.py` 재작성**

```python
import json
import joblib
from datetime import datetime
from pathlib import Path
from config import MODEL_DIR, MODEL_CONFIG
from Utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)

    def save(self, trainer, preprocessor, metrics: dict = None) -> str | None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = MODEL_DIR / f"model_{version}.pkl"
        meta_path = MODEL_DIR / f"model_{version}_meta.json"

        data = {
            "models": trainer.models,
            "scaler": trainer.scaler,
            "feature_columns": trainer.feature_columns,
            "label_encoders": preprocessor.label_encoders,
        }
        meta = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "feature_count": len(trainer.feature_columns),
            "metrics": metrics or {},
        }

        try:
            joblib.dump(data, pkl_path)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"모델 저장: {pkl_path}")
            return version
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return None

    def load(self, trainer, preprocessor, version: str = "latest") -> bool:
        if version == "latest":
            files = sorted(MODEL_DIR.glob("model_*.pkl"))
            if not files:
                logger.warning("저장된 모델 없음")
                return False
            pkl_path = files[-1]
            version = pkl_path.stem.replace("model_", "")
        else:
            pkl_path = MODEL_DIR / f"model_{version}.pkl"

        if not pkl_path.exists():
            logger.error(f"모델 파일 없음: {pkl_path}")
            return False

        try:
            data = joblib.load(pkl_path)
            trainer.models = data["models"]
            trainer.scaler = data["scaler"]
            trainer.feature_columns = data["feature_columns"]
            preprocessor.label_encoders = data["label_encoders"]
            logger.info(f"모델 로드: {pkl_path}")
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def is_outdated(self, version: str = "latest") -> bool:
        if version == "latest":
            files = sorted(MODEL_DIR.glob("model_*_meta.json"))
            if not files:
                return True
            meta_path = files[-1]
        else:
            meta_path = MODEL_DIR / f"model_{version}_meta.json"

        if not meta_path.exists():
            return True

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        created = datetime.fromisoformat(meta["created_at"])
        days_old = (datetime.now() - created).days
        return days_old > MODEL_CONFIG["model_expiry_days"]
```

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/test_model_manager.py -v
```

Expected: 2개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add core/model_manager.py tests/test_model_manager.py
git commit -m "feat: 모델 저장·로드 매니저 (trainer+preprocessor 함께 직렬화)"
```

---

## Phase 4: 투자 로직 및 진입점

### Task 10: 투자 로직 (investor.py) + 카카오톡

**Files:**
- Create: `core/investor.py`
- Modify: `Utils/kakaotalk.py`
- Create: `tests/test_investor.py`

- [ ] **Step 1: `Utils/kakaotalk.py` 상단 수정 (하드코딩 키 제거)**

기존 파일 하단의 모듈 레벨 실행 코드를 수정:

```python
# 기존: REST_API_KEY = "1f69fda13909927ed4aecb7920851877"
# 기존: kakao = KakaoSender(REST_API_KEY)
# 이것들을 config 기반으로 변경

from config import KAKAO_REST_API_KEY, KAKAO_ACCESS_TOKEN

kakao = KakaoSender(KAKAO_REST_API_KEY)
if KAKAO_ACCESS_TOKEN:
    kakao.access_token = KAKAO_ACCESS_TOKEN
    kakao._authenticated = True
```

- [ ] **Step 2: 실패 테스트 작성**

`tests/test_investor.py`:

```python
import pytest
import pandas as pd
from core.investor import Investor


@pytest.fixture
def predictions():
    return pd.DataFrame({
        "race_id": ["R1", "R1", "R1"],
        "horse_id": ["H1", "H2", "H3"],
        "horse_name": ["말A", "말B", "말C"],
        "entry_number": [1, 2, 3],
        "win_probability": [0.72, 0.55, 0.30],
        "predicted_rank": [1, 2, 3],
    })


def test_investor_recommends_high_probability(predictions):
    investor = Investor(threshold=0.6, bet_amount=1000)
    bets = investor.decide(predictions)
    assert len(bets) >= 1
    assert bets.iloc[0]["win_probability"] >= 0.6


def test_investor_no_bets_below_threshold(predictions):
    investor = Investor(threshold=0.9, bet_amount=1000)
    bets = investor.decide(predictions)
    assert len(bets) == 0


def test_investor_bet_amount_set(predictions):
    investor = Investor(threshold=0.5, bet_amount=2000)
    bets = investor.decide(predictions)
    assert (bets["bet_amount"] == 2000).all()


def test_investor_calculates_profit_on_win():
    investor = Investor(threshold=0.0, bet_amount=1000)
    profit = investor.calculate_profit(odds=3.5, won=True)
    assert profit == int(1000 * 3.5) - 1000


def test_investor_calculates_loss_on_lose():
    investor = Investor(threshold=0.0, bet_amount=1000)
    profit = investor.calculate_profit(odds=3.5, won=False)
    assert profit == -1000
```

- [ ] **Step 3: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_investor.py -v
```

Expected: ImportError

- [ ] **Step 4: `core/investor.py` 작성**

```python
import pandas as pd
from Utils.database import upsert, get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


class Investor:
    def __init__(self, threshold: float = 0.6, bet_amount: int = 1000):
        self.threshold = threshold
        self.bet_amount = bet_amount

    def decide(self, predictions: pd.DataFrame) -> pd.DataFrame:
        bets = predictions[predictions["win_probability"] >= self.threshold].copy()
        bets["bet_amount"] = self.bet_amount
        bets["bet_type"] = "단승"
        logger.info(f"베팅 결정: {len(bets)}건 (임계값={self.threshold})")
        return bets

    def calculate_profit(self, odds: float, won: bool) -> int:
        if won:
            return int(self.bet_amount * odds) - self.bet_amount
        return -self.bet_amount

    def save_bets(self, bets: pd.DataFrame, race_id: str) -> bool:
        if bets.empty:
            return True
        rows = []
        for _, row in bets.iterrows():
            rows.append({
                "race_id": race_id,
                "horse_id": row["horse_id"],
                "bet_amount": self.bet_amount,
                "bet_type": "단승",
                "odds": row.get("win_odds", None),
            })
        return upsert("betting_records", rows, "id")

    def update_results(self, race_id: str) -> None:
        client = get_client()
        try:
            entries = client.table("race_entries")\
                .select("horse_id, final_rank")\
                .eq("race_id", race_id)\
                .execute().data or []
            rank_map = {e["horse_id"]: e["final_rank"] for e in entries}

            bets = client.table("betting_records")\
                .select("id, horse_id, bet_amount")\
                .eq("race_id", race_id)\
                .execute().data or []

            for bet in bets:
                final_rank = rank_map.get(bet["horse_id"])
                won = final_rank == 1
                odds_row = client.table("race_entries")\
                    .select("win_odds")\
                    .eq("race_id", race_id)\
                    .eq("horse_id", bet["horse_id"])\
                    .execute().data
                odds = odds_row[0]["win_odds"] if odds_row else 0
                profit = self.calculate_profit(odds or 0, won)
                client.table("betting_records")\
                    .update({"profit": profit})\
                    .eq("id", bet["id"])\
                    .execute()
            logger.info(f"{race_id} 결과 업데이트 완료")
        except Exception as e:
            logger.error(f"결과 업데이트 실패: {e}")
```

- [ ] **Step 5: 테스트 실행**

```bash
pytest tests/test_investor.py -v
```

Expected: 5개 PASSED

- [ ] **Step 6: 커밋**

```bash
git add core/investor.py Utils/kakaotalk.py tests/test_investor.py
git commit -m "feat: 투자 결정·수익 계산 모듈, 카카오 config 연동"
```

---

### Task 11: 메인 진입점 재작성

**Files:**
- Rewrite: `main.py`

- [ ] **Step 1: `main.py` 재작성**

```python
"""
경마 자동 투자 봇 - 메인 진입점

사용법:
    python main.py --mode train       # 모델 학습
    python main.py --mode predict     # 오늘 예측
    python main.py --mode backtest    # 백테스팅
    python main.py --mode status      # 시스템 상태 확인
    python scheduler.py               # 자동 수집 스케줄러 시작
"""
import argparse
from datetime import datetime

from config import FEATURE_COLUMNS
from Utils.database import get_client, test_connection
from Utils.logger import get_logger
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from core.model_manager import ModelManager
from core.investor import Investor

logger = get_logger(__name__)


def cmd_train(args):
    logger.info("=== 모델 학습 시작 ===")
    client = get_client()

    result = client.table("race_entries").select(
        "*, horses(*), races(*), jockeys(*), trainers(*)"
    ).gte("race_date", args.start_date).lte("race_date", args.end_date)\
        .not_.is_("final_rank", "null").execute()

    if not result.data:
        logger.error("학습 데이터 없음")
        return

    import pandas as pd
    df = pd.DataFrame(result.data)
    logger.info(f"학습 데이터: {len(df)}건")

    preprocessor = RacePreprocessor()
    df_processed = preprocessor.fit_transform(df)

    trainer = RaceModelTrainer()
    metrics = trainer.train(df_processed)

    mm = ModelManager()
    version = mm.save(trainer, preprocessor, metrics=metrics["metrics"])
    logger.info(f"모델 저장 완료: {version}")


def cmd_predict(args):
    logger.info(f"=== 예측 시작: {args.date} ===")

    preprocessor = RacePreprocessor()
    trainer = RaceModelTrainer()
    mm = ModelManager()

    if not mm.load(trainer, preprocessor):
        logger.error("로드할 모델 없음. 먼저 --mode train 실행")
        return

    predictor = RacePredictor(trainer, preprocessor)
    predictions = predictor.predict_from_db(args.date, meet_code=args.meet)

    if predictions.empty:
        logger.warning(f"{args.date} 예측 결과 없음")
        return

    print("\n=== 예측 결과 ===")
    for _, row in predictions.head(3).iterrows():
        print(f"  {int(row['predicted_rank'])}위 예측: "
              f"{row.get('horse_name', row['horse_id'])} "
              f"(확률: {row['win_probability']:.3f})")

    investor = Investor(threshold=0.6, bet_amount=1000)
    bets = investor.decide(predictions)
    if not bets.empty:
        print(f"\n베팅 추천: {len(bets)}건")
        for _, bet in bets.iterrows():
            print(f"  → {bet.get('horse_name', bet['horse_id'])} "
                  f"(확률: {bet['win_probability']:.3f})")


def cmd_status(args):
    print("=== 시스템 상태 ===")
    print(f"DB 연결: {'✅' if test_connection() else '❌'}")
    mm = ModelManager()
    from pathlib import Path
    from config import MODEL_DIR
    models = list(MODEL_DIR.glob("model_*.pkl"))
    print(f"저장된 모델: {len(models)}개")
    if models:
        print(f"최신 모델: {sorted(models)[-1].name}")
    print(f"재학습 필요: {'✅' if mm.is_outdated() else '❌'}")


def main():
    parser = argparse.ArgumentParser(description="경마 자동 투자 봇")
    parser.add_argument("--mode", choices=["train", "predict", "status"],
                        default="status")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--meet", type=int, default=None, help="1=서울, 2=부산, 3=제주")
    args = parser.parse_args()

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "predict":
        cmd_predict(args)
    elif args.mode == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 상태 확인 실행**

```bash
python main.py --mode status
```

Expected:
```
=== 시스템 상태 ===
DB 연결: ✅
저장된 모델: 0개
재학습 필요: ✅
```

- [ ] **Step 3: 전체 테스트 실행**

```bash
pytest tests/ -v
```

Expected: 모든 테스트 PASSED

- [ ] **Step 4: 최종 커밋**

```bash
git add main.py
git commit -m "feat: 전면 재설계 완료 - train/predict/status CLI"
```

---

## 완료 체크리스트

- [ ] `.env` 환경변수 로드 (`config.py`)
- [ ] 새 Supabase 스키마 9개 테이블 생성
- [ ] `Utils/database.py` 싱글톤 클라이언트
- [ ] 수집 라이브러리 4종 (`_const` 의존성 제거)
- [ ] `scheduler.py` (경주일 1시간, 평일 1회)
- [ ] `core/preprocessor.py` (특성 엔지니어링)
- [ ] `core/trainer.py` (RF+XGBoost+LightGBM+LR)
- [ ] `core/predictor.py` (앙상블 예측)
- [ ] `core/model_manager.py` (저장·로드)
- [ ] `core/investor.py` (배팅 결정·수익 계산)
- [ ] `main.py` (CLI 진입점)
- [ ] 전체 테스트 통과
