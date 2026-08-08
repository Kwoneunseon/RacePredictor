# 구조 정리 B단계(경계 재정립) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Utils/function.py`, `scripts/feature_importance_report.py`,
`core/odds_collector.py`, `Utils/kakao_auth.py`를 실제 사용 레이어에
맞게 옮기고, `scripts/_backtest_common.py`가 `core/feature_builder.py`의
private 메서드를 직접 호출하던 캡슐화 위반을 해소한다.

**Architecture:** 5개의 서로 독립적인 파일 이동/개명 작업(A/B/C/E/F).
로직 변경은 없고(단, A의 `safe_str` 공백 제거는 의도된 동작 개선),
위치·이름·import 경로만 정리한다. 기존 pytest 스위트(192개)로 회귀를
검증하고, `Utils/function.py`는 전용 테스트가 없었으므로 이동과 함께
`tests/test_parsing.py`를 신규 추가한다.

**Tech Stack:** Python 3.11, pytest. 코드 이동 성격상 TDD는 "기존 동작을
보존하는지 테스트로 확인"하는 형태로 적용한다(신규 기능 없음).

**Spec:** `docs/superpowers/specs/2026-08-05-boundary-reorg-b-phase-design.md`

---

## ⚠️ 커밋 시 주의: 일부 파일에 이 계획과 무관한 기존 WIP 수정사항이 있음

`main.py`, `core/investor.py`, `library/collector_races.py`,
`scripts/_backtest_common.py` — 이 4개 파일은 **이 계획과 무관한, 이미
진행 중이던 다른 작업의 수정사항**을 working tree에 갖고 있다(git
status에 이미 `M`으로 표시돼 있음). 이 계획의 Task 2/4/5/6이 이
파일들을 건드리는데, 무심코 `git add <파일>`을 하면 이 계획과 상관없는
기존 WIP 변경까지 같이 커밋돼버린다.

**해당 Task의 커밋 스텝에서는 다음 절차를 따를 것** (일반 `git add`
대신):

1. `git diff -- <파일>`로 현재 working tree에 어떤 변경이 있는지 확인.
   이 계획에서 만든 변경과 기존 WIP 변경이 섞여 있는지 확인한다.
2. 섞여 있다면: `git show HEAD:<파일> > <파일>`로 파일을 HEAD 상태로
   되돌린 뒤, 이 Task의 Step에서 지정한 정확한 변경만 다시 적용하고
   `git add <파일>`로 스테이징한다.
3. 그 다음 원래 working tree 내용(기존 WIP + 이번 변경 모두 포함)을
   복원한다 — 예를 들어 되돌리기 전에 `cp <파일> <파일>.bak`으로
   백업해두고, 스테이징 후 `cp <파일>.bak <파일>`로 복원.
4. `git diff --cached -- <파일>`로 스테이징된 diff가 이 Task의 변경
   내용과 **정확히 일치**하는지(기존 WIP 라인이 섞여있지 않은지)
   최종 확인한 뒤에만 커밋한다.

이 파일들과 무관한 나머지 파일(신규 생성 파일, WIP가 없는 파일)은
평소처럼 `git add`해도 된다.

---

## Task 1: `library/parsing.py` 생성 + 테스트 추가 (작업 A-1)

**Files:**
- Create: `library/parsing.py`
- Test: `tests/test_parsing.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_parsing.py` 생성:

```python
from library.parsing import parse_date, safe_int, safe_float, safe_str


def test_parse_date_converts_yyyymmdd_to_iso():
    assert parse_date("20260805") == "2026-08-05"


def test_parse_date_returns_none_on_invalid_input():
    assert parse_date("not-a-date") is None
    assert parse_date(None) is None


def test_safe_int_converts_valid_value():
    assert safe_int("5") == 5
    assert safe_int(5) == 5


def test_safe_int_returns_default_on_empty_or_none():
    assert safe_int(None) == 0
    assert safe_int("") == 0
    assert safe_int("  ") == 0
    assert safe_int(None, default=-1) == -1


def test_safe_int_returns_default_on_invalid_string():
    assert safe_int("abc") == 0


def test_safe_float_converts_valid_value():
    assert safe_float("1.5") == 1.5
    assert safe_float(1.5) == 1.5


def test_safe_float_returns_default_on_empty_or_none():
    assert safe_float(None) == 0.0
    assert safe_float("") == 0.0


def test_safe_str_strips_whitespace():
    assert safe_str("  홍길동  ") == "홍길동"


def test_safe_str_returns_default_on_empty_or_none():
    assert safe_str(None) == ""
    assert safe_str("") == ""
    assert safe_str(None, default="?") == "?"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_parsing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'library.parsing'`

- [ ] **Step 3: `library/parsing.py` 생성**

`Utils/function.py`의 내용을 그대로 옮긴다(로직 변경 없음):

```python
from datetime import datetime

def parse_date(date_str):
    """날짜 문자열을 변환 (YYYYMMDD -> YYYY-MM-DD 문자열)"""
    try:
        date_obj = datetime.strptime(str(date_str), "%Y%m%d").date()
        return date_obj.isoformat()
    except:
        return None

def safe_int(value, default=0):
    """안전한 정수 변환"""
    try:
        return int(value) if value and str(value).strip() != '' else default
    except:
        return default

def safe_float(value, default=0.0):
    """안전한 실수 변환"""
    try:
        return float(value) if value and str(value).strip() != '' else default
    except:
        return None

def safe_str(value, default=''):
    """안전한 문자열 변환"""
    try:
        return str(value).strip() if value else default
    except:
        return default
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_parsing.py -v`
Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
git add library/parsing.py tests/test_parsing.py
git commit -m "feat: library/parsing.py 추가(Utils/function.py 이관 준비) + 단위테스트"
```

---

## Task 2: 4개 collector 파일의 import 전환 + `Utils/function.py` 제거 (작업 A-2)

**Files:**
- Modify: `library/collector_horse.py:8`
- Modify: `library/collector_races.py:7`
- Modify: `library/collector_race_detail.py:12`
- Modify: `library/collector_section_records.py:13`
- Modify: `Utils/__init__.py`
- Delete: `Utils/function.py`

- [ ] **Step 1: `library/collector_horse.py`의 import 수정**

`library/collector_horse.py:8`을 찾아서:

```python
from Utils import safe_int, safe_float
```

다음으로 교체:

```python
from library.parsing import safe_int, safe_float
```

- [ ] **Step 2: `library/collector_races.py`의 import 수정**

`library/collector_races.py:7`을 찾아서:

```python
from Utils import parse_date, safe_int, safe_float, safe_str
```

다음으로 교체:

```python
from library.parsing import parse_date, safe_int, safe_float, safe_str
```

- [ ] **Step 3: `library/collector_race_detail.py`의 import 수정**

`library/collector_race_detail.py:12`를 찾아서:

```python
from Utils import parse_date, safe_int, safe_float, safe_str
```

다음으로 교체:

```python
from library.parsing import parse_date, safe_int, safe_float, safe_str
```

- [ ] **Step 4: `library/collector_section_records.py`의 import 수정**

`library/collector_section_records.py:13`을 찾아서:

```python
from Utils import parse_date, safe_str
```

다음으로 교체:

```python
from library.parsing import parse_date, safe_str
```

- [ ] **Step 5: `Utils/__init__.py`에서 재노출 제거**

`Utils/__init__.py` 현재 전체 내용:

```python
from .function import *
```

빈 파일로 만든다(다른 `Utils` 하위 모듈은 각자 `from Utils.database import
...` 처럼 서브모듈 경로로 직접 import되고 있어 `__init__.py`가 아무것도
재노출하지 않아도 깨지는 곳이 없다 — Step 7에서 grep으로 재확인):

```python
```

(파일을 완전히 비운다. 0바이트 `__init__.py`.)

- [ ] **Step 6: `Utils/function.py` 삭제**

```bash
git rm "Utils/function.py"
```

- [ ] **Step 7: 옛 import가 남아있지 않은지 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "from Utils import\|Utils\.function\|from \.function" --include=*.py .`
Expected: 결과 없음(0건). 만약 남아있는 파일이 있다면 Step 1~4와 같은
방식으로 `library.parsing`으로 교체할 것.

- [ ] **Step 8: 전체 테스트 통과 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 201 passed (기존 192 + Task 1에서 추가한 9개)

- [ ] **Step 9: 커밋**

**`library/collector_races.py`는 이 계획과 무관한 기존 WIP 수정사항이
있는 파일이다 — 위 "커밋 시 주의" 섹션의 절차를 따라 이 파일만 격리
스테이징할 것.** 나머지 파일(`collector_horse.py`,
`collector_race_detail.py`, `collector_section_records.py`,
`Utils/__init__.py`)은 WIP가 없으므로 평소처럼 `git add`한다.
`git rm`이 이미 `Utils/function.py`의 삭제를 스테이징했다:

```bash
git add library/collector_horse.py library/collector_race_detail.py library/collector_section_records.py Utils/__init__.py
# library/collector_races.py는 격리 스테이징 절차로 별도 추가
git commit -m "refactor: library 4개 파일이 Utils 대신 library.parsing을 쓰도록 전환, Utils/function.py 제거"
```

---

## Task 3: `collector_jockeys.py`의 로컬 중복 정의 제거 (작업 A-3)

**Files:**
- Modify: `library/collector_jockeys.py:65-77`

- [ ] **Step 1: 로컬 `safe_int`/`safe_str` 정의 제거 + import 추가**

`library/collector_jockeys.py`에서 아래 블록을 찾는다(65-77행):

```python
def safe_int(value):
    """안전한 정수 변환"""
    try:
        return int(value) if value else 0
    except (ValueError, TypeError):
        return 0

def safe_str(value):
    """안전한 문자열 변환"""
    try:
        return str(value) if value else ''
    except:
        return ''
```

이 블록을 통째로 삭제한다. 그리고 파일 상단의 import 구역에 다음을
추가한다(기존 import들 바로 아래):

```python
from library.parsing import safe_int, safe_str
```

**주의**: `library/parsing.py`의 `safe_str`은 앞뒤 공백을 제거한다
(`str(value).strip()`)는 점이 기존 로컬 정의(공백 유지)와 다르다 —
스펙에서 이미 검토했듯 의도된 동작 개선이다.

- [ ] **Step 2: 테스트 실행**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_collectors.py -v`
Expected: 관련 테스트 전부 PASS. 실패하는 케이스가 있다면 `safe_str`의
공백 제거 때문에 기대값이 달라진 것인지 확인 — 그렇다면 테스트의
기대값을 공백 제거된 형태로 수정한다(기수명 앞뒤 공백은 제거되는 게
맞는 동작).

- [ ] **Step 3: 전체 테스트 통과 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 4: 커밋**

```bash
git add library/collector_jockeys.py
git commit -m "refactor: collector_jockeys.py의 safe_int/safe_str 로컬 중복 정의 제거"
```

---

## Task 4: `scripts/feature_importance_report.py` → `core/feature_importance_report.py` (작업 B)

**Files:**
- Move: `scripts/feature_importance_report.py` → `core/feature_importance_report.py`
- Modify: `main.py:116`
- Modify: `tests/test_feature_importance_report.py:5`

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv scripts/feature_importance_report.py core/feature_importance_report.py
```

- [ ] **Step 2: 파일 상단 docstring의 실행 예시 수정**

`core/feature_importance_report.py`(이동 후 경로)의 1~6행:

```python
"""
변수 중요도 HTML 리포트 생성.

모델을 재학습할 때마다 다시 실행하면 최신 모델 기준으로 리포트가 갱신된다.
    python -m scripts.feature_importance_report [--version latest]
"""
```

`python -m scripts.feature_importance_report` 부분만 교체:

```python
"""
변수 중요도 HTML 리포트 생성.

모델을 재학습할 때마다 다시 실행하면 최신 모델 기준으로 리포트가 갱신된다.
    python -m core.feature_importance_report [--version latest]
"""
```

(`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` 줄은
`core/`로 옮겨도 여전히 프로젝트 루트를 가리키므로 수정하지 않는다.)

- [ ] **Step 3: `main.py`의 import 수정**

`main.py:116`을 찾아서:

```python
        from scripts.feature_importance_report import generate_report, _load_meta
```

다음으로 교체:

```python
        from core.feature_importance_report import generate_report, _load_meta
```

- [ ] **Step 4: 테스트 파일의 import 수정**

`tests/test_feature_importance_report.py:5`를 찾아서:

```python
from scripts import feature_importance_report as fir
```

다음으로 교체:

```python
from core import feature_importance_report as fir
```

- [ ] **Step 5: 관련 테스트 실행**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_feature_importance_report.py -v`
Expected: 2 passed

- [ ] **Step 6: 전체 테스트 통과 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 7: 커밋**

**`main.py`는 이 계획과 무관한 기존 WIP 수정사항이 있는 파일이다 — 위
"커밋 시 주의" 섹션의 절차를 따라 `main.py`만 격리 스테이징할 것.**
`core/feature_importance_report.py`(이동된 파일)와
`tests/test_feature_importance_report.py`는 WIP가 없으므로 평소처럼
`git add`한다:

```bash
git add core/feature_importance_report.py tests/test_feature_importance_report.py
# main.py는 격리 스테이징 절차로 별도 추가
git commit -m "refactor: scripts/feature_importance_report.py를 core/로 이동(레이어 원칙 위반 해소)"
```

---

## Task 5: `core/odds_collector.py` → `core/select_odds.py` (작업 C)

**Files:**
- Move: `core/odds_collector.py` → `core/select_odds.py`
- Modify: `main.py:208`
- Modify: `core/investor.py:5`

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv core/odds_collector.py core/select_odds.py
```

(클래스명 `OddsCollector`는 그대로 둔다 — 클래스명 변경은 A단계 범위.)

- [ ] **Step 2: `main.py`의 import 수정**

`main.py:208`을 찾아서:

```python
    from core.odds_collector import OddsCollector
```

다음으로 교체:

```python
    from core.select_odds import OddsCollector
```

- [ ] **Step 3: `core/investor.py`의 import 수정**

`core/investor.py:5`를 찾아서:

```python
from core.odds_collector import OddsCollector
```

다음으로 교체:

```python
from core.select_odds import OddsCollector
```

- [ ] **Step 4: 옛 경로가 남아있지 않은지 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core.odds_collector\|core\\odds_collector" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS (`tests/test_investor.py`는 `core.investor`만 import하므로
영향 없음 — 이미 확인됨)

- [ ] **Step 6: 커밋**

**`main.py`와 `core/investor.py` 둘 다 이 계획과 무관한 기존 WIP
수정사항이 있는 파일이다 — 위 "커밋 시 주의" 섹션의 절차를 따라 두
파일 모두 각각 격리 스테이징할 것.** `core/select_odds.py`(이동된
파일)만 평소처럼 `git add`한다:

```bash
git add core/select_odds.py
# main.py, core/investor.py는 격리 스테이징 절차로 각각 별도 추가
git commit -m "refactor: core/odds_collector.py를 core/select_odds.py로 개명(library/scraper_odds.py와의 이름 혼동 해소)"
```

---

## Task 6: `_flatten_entries` 캡슐화 위반 해소 (작업 E)

**Files:**
- Modify: `core/feature_builder.py:72, 110`
- Modify: `scripts/_backtest_common.py:84`

- [ ] **Step 1: `core/feature_builder.py`의 메서드명 변경**

`core/feature_builder.py:110`을 찾아서:

```python
    def _flatten_entries(self, data: list) -> pd.DataFrame:
```

다음으로 교체:

```python
    def get_flatten_entries(self, data: list) -> pd.DataFrame:
```

(메서드 본문은 그대로 둔다 — 이름만 변경.)

- [ ] **Step 2: 내부 호출부 수정**

같은 파일 `core/feature_builder.py:72`를 찾아서:

```python
        df = self._flatten_entries(result.data)
```

다음으로 교체:

```python
        df = self.get_flatten_entries(result.data)
```

- [ ] **Step 3: `scripts/_backtest_common.py`의 호출부 수정**

`scripts/_backtest_common.py:84`를 찾아서:

```python
    return fb._flatten_entries(all_entries)
```

다음으로 교체:

```python
    return fb.get_flatten_entries(all_entries)
```

- [ ] **Step 4: 옛 이름이 남아있지 않은지 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "_flatten_entries" --include=*.py .`
Expected: 결과 없음(private 메서드 완전히 대체됨)

- [ ] **Step 5: feature_builder 테스트 실행**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_feature_builder.py -v`
Expected: 전부 PASS

- [ ] **Step 6: 전체 테스트 통과 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 7: 커밋**

**`scripts/_backtest_common.py`는 이 계획과 무관한 기존 WIP
수정사항이 있는 파일이다 — 위 "커밋 시 주의" 섹션의 절차를 따라 이
파일만 격리 스테이징할 것.** `core/feature_builder.py`는 WIP가 없으므로
평소처럼 `git add`한다:

```bash
git add core/feature_builder.py
# scripts/_backtest_common.py는 격리 스테이징 절차로 별도 추가
git commit -m "refactor: RaceFeatureBuilder._flatten_entries -> get_flatten_entries(공개), scripts의 private 메서드 직접 호출 제거"
```

---

## Task 7: `Utils/kakao_auth.py` → `scripts/kakao_auth.py` (작업 F)

**Files:**
- Move: `Utils/kakao_auth.py` → `scripts/kakao_auth.py`
- Modify: `README.md:43`

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv Utils/kakao_auth.py scripts/kakao_auth.py
```

- [ ] **Step 2: `ENV_PATH`가 여전히 프로젝트 루트를 가리키는지 확인**

`scripts/kakao_auth.py`의 `ENV_PATH` 정의(`Path(__file__).parent.parent /
".env"`)를 확인한다. `Utils/`와 `scripts/`는 둘 다 프로젝트 루트
바로 아래 1단계 디렉토리이므로 `parent.parent`는 이동 후에도 동일하게
프로젝트 루트를 가리킨다 — 코드 수정 불필요. 다음 명령으로 직접
검증한다:

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "from pathlib import Path; p = Path('scripts/kakao_auth.py').resolve().parent.parent / '.env'; print(p)"`
Expected: 프로젝트 루트의 `.env` 경로가 출력됨(예:
`F:\Project\경마 자동 투자 봇\.env`)

- [ ] **Step 3: `README.md`의 실행 안내 수정**

`README.md:43`을 찾아서:

```
카카오톡 알림을 쓰려면 `python -m Utils.kakao_auth`로 토큰을 최초 1회
```

다음으로 교체:

```
카카오톡 알림을 쓰려면 `python -m scripts.kakao_auth`로 토큰을 최초 1회
```

- [ ] **Step 4: import하는 곳이 없는지 재확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "Utils.kakao_auth\|Utils\\kakao_auth\|from Utils import kakao_auth" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 6: 커밋**

```bash
git add scripts/kakao_auth.py README.md
git commit -m "refactor: Utils/kakao_auth.py를 scripts/kakao_auth.py로 이동(1회성 스크립트 정의에 맞춤)"
```

---

## Task 8: 최종 점검

**Files:**
- Read only: 전체 저장소

- [ ] **Step 1: 옛 경로/이름 전수 재확인**

Run:
```bash
cd "F:/Project/경마 자동 투자 봇"
grep -rn "Utils\.function\|Utils import.*safe_\|Utils import.*parse_date\|scripts\.feature_importance_report\|core\.odds_collector\|Utils\.kakao_auth\|_flatten_entries" --include=*.py .
```
Expected: 결과 없음(전부 새 경로/이름으로 치환 완료)

- [ ] **Step 2: 전체 테스트 스위트 최종 실행**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 201 passed (기존 192 + Task 1에서 추가한 9개)

- [ ] **Step 3: `git status`로 누락된 변경 없는지 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && git status --short`
Expected: 이 계획과 무관한 기존 WIP 파일들 외에는 깨끗함(모든 Task가
이미 각자 커밋됨)

---

## 이 계획에 포함되지 않은 것 (다음 단계)

- `docs/architecture.html` 갱신(새 파일 위치/이름 반영) — 이 B단계
  구현이 끝난 뒤 별도 작업으로 진행(스펙의 "다음 단계" 참고).
- D(`odds_collector`/`feature_builder` 배당 매핑 로직 통합) — 회귀
  위험으로 별도 스펙 필요.
- A단계(전체 파일명 동사형 전환) — 별도 브레인스토밍/스펙/계획.
