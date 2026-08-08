# 구조 정리 B단계 설계: 경계 재정립

## 배경

구조 정리 3단계(C 문서화 → B 경계 재정립 → A 네이밍 정리) 중 두 번째
단계. C단계에서 작성한 `docs/architecture.html`의 "알려진 구조적 이슈"
섹션과, 이번 브레인스토밍을 위해 architecture-expert가 수행한 추가
진단([[project_boundary_reorg_b_phase]] 메모리 참고)을 입력 자료로 삼는다.

architecture-expert 진단 결과, 순환참조·`library`↔`core` 역방향 의존
같은 심각한 문제는 없었고, 아래 5개의 독립적이고 저위험인 이동/개명
작업으로 범위가 좁혀졌다. `odds_collector.py`/`feature_builder.py`의
배당 조회 로직 중복 통합(진단 보고서의 "D" 항목)은 회귀 위험이 있어
이번 B단계 범위에서 제외하고 별도 작업으로 미룬다.

## 범위

| ID | 작업 | 위험도 |
|---|---|---|
| A | `Utils/function.py` → `library/parsing.py` 이동 + `collector_jockeys.py` 중복 제거 + 단위테스트 추가 | 낮음 |
| B | `scripts/feature_importance_report.py` → `core/feature_importance_report.py` 이동 | 낮음~중간 |
| C | `core/odds_collector.py` → `core/select_odds.py` 개명 | 낮음 |
| E | `core/feature_builder.py`에 `get_flatten_entries()` 공개 메서드 추가, `scripts/_backtest_common.py`가 이걸 쓰도록 수정 | 매우 낮음 |
| F | `Utils/kakao_auth.py` → `scripts/kakao_auth.py` 이동 | 매우 낮음 |

A~F는 서로 다른 파일을 건드리므로 순서와 무관하게 진행 가능하다(D는
제외).

## 네이밍 컨벤션 결정 (브레인스토밍 중 확정)

- `select_`: "조회한다"(SQL SELECT 개념) — 여러 후보 중 하나를 "선택"한다는
  뜻이 아니다. 예: `select_odds.py`.
- `choose_`: "여러 옵션 중 하나를 선택한다"는 뜻일 때 사용(이번 범위에는
  해당 사례 없음).
- `get_`: 이미 가진 데이터를 변환/가공해 반환하는 함수에도 폭넓게 사용.
  예: `get_flatten_entries()`.
- 전체 `core`/`library` 파일명을 동사형으로 전면 전환하는 것은 A단계
  범위다. 이번 B단계에서는 지금 이동/개명하는 파일에 한해서만 위 규칙을
  적용한다([[project_a_phase_naming_convention]] 참고).

## A. `Utils/function.py` → `library/parsing.py`

**현황 확인**: `Utils/function.py`는 `parse_date`, `safe_int`, `safe_float`,
`safe_str` 4개 순수 함수만 담고 있다. 실사용처는 아래 4개 파일뿐이며
`core`/`main.py`/`scheduler.py` 어디서도 쓰지 않는다 — "전 레이어 공용"이
아니라 100% `library` 전용이다.

```
library/collector_horse.py:8           from Utils import safe_int, safe_float
library/collector_races.py:7           from Utils import parse_date, safe_int, safe_float, safe_str
library/collector_race_detail.py:12    from Utils import parse_date, safe_int, safe_float, safe_str
library/collector_section_records.py:13 from Utils import parse_date, safe_str
```

`Utils/__init__.py`는 `from .function import *`로 재노출하고 있어 위
4개 파일은 `Utils`(패키지)에서 직접 import하는 형태다.

`library/collector_jockeys.py:65-77`는 이 공용 함수를 쓰지 않고
`safe_int`/`safe_str`을 로컬에 재정의하고 있다(그랩백 파일이 있어도
일관되게 안 쓰인다는 증거):

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

**작업**:
1. `Utils/function.py`를 그대로 `library/parsing.py`로 이동(내용 불변).
2. `Utils/__init__.py`에서 `from .function import *` 줄 삭제(더 이상
   `Utils`가 이 함수들을 재노출하지 않음 — `Utils`는 진짜 전 레이어
   공용만 남긴다는 원칙에 맞춤).
3. `library/__init__.py`는 건드리지 않는다. 아래 4개 파일의 import를
   패키지 재노출이 아니라 **직접 import**로 바꾼다(`from library.parsing
   import ...`) — 어느 함수가 어느 파일에서 오는지 import 문만 보고
   바로 알 수 있게, 이번 기회에 재노출 관습을 하나 줄인다.
4. 아래 4개 파일의 import를 `Utils` → `library.parsing`으로 수정:
   - `library/collector_horse.py:8` `from Utils import safe_int,
     safe_float` → `from library.parsing import safe_int, safe_float`
   - `library/collector_races.py:7` `from Utils import parse_date,
     safe_int, safe_float, safe_str` → `from library.parsing import
     parse_date, safe_int, safe_float, safe_str`
   - `library/collector_race_detail.py:12` (동일 함수 목록, 동일 방식)
   - `library/collector_section_records.py:13` `from Utils import
     parse_date, safe_str` → `from library.parsing import parse_date,
     safe_str`
5. `library/collector_jockeys.py:65-77`의 로컬 `safe_int`/`safe_str`
   정의를 삭제하고 `from library.parsing import safe_int, safe_str`로
   교체한다. 동작 차이를 미리 확인했다:
   - `safe_int`: `collector_jockeys`의 `int(value) if value else 0`과
     `library/parsing.py`의 `int(value) if value and str(value).strip()
     != '' else default`(default=0)는 실질적으로 동일하게 동작한다
     (0/None/빈문자열/공백문자열 전부 0을 반환) — 안전하게 교체 가능.
   - `safe_str`: `collector_jockeys`의 `str(value) if value else ''`은
     공백을 그대로 두지만, `library/parsing.py`의 `str(value).strip()
     if value else default`는 앞뒤 공백을 제거한다. 기수명에 공백이
     섞여 들어오는 경우 결과가 달라지는 **의도적인 동작 개선**이다
     (제거가 더 올바른 방향). 교체 후 `tests/test_collectors.py`의
     관련 테스트를 실행해 실제 데이터에서 문제가 없는지 확인한다.
6. 신규 테스트 `tests/test_parsing.py` 추가 — `parse_date`, `safe_int`,
   `safe_float`, `safe_str` 각각 정상 케이스 + 예외 케이스(빈 문자열,
   `None`, 잘못된 형식) 스모크 테스트. 현재 이 4개 함수는 전용 테스트가
   0개다.

## B. `scripts/feature_importance_report.py` → `core/feature_importance_report.py`

**현황**: `main.py`의 `cmd_train()`이 `scripts/`를 직접 import하는
유일한 사례이며, "scripts는 core/library를 소비만 한다"는 레이어
원칙과 반대 방향이다. 역할이 대칭인 `core/prediction_report.py`(예측
결과 리포트)는 이미 `core/`에 있어 배치 기준이 어긋난다.

**작업**:
1. `scripts/feature_importance_report.py`를
   `core/feature_importance_report.py`로 이동(내용 불변 — 이 파일은
   `argparse` 기반 CLI(`if __name__ == "__main__":`)와 라이브러리 함수가
   한 파일에 섞여 있는데, 이번 B단계에서는 파일 위치만 옮기고 내부
   구조 분리는 하지 않는다 — 범위 밖).
2. `main.py:116`의 import 수정:
   `from scripts.feature_importance_report import generate_report, _load_meta`
   → `from core.feature_importance_report import generate_report, _load_meta`
3. `tests/test_feature_importance_report.py:5`의 import 수정:
   `from scripts import feature_importance_report as fir`
   → `from core import feature_importance_report as fir`
4. 파일 상단 docstring(`scripts/feature_importance_report.py:1-6`)에
   실행 예시가 `python -m scripts.feature_importance_report [--version
   latest]`로 박혀 있다. 이동 후 `python -m core.feature_importance_report
   [--version latest]`로 docstring 자체를 수정한다. 이 파일은
   `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`로
   프로젝트 루트를 수동으로 `sys.path`에 넣는데, `core/`로 옮겨도
   `Path(__file__).resolve().parent.parent`는 여전히 프로젝트 루트를
   가리키므로(위치가 한 단계 아래로 바뀌어도 `parent.parent`는 동일하게
   두 단계 위) 이 로직은 수정할 필요 없다.

## C. `core/odds_collector.py` → `core/select_odds.py`

**현황**: `library/scraper_odds.py`(배당 스크래핑 함수)와 이름이
비슷해 혼동을 유발한다. 실제로는 `library/scraper_odds.py`를 호출하고
실패 시 DB 중앙값으로 폴백하는 "베팅용 배당 조회 파사드"다.

**작업**: 파일명과 그 안의 `OddsCollector` 클래스명은 그대로 두고
**파일 경로만** 바꾼다(클래스명 변경은 A단계 스코프 — 이번엔 파일
위치/이름만).

1. `core/odds_collector.py` → `core/select_odds.py`로 이동.
2. import 수정 2곳:
   - `main.py:208` `from core.odds_collector import OddsCollector` →
     `from core.select_odds import OddsCollector`
   - `core/investor.py:5` `from core.odds_collector import OddsCollector` →
     `from core.select_odds import OddsCollector`
3. 전용 테스트 파일 없음(간접적으로 `tests/test_investor.py`에서만
   참조될 가능성 — 구현 시 grep으로 재확인).

## E. `_flatten_entries` 캡슐화 위반 해소

**현황**: `scripts/_backtest_common.py:84`가
`fb._flatten_entries(all_entries)`로 `core/feature_builder.py`의 private
메서드(`RaceFeatureBuilder._flatten_entries`, `core/feature_builder.py:110-134`)를
직접 호출한다. 이 메서드는 Supabase 조인 결과(중첩 dict)를 평평한
`DataFrame`으로 변환한다.

**작업**:
1. `core/feature_builder.py`의 `RaceFeatureBuilder`에 공개 메서드
   `get_flatten_entries(self, data: list) -> pd.DataFrame`을 추가.
   구현은 기존 `_flatten_entries`를 그대로 감싸거나(`return
   self._flatten_entries(data)`), `_flatten_entries` 자체를
   `get_flatten_entries`로 개명하고 기존 내부 호출부(`core/feature_builder.py:72`의
   `self._flatten_entries(result.data)`)도 새 이름으로 맞춘다(개명 쪽을
   기본으로 한다 — 같은 로직을 두 이름으로 유지할 이유 없음).
2. `scripts/_backtest_common.py:84`를
   `fb.get_flatten_entries(all_entries)`로 수정.

## F. `Utils/kakao_auth.py` → `scripts/kakao_auth.py`

**현황**: `input()`으로 사용자 입력을 받는 1회성 대화형 토큰 발급
스크립트. 코드 어디서도 import하지 않음(오직 `__main__`으로만 실행).
"전 레이어 공용" 계층인 `Utils/`보다 "1회성 스크립트" 계층인 `scripts/`
정의와 정확히 일치한다.

**작업**:
1. `Utils/kakao_auth.py` → `scripts/kakao_auth.py`로 이동.
2. `ENV_PATH`(`Path(__file__).parent.parent / ".env"`, 최근 커밋에서
   `Utils/kakaotalk.py`와 일치하도록 수정된 값)가 새 위치
   (`scripts/kakao_auth.py`)에서도 프로젝트 루트의 `.env`를 정확히
   가리키는지 확인 — `scripts/`도 루트 바로 아래 1단계 디렉토리라
   `parent.parent`면 여전히 정확해야 하지만, 이동 후 반드시 실제 경로
   출력으로 재검증한다.
3. `README.md`의 `python -m Utils.kakao_auth` 안내를
   `python -m scripts.kakao_auth`로 수정.
4. import하는 곳이 없으므로 코드 쪽 추가 수정은 불필요.

## 검증 방법

- 각 하위 작업 완료 후 `pytest` 전체 실행, 192개(+ A에서 추가하는
  `test_parsing.py`) 전부 통과 확인.
- 이동/개명된 각 모듈에 대해 `python -c "import <새 경로>"`로 임포트
  자체가 깨지지 않는지 개별 확인.
- 전체 작업 완료 후 `grep -rn "Utils.function\|Utils import.*safe_\|Utils import.*parse_date\|scripts.feature_importance_report\|core.odds_collector\|Utils.kakao_auth\|_flatten_entries" .` 로
  옛 경로/이름을 참조하는 곳이 남아있지 않은지 전수 확인.

## 이번 단계에서 하지 않는 것

- D(`odds_collector`/`feature_builder`의 배당 조회 매핑 로직 통합) —
  회귀 위험 때문에 별도 스펙으로 분리.
- 파일명 전체를 동사형으로 바꾸는 것 — A단계(네이밍 정리) 범위.
- `scripts/feature_importance_report.py` 내부의 CLI/라이브러리 함수
  분리 — 이번엔 위치만 이동.
- `Utils/database.py`가 직접 import하는 `postgrest._sync.request_builder`
  (private 서브모듈) 문제 — 별도 이슈, 이번 스펙 범위 아님.

## 다음 단계

이 스펙 승인 후 `writing-plans` 스킬로 구현 계획을 만든다. 이 작업
완료 후 `docs/architecture.html`을 새 구조에 맞게 갱신하는 작업을
포함시킨다([[project_horseracing_bot]]의 문서 유지보수 메모 참고).
완료 후에는 A단계(네이밍 정리)를 별도 스펙으로 이어간다.
