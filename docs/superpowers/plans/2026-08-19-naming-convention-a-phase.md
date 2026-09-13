# 구조 정리 A단계(네이밍 정리) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **반드시 태스크 번호 순서대로 실행할 것** — 뒤 태스크의 소비처 파일 이름이 앞 태스크의 완료 여부에 따라 달라진다.

**Goal:** `library/`, `core/`의 21개 파일을 명사형에서 동사형으로 개명하고,
모든 소비처의 import 경로와 대응 테스트 파일명을 함께 갱신한다.

**Architecture:** `git mv`로 파일명만 바꾸고(클래스/함수/로직은 무변경),
각 소비처의 import 문 텍스트를 정확히 치환한다. 대상 파일의 절반가량이
이미 커밋되지 않은 다른 작업(WIP)을 갖고 있어, 그 WIP를 손상시키지 않게
격리하는 절차(절차 R, 절차 I — 아래 정의)를 매 태스크에서 필요할 때마다
적용한다.

**Tech Stack:** Python 3.11, pytest, git.

**Spec:** `docs/superpowers/specs/2026-08-19-naming-convention-a-phase-design.md`

---

## ⚠️ 실행 순서 필수 준수

태스크 순서가 뒤바뀌면 안 된다. 예를 들어 Task 11(`race_filter.py` 개명)이
끝나야 Task 17(`predictor.py` 개명)에서 `core/predictor.py`의 import가
이미 `core.filter_races`로 바뀐 상태임을 전제로 다음 단계를 진행할 수
있다. 각 태스크의 "소비처" 목록은 **그 시점까지 앞 태스크들이 이미
끝났다는 가정** 하에 정확한 현재 파일명으로 적어뒀다.

---

## 공용 절차 정의

### 절차 R — 그 자체가 더러운(WIP 있는) 대상 파일을 개명

`git status --short`에 대상 파일이 `M`으로 표시되는 경우 사용한다.
자기참조 docstring/주석 수정이 필요 없는 파일 기준 기본형:

```bash
cp <OLD_PATH> <OLD_PATH>.bak
git show HEAD:<OLD_PATH> > <OLD_PATH>
git mv <OLD_PATH> <NEW_PATH>
cp <OLD_PATH>.bak <NEW_PATH>
rm <OLD_PATH>.bak
git diff --cached --stat -- <NEW_PATH>
```//마지막 줄로 스테이징된 변경이 순수 rename(내용 diff 0줄)인지 확인

자기참조 docstring/주석도 함께 고쳐야 하는 파일은 아래처럼 4번째 줄
(`git mv` 전)에 편집을 끼워 넣는다:

```bash
cp <OLD_PATH> <OLD_PATH>.bak
git show HEAD:<OLD_PATH> > <OLD_PATH>
# 이 시점에서 되돌려진 <OLD_PATH>에 자기참조 docstring/주석을 직접 편집(아래 각 태스크에 명시된 old_string -> new_string)
git mv <OLD_PATH> <NEW_PATH>
cp <OLD_PATH>.bak <NEW_PATH>
rm <OLD_PATH>.bak
git diff --cached --stat -- <NEW_PATH>
```
이 경우 `git diff --cached`에는 rename + 자기참조 문구만큼의 소량 라인
diff가 함께 보이는 게 정상이다(그 외 다른 라인은 없어야 함 — WIP가
섞이지 않았는지 `git diff --cached -- <NEW_PATH>`로 반드시 확인).

**주의**: `<OLD_PATH>.bak`을 새 경로로 복사하는 이유는, 백업본이
"원래 작업 트리 내용(WIP 포함)"을 담고 있고, 이걸 새 파일명 아래
그대로 복원해야 WIP가 유실되지 않기 때문이다. 백업 복사 후
작업 트리의 `<NEW_PATH>`는 (HEAD+자기참조수정+WIP) 상태가 되고,
스테이징된 내용은 (HEAD+자기참조수정)만 있는 상태가 된다 — 이 차이가
정확히 WIP 라인들이어야 한다.

### 절차 I — 더러운 소비처 파일의 import 한 줄만 수정

대상 파일 자신은 이번 태스크에서 개명하지 않지만(다른 파일을 개명하는
김에 import만 고치는 경우), 그 소비처 파일이 `git status --short`에서
`M`으로 표시되는 경우 사용한다.

```bash
# 1. 현재 라이브 파일에 의도한 편집(old_string -> new_string)을 직접 적용
# 2. 백업
cp <PATH> <PATH>.bak
# 3. HEAD로 되돌림
git show HEAD:<PATH> > <PATH>
# 4. 되돌린 클린 사본에 1번과 동일한 편집을 다시 적용
# 5. 스테이징
git add <PATH>
# 6. 스테이징된 diff가 의도한 한 줄뿐인지 확인
git diff --cached -- <PATH>
# 7. 백업(WIP+편집 모두 포함) 복원
cp <PATH>.bak <PATH>
rm <PATH>.bak
# 8. 스테이징 내용이 작업 트리 복원과 무관하게 그대로인지 재확인
git diff --cached -- <PATH>
```

### 절차 N — 깨끗한 소비처 파일의 import 한 줄 수정

대상 파일이 `git status --short`에 나타나지 않는 경우, 그냥 평소처럼
`old_string`을 `new_string`으로 편집하고 `git add`한다. 격리 절차 불필요.

---

## 현재 WIP 파일 목록 (2026-08-19 기준)

아래 파일들은 이 계획 시작 시점에 이미 `git status --short`에서 `M`으로
표시되어 있다 — 이 목록에 있는 파일을 건드릴 때는 절차 R 또는 절차 I를
쓴다. 목록에 없는 파일은 절차 N(평소처럼)을 쓴다.

```
.vscode/launch.json
core/ev_calculator.py
core/investor.py
core/model_manager.py
core/prediction_report.py
core/predictor.py
core/trainer.py
library/collector_races.py
library/scraper_score.py
main.py
scripts/_backtest_common.py
tests/test_ev_calculator.py
tests/test_model_manager.py
tests/test_prediction_report.py
tests/test_predictor.py
tests/test_scraper_score.py
tests/test_trainer.py
```

각 태스크를 시작하기 전에 `git status --short`를 다시 실행해서 이
목록이 여전히 유효한지 확인할 것(이전 태스크의 커밋으로 목록이 바뀌지
않는다 — 절차 R/I는 항상 원래 WIP를 작업 트리에 그대로 복원하기
때문에, 이 파일들은 계획이 끝날 때까지 계속 `M` 상태로 남아있는 게
정상이다).

---

## Task 1: (A-1 파일럿) `core/rank_distribution.py` → `core/calculate_rank_probabilities.py`

**Files:**
- Move: `core/rank_distribution.py` → `core/calculate_rank_probabilities.py`
- Move: `tests/test_rank_distribution.py` → `tests/test_calculate_rank_probabilities.py`

두 파일 모두 깨끗함(WIP 없음) — 절차 N 범위, 소비처도 테스트 파일 하나뿐.

- [ ] **Step 1: 소스 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv core/rank_distribution.py core/calculate_rank_probabilities.py
```

- [ ] **Step 2: 테스트 파일 이동 + import 수정**

```bash
git mv tests/test_rank_distribution.py tests/test_calculate_rank_probabilities.py
```

`tests/test_calculate_rank_probabilities.py`에서:

```python
from core.rank_distribution import (
```

다음으로 교체:

```python
from core.calculate_rank_probabilities import (
```

- [ ] **Step 3: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.calculate_rank_probabilities"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_calculate_rank_probabilities.py -v`
Expected: 15 passed

- [ ] **Step 4: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.rank_distribution\|core import rank_distribution" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 5: 커밋**

```bash
git add core/calculate_rank_probabilities.py tests/test_calculate_rank_probabilities.py
git commit -m "refactor: core/rank_distribution.py를 core/calculate_rank_probabilities.py로 개명"
```

---

## Task 2: `library/collector_horse.py` → `library/collect_horse.py`

**Files:**
- Move: `library/collector_horse.py` → `library/collect_horse.py`
- Modify: `tests/test_collectors.py` (절차 N — 깨끗함)

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv library/collector_horse.py library/collect_horse.py
```

- [ ] **Step 2: `tests/test_collectors.py` import 수정**

```python
    from library.collector_horse import fetch_horse_data
```
→
```python
    from library.collect_horse import fetch_horse_data
```

그리고:
```python
    from library.collector_horse import save_to_supabase_batch
```
→
```python
    from library.collect_horse import save_to_supabase_batch
```

- [ ] **Step 3: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.collect_horse"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_collectors.py -v`
Expected: 5 passed

- [ ] **Step 4: 커밋**

```bash
git add library/collect_horse.py tests/test_collectors.py
git commit -m "refactor: library/collector_horse.py를 library/collect_horse.py로 개명"
```

---

## Task 3: `library/collector_jockeys.py` → `library/collect_jockeys.py`

**Files:**
- Move: `library/collector_jockeys.py` → `library/collect_jockeys.py`
- Modify: `tests/test_collectors.py` (절차 N)

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv library/collector_jockeys.py library/collect_jockeys.py
```

- [ ] **Step 2: `tests/test_collectors.py` import 수정**

```python
    from library.collector_jockeys import save_to_supabase_batch
```
→
```python
    from library.collect_jockeys import save_to_supabase_batch
```

그리고:
```python
    from library.collector_jockeys import parse_jockey_data
```
→
```python
    from library.collect_jockeys import parse_jockey_data
```

- [ ] **Step 3: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.collect_jockeys"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_collectors.py -v`
Expected: 5 passed

- [ ] **Step 4: 커밋**

```bash
git add library/collect_jockeys.py tests/test_collectors.py
git commit -m "refactor: library/collector_jockeys.py를 library/collect_jockeys.py로 개명"
```

---

## Task 4: `library/collector_trainers.py` → `library/collect_trainers.py`

**Files:**
- Move: `library/collector_trainers.py` → `library/collect_trainers.py`
- Modify: `tests/test_collectors.py` (절차 N)

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv library/collector_trainers.py library/collect_trainers.py
```

- [ ] **Step 2: `tests/test_collectors.py` import 수정**

```python
    from library.collector_trainers import save_to_supabase_batch
```
→
```python
    from library.collect_trainers import save_to_supabase_batch
```

- [ ] **Step 3: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.collect_trainers"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_collectors.py -v`
Expected: 5 passed

- [ ] **Step 4: 커밋**

```bash
git add library/collect_trainers.py tests/test_collectors.py
git commit -m "refactor: library/collector_trainers.py를 library/collect_trainers.py로 개명"
```

---

## Task 5: `library/collector_race_detail.py` → `library/collect_race_detail.py`

**Files:**
- Move: `library/collector_race_detail.py` → `library/collect_race_detail.py`

외부 소비처 없음(자기참조 print문 1곳만 수정) — 절차 N.

- [ ] **Step 1: 자기참조 문구 수정 후 파일 이동**

`library/collector_race_detail.py`에서:
```python
        print("사용법: python -m library.collector_race_detail START END  (YYYY-MM-DD)")
```
→
```python
        print("사용법: python -m library.collect_race_detail START END  (YYYY-MM-DD)")
```

그다음:
```bash
cd "F:/Project/경마 자동 투자 봇"
git mv library/collector_race_detail.py library/collect_race_detail.py
```

- [ ] **Step 2: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.collect_race_detail"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -n "collector_race_detail" library/collect_race_detail.py`
Expected: 결과 없음

- [ ] **Step 3: 커밋**

```bash
git add library/collect_race_detail.py
git commit -m "refactor: library/collector_race_detail.py를 library/collect_race_detail.py로 개명"
```

---

## Task 6: `library/collector_section_records.py` → `library/collect_section_records.py`

**Files:**
- Move: `library/collector_section_records.py` → `library/collect_section_records.py`

외부 소비처 없음 — 절차 N.

- [ ] **Step 1: 자기참조 문구 수정 후 파일 이동**

`library/collector_section_records.py`에서:
```python
        print("사용법: python -m library.collector_section_records START END  (YYYY-MM-DD)")
```
→
```python
        print("사용법: python -m library.collect_section_records START END  (YYYY-MM-DD)")
```

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv library/collector_section_records.py library/collect_section_records.py
```

- [ ] **Step 2: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.collect_section_records"`
Expected: 에러 없음

- [ ] **Step 3: 커밋**

```bash
git add library/collect_section_records.py
git commit -m "refactor: library/collector_section_records.py를 library/collect_section_records.py로 개명"
```

---

## Task 7: `library/scraper_weights.py` → `library/scrape_weights.py`

**Files:**
- Move: `library/scraper_weights.py` → `library/scrape_weights.py`
- Modify: `core/feature_builder.py` (절차 N — 깨끗함)
- Modify: `scheduler.py` (절차 N — 깨끗함)

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv library/scraper_weights.py library/scrape_weights.py
```

- [ ] **Step 2: `core/feature_builder.py` import 수정**

```python
            from library.scraper_weights import fetch_race_weights
```
→
```python
            from library.scrape_weights import fetch_race_weights
```

- [ ] **Step 3: `scheduler.py` import 수정**

```python
        from library.scraper_weights import update_weights_to_db
```
→
```python
        from library.scrape_weights import update_weights_to_db
```

- [ ] **Step 4: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.scrape_weights; import core.feature_builder; import scheduler"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS(단, `tests/test_collector_races_payouts.py`는 B단계에서
이미 기록된 별도 이슈로 실패할 수 있음 — 아래 "알려진 예외" 참고)

- [ ] **Step 5: 커밋**

```bash
git add library/scrape_weights.py core/feature_builder.py scheduler.py
git commit -m "refactor: library/scraper_weights.py를 library/scrape_weights.py로 개명"
```

---

## 알려진 예외: `tests/test_collector_races_payouts.py`

이 테스트는 `library.collector_races`(Task 9 이후 `library.collect_races`)의
`filter_duplicates_in_payouts`를 import하는데, 이 함수는 현재
**커밋된 HEAD에는 없고 작업 트리의 WIP에만 존재**한다(B단계 리뷰에서
이미 발견·기록된 별도 이슈, 이 계획의 범위 밖). 즉:

- 이 계획 전체를 진행하는 동안 **작업 트리에서 pytest를 돌리면 이 테스트는
  통과**한다(WIP가 작업 트리에 남아있으므로).
- 하지만 Task 9(절차 R로 `library/collector_races.py`를 개명)에서
  `git show HEAD:...`로 파일을 되돌리는 순간에는 이 함수가 없는 상태가
  잠깐 나타난다 — 절차 R이 끝나면(백업 복원) 작업 트리에는 다시 함수가
  존재하므로 최종 상태에서는 문제없다.
- 이 계획의 모든 "전체 테스트 통과" 검증은 **작업 트리 기준(WIP 포함)**으로
  본다 — 이게 이 세션 내내 유지해온 방식이다. 새로 커밋된 히스토리만
  체크아웃하면 이 테스트가 실패하는 것은 A단계 시작 전부터 있던 상태이고,
  이 계획으로 고치지 않는다.

---

## Task 8: `library/scraper_odds.py` → `library/scrape_odds.py`

**Files:**
- Move: `library/scraper_odds.py` → `library/scrape_odds.py`
- Modify: `core/feature_builder.py` (절차 N)
- Modify: `core/select_odds.py` (절차 N)
- Modify: `tests/test_feature_builder.py` (절차 N — `patch("library.scraper_odds...")`
  형태의 mock 참조 3곳이 있음, 이 파일 자체는 Task 19에서 개명됨)
- Modify: `tests/test_scraper_odds.py` → rename to `tests/test_scrape_odds.py` (절차 N)

- [ ] **Step 1: 자기참조 docstring 수정 후 파일 이동**

`library/scraper_odds.py`에서:
```python
    from library.scraper_odds import fetch_race_win_odds
```
→
```python
    from library.scrape_odds import fetch_race_win_odds
```

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv library/scraper_odds.py library/scrape_odds.py
```

- [ ] **Step 2: `core/feature_builder.py` import 수정**

```python
import library.scraper_odds as scraper_odds
```
→
```python
import library.scrape_odds as scraper_odds
```

(별칭 `scraper_odds`는 그대로 둔다 — 별칭명 변경은 범위 밖.)

- [ ] **Step 3: `core/select_odds.py` import 수정**

```python
            from library.scraper_odds import fetch_race_win_odds
```
→
```python
            from library.scrape_odds import fetch_race_win_odds
```

- [ ] **Step 4: 테스트 파일 이동 + import 일괄 수정**

```bash
git mv tests/test_scraper_odds.py tests/test_scrape_odds.py
```

`tests/test_scrape_odds.py` 안의 아래 8개 줄 전부(`library.scraper_odds`
→ `library.scrape_odds`, 뒤에 붙는 함수명은 그대로):

```python
    from library.scraper_odds import _parse_odds_table
```
(3회 반복, 동일 텍스트)
```python
    from library.scraper_odds import _try_parse_odds
```
(3회 반복, 동일 텍스트)
```python
    from library.scraper_odds import fetch_race_win_odds
```
(2회 반복, 동일 텍스트)

전부 `library.scraper_odds` → `library.scrape_odds`로 교체(함수명은
그대로 유지). 다음 명령으로 한 번에 치환 후 결과를 확인한다:

```bash
cd "F:/Project/경마 자동 투자 봇"
python -c "
p = 'tests/test_scrape_odds.py'
s = open(p, encoding='utf-8').read()
n = s.count('library.scraper_odds')
s2 = s.replace('library.scraper_odds', 'library.scrape_odds')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced', n)
"
```
Expected 출력: `replaced 10`(`patch("library.scraper_odds.requests.post", ...)`
형태의 mock 패치 대상 문자열 2곳도 포함된 개수 — 이 두 곳도 반드시
함께 바뀌어야 mock이 새 모듈 경로를 정확히 가리킨다)

- [ ] **Step 5: `tests/test_feature_builder.py`의 mock 패치 대상 수정 (절차 N)**

이 파일은 이번 계획에서 개명 대상이 아니지만(Task 19에서 개명됨),
`library.scraper_odds.fetch_race_win_odds`를 문자열로 지정하는
`patch(...)` 호출이 3곳 있어 지금 고쳐두지 않으면 Task 8 커밋 직후
이 파일의 관련 테스트가 깨진다:

```python
    with patch("library.scraper_odds.fetch_race_win_odds", return_value={}):
```
(2회 반복, 동일 텍스트)
```python
    with patch("library.scraper_odds.fetch_race_win_odds", return_value={1: 5.3}):
```

```bash
cd "F:/Project/경마 자동 투자 봇"
python -c "
p = 'tests/test_feature_builder.py'
s = open(p, encoding='utf-8').read()
n = s.count('library.scraper_odds')
s2 = s.replace('library.scraper_odds', 'library.scrape_odds')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced', n)
"
```
Expected 출력: `replaced 3`

- [ ] **Step 6: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.scrape_odds; import core.feature_builder; import core.select_odds"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_scrape_odds.py tests/test_feature_builder.py -v`
Expected: 전부 PASS

- [ ] **Step 7: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "library\.scraper_odds\|library import.*scraper_odds" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 8: 커밋**

```bash
git add library/scrape_odds.py core/feature_builder.py core/select_odds.py tests/test_feature_builder.py tests/test_scrape_odds.py
git commit -m "refactor: library/scraper_odds.py를 library/scrape_odds.py로 개명"
```

---

## Task 9: `library/collector_races.py` → `library/collect_races.py`

**Files:**
- Move(절차 R — WIP 있음): `library/collector_races.py` → `library/collect_races.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 I — WIP 있음): `library/scraper_score.py`
- Modify(절차 I — WIP 있음): `tests/test_scraper_score.py` (`patch("library.collector_races...")`
  형태의 mock 참조 6곳 — 이 파일 자체는 Task 10에서 개명됨)
- Modify(절차 N — 깨끗함): `scripts/backfill_place_payouts.py`
- Modify + rename(절차 N — 깨끗함): `tests/test_collector_races_payouts.py` → `tests/test_collect_races_payouts.py`

- [ ] **Step 1: `git status --short`로 WIP 목록 재확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && git status --short`
Expected: `library/collector_races.py`, `main.py`, `library/scraper_score.py`가
여전히 `M`으로 표시됨(위 "현재 WIP 파일 목록" 참고)

- [ ] **Step 2: `library/collector_races.py` 절차 R로 개명**

자기참조 주석도 함께 고친다. 절차 R의 "HEAD로 되돌린 뒤" 단계에서
아래 두 줄을 편집:

```python
    # 사용법: python -m library.collector_races [시작일 종료일]
    # 예)    python -m library.collector_races 20240101 20241231
```
→
```python
    # 사용법: python -m library.collect_races [시작일 종료일]
    # 예)    python -m library.collect_races 20240101 20241231
```

전체 절차:
```bash
cd "F:/Project/경마 자동 투자 봇"
cp library/collector_races.py library/collector_races.py.bak
git show HEAD:library/collector_races.py > library/collector_races.py
```
이제 `library/collector_races.py`(되돌려진 HEAD 상태)를 열어 위 두 줄을
편집한다. 그다음:
```bash
git mv library/collector_races.py library/collect_races.py
cp library/collector_races.py.bak library/collect_races.py
rm library/collector_races.py.bak
git diff --cached --stat -- library/collect_races.py
```
Expected: rename으로 표시되고 content diff는 2줄(주석) 정도만 있어야 함

- [ ] **Step 3: `main.py` 절차 I로 import 수정**

```python
        from library.collector_races import collecting_race_results
```
→
```python
        from library.collect_races import collecting_race_results
```

- [ ] **Step 4: `library/scraper_score.py` 절차 I로 import 수정**

```python
    from library.collector_races import fetch_race_data_all, parse_and_normalize_race_data
```
→
```python
    from library.collect_races import fetch_race_data_all, parse_and_normalize_race_data
```

- [ ] **Step 5: `scripts/backfill_place_payouts.py` 절차 N으로 import 수정**

```python
from library.collector_races import (
    fetch_race_data_all,
    parse_and_normalize_race_data,
    filter_duplicates_in_payouts,
    PLC_ODDS_SUPPORTED_MEET_CODES,
)
```
→
```python
from library.collect_races import (
    fetch_race_data_all,
    parse_and_normalize_race_data,
    filter_duplicates_in_payouts,
    PLC_ODDS_SUPPORTED_MEET_CODES,
)
```

- [ ] **Step 6: `tests/test_scraper_score.py` 절차 I로 mock 패치 대상 수정**

이 파일은 `library.collector_races.fetch_race_data_all`을 문자열로
지정하는 `patch(...)` 호출이 6곳 있다. 절차 I를 적용하되, "의도한
편집"은 아래 일괄 치환 스크립트로 한다:

```python
    with patch("library.collector_races.fetch_race_data_all", return_value=(api_resp, 1)), \
```
(형태가 조금씩 다른 6곳 모두 `library.collector_races.fetch_race_data_all`
문자열을 포함 — 전부 `library.collect_races.fetch_race_data_all`로 교체)

절차 I 1번(라이브 파일에 편집 적용) 단계에서:
```bash
cd "F:/Project/경마 자동 투자 봇"
python -c "
p = 'tests/test_scraper_score.py'
s = open(p, encoding='utf-8').read()
n = s.count('library.collector_races')
s2 = s.replace('library.collector_races', 'library.collect_races')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced', n)
"
```
Expected 출력: `replaced 6`

이후 절차 I의 나머지 단계(백업, HEAD 리셋, 클린 사본에 동일 치환
재적용, 스테이징, `git diff --cached`로 6곳뿐인지 확인, 백업 복원,
재확인)를 그대로 수행한다.

- [ ] **Step 7: 테스트 파일 이동 + import 수정 (절차 N)**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv tests/test_collector_races_payouts.py tests/test_collect_races_payouts.py
```

`tests/test_collect_races_payouts.py`에서:
```python
from library.collector_races import (
    parse_and_normalize_race_data,
    filter_duplicates_in_payouts,
    PLC_ODDS_SUPPORTED_MEET_CODES,
)
```
→
```python
from library.collect_races import (
    parse_and_normalize_race_data,
    filter_duplicates_in_payouts,
    PLC_ODDS_SUPPORTED_MEET_CODES,
)
```

- [ ] **Step 8: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.collect_races; import main; import library.scraper_score"`
Expected: 에러 없음(작업 트리 기준 — WIP가 복원돼 있으므로
`filter_duplicates_in_payouts` 등이 존재함)

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_scraper_score.py -v`
Expected: 전부 PASS

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS(작업 트리 기준. "알려진 예외" 섹션 참고 — 이 시점엔
WIP가 복원돼 있으므로 실제로는 통과해야 함)

- [ ] **Step 9: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "library\.collector_races\|library import.*collector_races" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 10: 커밋**

```bash
git add library/collect_races.py tests/test_collect_races_payouts.py scripts/backfill_place_payouts.py
git commit -m "refactor: library/collector_races.py를 library/collect_races.py로 개명"
```

**주의**: `main.py`, `library/scraper_score.py`, `tests/test_scraper_score.py`는
절차 I로 이미 스테이징되어 있으므로 이 커밋에 자동으로 포함된다(위
Step 3/4/6에서 `git add`까지 절차 I 안에서 수행함). 별도로 다시
`git add`할 필요 없음 — 단, 절차 I의 마지막 단계(백업 복원 후
`git diff --cached` 재확인)까지 빠짐없이 수행했는지 재차 확인할 것.

---

## Task 10: `library/scraper_score.py` → `library/scrape_score.py`

**Files:**
- Move(절차 R — WIP 있음): `library/scraper_score.py` → `library/scrape_score.py`
- Modify(절차 I — WIP 있음): `main.py`
- Move + Modify(절차 R — WIP 있음): `tests/test_scraper_score.py` → `tests/test_scrape_score.py`

- [ ] **Step 1: `library/scraper_score.py` 절차 R로 개명**

자기참조 docstring도 함께 고친다:

```python
사용:
    from library.scraper_score import update_results_to_db
    updated = update_results_to_db(meet_code=3, race_date="2026-06-05")

    from library.scraper_score import update_race_payouts_to_db
    updated = update_race_payouts_to_db(meet_code=3, race_date="2026-06-05")  # race_payouts 테이블 필요
```
→
```python
사용:
    from library.scrape_score import update_results_to_db
    updated = update_results_to_db(meet_code=3, race_date="2026-06-05")

    from library.scrape_score import update_race_payouts_to_db
    updated = update_race_payouts_to_db(meet_code=3, race_date="2026-06-05")  # race_payouts 테이블 필요
```

```bash
cd "F:/Project/경마 자동 투자 봇"
cp library/scraper_score.py library/scraper_score.py.bak
git show HEAD:library/scraper_score.py > library/scraper_score.py
```
되돌려진 파일에 위 docstring 편집을 적용한 뒤:
```bash
git mv library/scraper_score.py library/scrape_score.py
cp library/scraper_score.py.bak library/scrape_score.py
rm library/scraper_score.py.bak
git diff --cached --stat -- library/scrape_score.py
```

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
    from library.scraper_score import update_all_meets, update_all_final_ranks
```
→
```python
    from library.scrape_score import update_all_meets, update_all_final_ranks
```

- [ ] **Step 3: 테스트 파일 절차 R로 개명 + import 수정**

`tests/test_scraper_score.py`가 WIP 목록에 있으므로 절차 R을 쓴다.
HEAD로 되돌린 뒤, import를 수정한다:

```python
from library.scraper_score import (
    _parse_circled,
    _parse_odds_groups,
    _entry_numbers_from_circled,
    BET_TYPE_CELL_INDEX,
)
```
→
```python
from library.scrape_score import (
    _parse_circled,
    _parse_odds_groups,
    _entry_numbers_from_circled,
    BET_TYPE_CELL_INDEX,
)
```

그리고 파일 안의 나머지 지연 import(`from library.scraper_score import
...`) 전부를 `from library.scrape_score import ...`로 바꾼다(함수명은
그대로). 아래 전체 절차에서 일괄 치환한다.

전체 절차(절차 R 확장 — HEAD로 되돌린 사본과 WIP 백업본 양쪽에 동일
치환을 적용):
```bash
cd "F:/Project/경마 자동 투자 봇"
cp tests/test_scraper_score.py tests/test_scraper_score.py.bak
git show HEAD:tests/test_scraper_score.py > tests/test_scraper_score.py
python -c "
p = 'tests/test_scraper_score.py'
s = open(p, encoding='utf-8').read()
n = s.count('library.scraper_score')
s2 = s.replace('library.scraper_score', 'library.scrape_score')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced(HEAD copy)', n)
"
git mv tests/test_scraper_score.py tests/test_scrape_score.py
# WIP 백업본에도 동일 치환을 적용해서 새 경로에 복원
python -c "
p = 'tests/test_scraper_score.py.bak'
s = open(p, encoding='utf-8').read()
n = s.count('library.scraper_score')
s2 = s.replace('library.scraper_score', 'library.scrape_score')
open('tests/test_scrape_score.py', 'w', encoding='utf-8').write(s2)
print('replaced(WIP backup)', n)
"
rm tests/test_scraper_score.py.bak
git diff --cached --stat -- tests/test_scrape_score.py
```
Expected: `replaced(HEAD copy) 10`, `replaced(WIP backup) 19`(WIP가 추가한
테스트 케이스만큼 HEAD보다 많다 — 실행 시점에 숫자가 다르게 나오면
`git show HEAD:tests/test_scraper_score.py`와 현재 작업 트리 파일을
직접 비교해 원인을 먼저 파악할 것, 무시하고 진행하지 말 것)

- [ ] **Step 4: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import library.scrape_score; import main"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_scrape_score.py -v`
Expected: 전부 PASS

- [ ] **Step 5: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "library\.scraper_score" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 6: 전체 테스트**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 7: 커밋**

```bash
git add library/scrape_score.py tests/test_scrape_score.py
git commit -m "refactor: library/scraper_score.py를 library/scrape_score.py로 개명"
```

(`main.py`는 Step 2의 절차 I에서 이미 스테이징됨.)

---

## Task 11: `core/race_filter.py` → `core/filter_races.py`

**Files:**
- Move(절차 N — 깨끗함): `core/race_filter.py` → `core/filter_races.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 I — WIP 있음): `core/predictor.py`
- Move + Modify(절차 N — 깨끗함): `tests/test_race_filter.py` → `tests/test_filter_races.py`

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv core/race_filter.py core/filter_races.py
```

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
    from core.race_filter import RaceFilter
```
→
```python
    from core.filter_races import RaceFilter
```

- [ ] **Step 3: `core/predictor.py` 절차 I로 import 수정**

```python
    from core.race_filter import RaceFilter
```
→
```python
    from core.filter_races import RaceFilter
```

- [ ] **Step 4: 테스트 파일 이동 + import 수정 (절차 N)**

```bash
git mv tests/test_race_filter.py tests/test_filter_races.py
```

```python
from core.race_filter import RaceFilter
```
→
```python
from core.filter_races import RaceFilter
```

- [ ] **Step 5: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.filter_races; import core.predictor; import main"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_filter_races.py -v`
Expected: 전부 PASS

- [ ] **Step 6: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.race_filter" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 7: 커밋**

```bash
git add core/filter_races.py tests/test_filter_races.py
git commit -m "refactor: core/race_filter.py를 core/filter_races.py로 개명"
```

(`main.py`, `core/predictor.py`는 Step 2/3의 절차 I에서 이미 스테이징됨.)

---

## Task 12: `core/history_rebuilder.py` → `core/rebuild_history.py`

**Files:**
- Move(절차 N — 깨끗함): `core/history_rebuilder.py` → `core/rebuild_history.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 I — WIP 있음): `.vscode/launch.json`
- Move + Modify(절차 N — 깨끗함): `tests/test_history_rebuilder.py` → `tests/test_rebuild_history.py`

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv core/history_rebuilder.py core/rebuild_history.py
```

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
        from core.history_rebuilder import rebuild
```
→
```python
        from core.rebuild_history import rebuild
```

- [ ] **Step 3: `.vscode/launch.json` 절차 I로 경로 수정**

```json
            "program": "${workspaceFolder}/core/history_rebuilder.py",
```
→
```json
            "program": "${workspaceFolder}/core/rebuild_history.py",
```

- [ ] **Step 4: 테스트 파일 이동 + import 수정 (절차 N)**

```bash
git mv tests/test_history_rebuilder.py tests/test_rebuild_history.py
```

```python
from core.history_rebuilder import compute_history
```
→
```python
from core.rebuild_history import compute_history
```

- [ ] **Step 5: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.rebuild_history; import main"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_rebuild_history.py -v`
Expected: 전부 PASS

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import json; json.load(open('.vscode/launch.json', encoding='utf-8'))"`
Expected: 에러 없음(JSON 문법 손상 여부 확인)

- [ ] **Step 6: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.history_rebuilder\|core/history_rebuilder" . --include=*.py --include=*.json`
Expected: 결과 없음

- [ ] **Step 7: 커밋**

```bash
git add core/rebuild_history.py tests/test_rebuild_history.py
git commit -m "refactor: core/history_rebuilder.py를 core/rebuild_history.py로 개명"
```

(`main.py`, `.vscode/launch.json`은 Step 2/3의 절차 I에서 이미 스테이징됨.)

---

## Task 13: `core/feature_importance_report.py` → `core/report_feature_importance.py`

**Files:**
- Move(절차 N — 깨끗함): `core/feature_importance_report.py` → `core/report_feature_importance.py`
- Modify(절차 I — WIP 있음): `main.py`
- Move + Modify(절차 N — 깨끗함): `tests/test_feature_importance_report.py` → `tests/test_report_feature_importance.py`

- [ ] **Step 1: 자기참조 docstring 수정 후 파일 이동**

`core/feature_importance_report.py`에서:
```python
    python -m core.feature_importance_report [--version latest]
```
→
```python
    python -m core.report_feature_importance [--version latest]
```

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv core/feature_importance_report.py core/report_feature_importance.py
```

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
        from core.feature_importance_report import generate_report, _load_meta
```
→
```python
        from core.report_feature_importance import generate_report, _load_meta
```

- [ ] **Step 3: 테스트 파일 이동 + import 수정 (절차 N)**

```bash
git mv tests/test_feature_importance_report.py tests/test_report_feature_importance.py
```

```python
from core import feature_importance_report as fir
```
→
```python
from core import report_feature_importance as fir
```

- [ ] **Step 4: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.report_feature_importance; import main"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_report_feature_importance.py -v`
Expected: 2 passed

- [ ] **Step 5: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.feature_importance_report\|core import feature_importance_report" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 6: 커밋**

```bash
git add core/report_feature_importance.py tests/test_report_feature_importance.py
git commit -m "refactor: core/feature_importance_report.py를 core/report_feature_importance.py로 개명"
```

(`main.py`는 Step 2의 절차 I에서 이미 스테이징됨.)

---

## Task 14: `core/ev_calculator.py` → `core/calculate_ev.py`

**Files:**
- Move(절차 R — WIP 있음): `core/ev_calculator.py` → `core/calculate_ev.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 I — WIP 있음): `core/investor.py` (import 한 줄만 — 파일 자체 개명은 Task 15)
- Modify(절차 I — WIP 있음): `core/prediction_report.py` (import 한 줄만 — 파일 자체 개명은 Task 16)
- Move + Modify(절차 R — WIP 있음): `tests/test_ev_calculator.py` → `tests/test_calculate_ev.py`

- [ ] **Step 1: `core/ev_calculator.py` 절차 R로 개명**

자기참조 없음(별도 docstring 편집 불필요) — 기본형 절차 R:

```bash
cd "F:/Project/경마 자동 투자 봇"
cp core/ev_calculator.py core/ev_calculator.py.bak
git show HEAD:core/ev_calculator.py > core/ev_calculator.py
git mv core/ev_calculator.py core/calculate_ev.py
cp core/ev_calculator.py.bak core/calculate_ev.py
rm core/ev_calculator.py.bak
git diff --cached --stat -- core/calculate_ev.py
```
Expected: 순수 rename(내용 diff 0줄)

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
from core.ev_calculator import BET_TYPE_LABELS_KO
```
→
```python
from core.calculate_ev import BET_TYPE_LABELS_KO
```

- [ ] **Step 3: `core/investor.py` 절차 I로 import 수정**

```python
from core.ev_calculator import EVCalculator
```
→
```python
from core.calculate_ev import EVCalculator
```

- [ ] **Step 4: `core/prediction_report.py` 절차 I로 import 수정**

```python
from core.ev_calculator import BET_TYPE_LABELS_KO
```
→
```python
from core.calculate_ev import BET_TYPE_LABELS_KO
```

- [ ] **Step 5: 테스트 파일 절차 R로 개명 + import 수정**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp tests/test_ev_calculator.py tests/test_ev_calculator.py.bak
git show HEAD:tests/test_ev_calculator.py > tests/test_ev_calculator.py
```
되돌려진 사본에서:
```python
from core.ev_calculator import EVCalculator, BetRecommendation
```
→
```python
from core.calculate_ev import EVCalculator, BetRecommendation
```
그다음:
```bash
git mv tests/test_ev_calculator.py tests/test_calculate_ev.py
```
백업본에도 동일 치환 적용 후 새 경로로 복원:
```bash
python -c "
p = 'tests/test_ev_calculator.py.bak'
s = open(p, encoding='utf-8').read()
s2 = s.replace('from core.ev_calculator import', 'from core.calculate_ev import')
open('tests/test_calculate_ev.py', 'w', encoding='utf-8').write(s2)
"
rm tests/test_ev_calculator.py.bak
git diff --cached --stat -- tests/test_calculate_ev.py
```

- [ ] **Step 6: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.calculate_ev; import core.investor; import core.prediction_report; import main"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_calculate_ev.py -v`
Expected: 전부 PASS

- [ ] **Step 7: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.ev_calculator" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 8: 커밋**

```bash
git add core/calculate_ev.py tests/test_calculate_ev.py
git commit -m "refactor: core/ev_calculator.py를 core/calculate_ev.py로 개명"
```

(`main.py`, `core/investor.py`, `core/prediction_report.py`는 Step 2~4의
절차 I에서 이미 스테이징됨.)

---

## Task 15: `core/investor.py` → `core/decide_bets.py`

**Files:**
- Move(절차 R — WIP 있음): `core/investor.py` → `core/decide_bets.py`
- Modify(절차 I — WIP 있음): `main.py`
- Move + Modify(절차 N — 깨끗함): `tests/test_investor.py` → `tests/test_decide_bets.py`

- [ ] **Step 1: `core/investor.py` 절차 R로 개명**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp core/investor.py core/investor.py.bak
git show HEAD:core/investor.py > core/investor.py
git mv core/investor.py core/decide_bets.py
cp core/investor.py.bak core/decide_bets.py
rm core/investor.py.bak
git diff --cached --stat -- core/decide_bets.py
```
Expected: 순수 rename(Task 14에서 이미 커밋된 `core.calculate_ev` import는
HEAD에 포함돼 있으므로 diff 0줄)

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
    from core.investor import Investor
```
→
```python
    from core.decide_bets import Investor
```

- [ ] **Step 3: 테스트 파일 이동 + import 수정 (절차 N)**

```bash
git mv tests/test_investor.py tests/test_decide_bets.py
```

```python
from core.investor import Investor
```
→
```python
from core.decide_bets import Investor
```

- [ ] **Step 4: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.decide_bets; import main"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_decide_bets.py -v`
Expected: 전부 PASS

- [ ] **Step 5: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.investor\b" --include=*.py .`
Expected: 결과 없음(단, `self.odds_collector`처럼 변수/속성명에 우연히
포함된 문자열은 대상 아님 — 이 grep 패턴은 `core.investor`라는 import
경로 문자열만 잡으므로 해당 없음)

- [ ] **Step 6: 커밋**

```bash
git add core/decide_bets.py tests/test_decide_bets.py
git commit -m "refactor: core/investor.py를 core/decide_bets.py로 개명"
```

(`main.py`는 Step 2의 절차 I에서 이미 스테이징됨.)

---

## Task 16: `core/prediction_report.py` → `core/report_predictions.py`

**Files:**
- Move(절차 R — WIP 있음): `core/prediction_report.py` → `core/report_predictions.py`
- Modify(절차 I — WIP 있음): `main.py`
- Move + Modify(절차 R — WIP 있음): `tests/test_prediction_report.py` → `tests/test_report_predictions.py`

- [ ] **Step 1: `core/prediction_report.py` 절차 R로 개명**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp core/prediction_report.py core/prediction_report.py.bak
git show HEAD:core/prediction_report.py > core/prediction_report.py
git mv core/prediction_report.py core/report_predictions.py
cp core/prediction_report.py.bak core/report_predictions.py
rm core/prediction_report.py.bak
git diff --cached --stat -- core/report_predictions.py
```
Expected: 순수 rename

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
        from core.prediction_report import save_report
```
→
```python
        from core.report_predictions import save_report
```

- [ ] **Step 3: 테스트 파일 절차 R로 개명(WIP 있음)**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp tests/test_prediction_report.py tests/test_prediction_report.py.bak
git show HEAD:tests/test_prediction_report.py > tests/test_prediction_report.py
```
**주의**: 이 테스트 파일은 `from core.prediction_report import X` 형태가
아니라 `from core import prediction_report`처럼 모듈 자체를 이름으로
import하고, 이후 `prediction_report.render_html(...)`,
`prediction_report.save_report(...)` 형태로 17곳에서 참조한다. 따라서
`core.prediction_report`라는 부분 문자열 자체가 파일 안에 존재하지
않는다(단순 치환 스크립트로는 아무것도 안 바뀐다). import 문 한 줄만
아래처럼 별칭을 붙여 고치면 나머지 17곳은 전혀 손댈 필요가 없다:

되돌려진 사본에서:
```python
from core import prediction_report
```
→
```python
from core import report_predictions as prediction_report
```

(이 파일 안의 `feature_importance_report` 테스트가 이미 `from core
import report_feature_importance as fir`처럼 별칭을 쓰는 것과 같은
방식 — Task 13에서 이미 적용한 패턴이다.)

```bash
git mv tests/test_prediction_report.py tests/test_report_predictions.py
```
백업본에도 동일 치환을 적용해 새 경로로 복원:
```bash
python -c "
p = 'tests/test_prediction_report.py.bak'
s = open(p, encoding='utf-8').read()
n = s.count('from core import prediction_report')
s2 = s.replace('from core import prediction_report', 'from core import report_predictions as prediction_report')
open('tests/test_report_predictions.py', 'w', encoding='utf-8').write(s2)
print('replaced', n)
"
```
Expected: `replaced 1`

```bash
rm tests/test_prediction_report.py.bak
git diff --cached --stat -- tests/test_report_predictions.py
```

- [ ] **Step 4: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.report_predictions; import main"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_report_predictions.py -v`
Expected: 전부 PASS

- [ ] **Step 5: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.prediction_report" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 6: 커밋**

```bash
git add core/report_predictions.py tests/test_report_predictions.py
git commit -m "refactor: core/prediction_report.py를 core/report_predictions.py로 개명"
```

(`main.py`는 Step 2의 절차 I에서 이미 스테이징됨.)

---

## Task 17: `core/predictor.py` → `core/predict_races.py`

**Files:**
- Move(절차 R — WIP 있음): `core/predictor.py` → `core/predict_races.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 N — 깨끗함): `scheduler.py`
- Move + Modify(절차 R — WIP 있음): `tests/test_predictor.py` → `tests/test_predict_races.py`

- [ ] **Step 1: `core/predictor.py` 절차 R로 개명**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp core/predictor.py core/predictor.py.bak
git show HEAD:core/predictor.py > core/predictor.py
git mv core/predictor.py core/predict_races.py
cp core/predictor.py.bak core/predict_races.py
rm core/predictor.py.bak
git diff --cached --stat -- core/predict_races.py
```
Expected: 순수 rename(Task 11에서 이미 커밋된 `core.filter_races`
import가 HEAD에 포함돼 있으므로 diff 0줄)

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
from core.predictor import RacePredictor
```
→
```python
from core.predict_races import RacePredictor
```

- [ ] **Step 3: `scheduler.py` 절차 N으로 import 수정**

```python
    from core.predictor import predict_today, format_kakao_message
```
→
```python
    from core.predict_races import predict_today, format_kakao_message
```

- [ ] **Step 4: 테스트 파일 절차 R로 개명**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp tests/test_predictor.py tests/test_predictor.py.bak
git show HEAD:tests/test_predictor.py > tests/test_predictor.py
```
되돌려진 사본에서 `core.predictor` 참조 3곳(1개는 파일 상단, 2개는
함수 내부 지연 import)을 전부 `core.predict_races`로 교체:
```bash
python -c "
p = 'tests/test_predictor.py'
s = open(p, encoding='utf-8').read()
n = s.count('core.predictor')
s2 = s.replace('core.predictor', 'core.predict_races')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced(HEAD copy)', n)
"
git mv tests/test_predictor.py tests/test_predict_races.py
```
백업본에도 동일 치환 적용 후 새 경로로 복원:
```bash
python -c "
p = 'tests/test_predictor.py.bak'
s = open(p, encoding='utf-8').read()
s2 = s.replace('core.predictor', 'core.predict_races')
open('tests/test_predict_races.py', 'w', encoding='utf-8').write(s2)
"
rm tests/test_predictor.py.bak
git diff --cached --stat -- tests/test_predict_races.py
```

- [ ] **Step 5: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.predict_races; import main; import scheduler"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_predict_races.py -v`
Expected: 전부 PASS

- [ ] **Step 6: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.predictor\b" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 7: 커밋**

```bash
git add core/predict_races.py tests/test_predict_races.py scheduler.py
git commit -m "refactor: core/predictor.py를 core/predict_races.py로 개명"
```

(`main.py`는 Step 2의 절차 I에서 이미 스테이징됨.)

---

## Task 18: `core/preprocessor.py` → `core/preprocess_features.py`

**Files:**
- Move(절차 N — 깨끗함): `core/preprocessor.py` → `core/preprocess_features.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 I — WIP 있음): `scripts/_backtest_common.py`
- Modify(절차 N — 깨끗함): `core/report_feature_importance.py`
- Modify(절차 N — 깨끗함): `scripts/backtest_win_tax_adjusted.py`
- Modify(절차 N — 깨끗함): `scripts/diagnose_calibration.py`
- Move + Modify(절차 N — 깨끗함): `tests/test_preprocessor.py` → `tests/test_preprocess_features.py`

`core/predict_races.py`는 `RacePreprocessor`를 직접 import하지 않고
생성자 인자로만 받으므로(호출부인 `main.py`/`scheduler.py`가 인스턴스를
만들어 넘김) 이 태스크의 소비처가 아니다 — 건드리지 않는다.

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv core/preprocessor.py core/preprocess_features.py
```

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
from core.preprocessor import RacePreprocessor
```
→
```python
from core.preprocess_features import RacePreprocessor
```

- [ ] **Step 3: `scripts/_backtest_common.py` 절차 I로 import 수정**

```python
from core.preprocessor import RacePreprocessor
```
→
```python
from core.preprocess_features import RacePreprocessor
```

- [ ] **Step 4: `core/report_feature_importance.py` 절차 N으로 import 수정**

```python
from core.preprocessor import RacePreprocessor
```
→
```python
from core.preprocess_features import RacePreprocessor
```

- [ ] **Step 5: `scripts/backtest_win_tax_adjusted.py` 절차 N으로 import 수정**

```python
from core.preprocessor import RacePreprocessor
```
→
```python
from core.preprocess_features import RacePreprocessor
```

- [ ] **Step 6: `scripts/diagnose_calibration.py` 절차 N으로 import 수정**

```python
from core.preprocessor import RacePreprocessor
```
→
```python
from core.preprocess_features import RacePreprocessor
```

- [ ] **Step 7: 테스트 파일 이동 + import 수정 (절차 N)**

```bash
git mv tests/test_preprocessor.py tests/test_preprocess_features.py
```

```python
from core.preprocessor import RacePreprocessor
```
→
```python
from core.preprocess_features import RacePreprocessor
```

- [ ] **Step 8: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.preprocess_features; import main; import scripts._backtest_common; import core.report_feature_importance"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_preprocess_features.py -v`
Expected: 전부 PASS

- [ ] **Step 9: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.preprocessor" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 10: 전체 테스트**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 11: 커밋**

```bash
git add core/preprocess_features.py tests/test_preprocess_features.py core/report_feature_importance.py scripts/backtest_win_tax_adjusted.py scripts/diagnose_calibration.py
git commit -m "refactor: core/preprocessor.py를 core/preprocess_features.py로 개명"
```

(`main.py`, `scripts/_backtest_common.py`는 Step 2/3의 절차 I에서 이미
스테이징됨.)

---

## Task 19: `core/feature_builder.py` → `core/build_features.py`

**Files:**
- Move(절차 N — 깨끗함): `core/feature_builder.py` → `core/build_features.py`
- Modify(절차 I — WIP 있음): `main.py` (3곳)
- Modify(절차 I — WIP 있음): `scripts/_backtest_common.py`
- Modify(절차 I — WIP 있음): `core/predict_races.py`
- Modify(절차 N — 깨끗함): `scripts/backtest_win_tax_adjusted.py`
- Modify(절차 N — 깨끗함): `scripts/diagnose_calibration.py`
- Move + Modify(절차 N — 깨끗함): `tests/test_feature_builder.py` → `tests/test_build_features.py`

**참고**: `core/predict_races.py`는 Task 17에서 개명됐지만, Task 17의
원본 WIP(구 `core/predictor.py`의 11줄 diff)를 그대로 이어받아 여전히
`git status --short`에 `M`으로 나타난다 — 절차 I 대상이다.

- [ ] **Step 1: 파일 이동**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv core/feature_builder.py core/build_features.py
```

- [ ] **Step 2: `main.py` 절차 I로 import 3곳 일괄 수정**

`main.py`에는 아래와 동일한 텍스트가 3곳(지연 import)에 있다:

```python
        from core.feature_builder import RaceFeatureBuilder
```

절차 I의 "1번: 라이브 파일에 편집 적용" 단계에서 아래 명령으로 3곳을
한 번에 치환한다:

```bash
cd "F:/Project/경마 자동 투자 봇"
python -c "
p = 'main.py'
s = open(p, encoding='utf-8').read()
n = s.count('from core.feature_builder import RaceFeatureBuilder')
s2 = s.replace('from core.feature_builder import RaceFeatureBuilder', 'from core.build_features import RaceFeatureBuilder')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced', n)
"
```
Expected: `replaced 3`

절차 I의 나머지 단계(백업, HEAD 리셋, 클린 사본에 동일 치환 재적용,
스테이징, 검증, 백업 복원, 재검증)를 그대로 수행한다 — "클린 사본에
재적용"할 때도 위와 동일한 python 치환 명령을 HEAD로 되돌린 직후의
`main.py`에 대해 실행하면 된다.

- [ ] **Step 3: `scripts/_backtest_common.py` 절차 I로 import 수정**

```python
from core.feature_builder import RaceFeatureBuilder
```
→
```python
from core.build_features import RaceFeatureBuilder
```

- [ ] **Step 4: `core/predict_races.py` 절차 I로 import 수정**

```python
from core.feature_builder import RaceFeatureBuilder
```
→
```python
from core.build_features import RaceFeatureBuilder
```

- [ ] **Step 5: `scripts/backtest_win_tax_adjusted.py` 절차 N으로 import 수정**

```python
from core.feature_builder import RaceFeatureBuilder
```
→
```python
from core.build_features import RaceFeatureBuilder
```

- [ ] **Step 6: `scripts/diagnose_calibration.py` 절차 N으로 import 수정**

```python
from core.feature_builder import RaceFeatureBuilder
```
→
```python
from core.build_features import RaceFeatureBuilder
```

- [ ] **Step 7: 테스트 파일 이동 + import/patch 참조 일괄 수정 (절차 N)**

```bash
cd "F:/Project/경마 자동 투자 봇"
git mv tests/test_feature_builder.py tests/test_build_features.py
python -c "
p = 'tests/test_build_features.py'
s = open(p, encoding='utf-8').read()
n = s.count('core.feature_builder')
s2 = s.replace('core.feature_builder', 'core.build_features')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced', n)
"
```
Expected: `replaced 21`(`patch("core.feature_builder.get_client", ...)`
형태의 mock 패치 대상 문자열 4곳도 포함된 개수)

- [ ] **Step 8: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.build_features; import main; import scripts._backtest_common; import core.predict_races"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_build_features.py -v`
Expected: 전부 PASS

- [ ] **Step 9: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.feature_builder" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 10: 전체 테스트**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 11: 커밋**

```bash
git add core/build_features.py tests/test_build_features.py scripts/backtest_win_tax_adjusted.py scripts/diagnose_calibration.py
git commit -m "refactor: core/feature_builder.py를 core/build_features.py로 개명"
```

(`main.py`, `scripts/_backtest_common.py`, `core/predict_races.py`는
Step 2~4의 절차 I에서 이미 스테이징됨.)

---

## Task 20: `core/trainer.py` → `core/train_models.py`

**Files:**
- Move(절차 R — WIP 있음): `core/trainer.py` → `core/train_models.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 I — WIP 있음): `scripts/_backtest_common.py`
- Modify(절차 N — 깨끗함): `core/report_feature_importance.py`
- Modify(절차 N — 깨끗함): `scripts/backtest_win_tax_adjusted.py`
- Modify(절차 N — 깨끗함): `scripts/diagnose_calibration.py`
- Move + Modify(절차 R — WIP 있음): `tests/test_trainer.py` → `tests/test_train_models.py`

- [ ] **Step 1: `core/trainer.py` 절차 R로 개명**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp core/trainer.py core/trainer.py.bak
git show HEAD:core/trainer.py > core/trainer.py
git mv core/trainer.py core/train_models.py
cp core/trainer.py.bak core/train_models.py
rm core/trainer.py.bak
git diff --cached --stat -- core/train_models.py
```
Expected: 순수 rename

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
from core.trainer import RaceModelTrainer
```
→
```python
from core.train_models import RaceModelTrainer
```

- [ ] **Step 3: `scripts/_backtest_common.py` 절차 I로 import 수정**

```python
from core.trainer import RaceModelTrainer
```
→
```python
from core.train_models import RaceModelTrainer
```

- [ ] **Step 4: `core/report_feature_importance.py` 절차 N으로 import 수정**

```python
from core.trainer import RaceModelTrainer
```
→
```python
from core.train_models import RaceModelTrainer
```

- [ ] **Step 5: `scripts/backtest_win_tax_adjusted.py` 절차 N으로 import 수정**

```python
from core.trainer import RaceModelTrainer, CALIBRATION_EVAL_CUTOFF
```
→
```python
from core.train_models import RaceModelTrainer, CALIBRATION_EVAL_CUTOFF
```

- [ ] **Step 6: `scripts/diagnose_calibration.py` 절차 N으로 import 수정**

```python
from core.trainer import RaceModelTrainer, CALIBRATION_EVAL_CUTOFF
```
→
```python
from core.train_models import RaceModelTrainer, CALIBRATION_EVAL_CUTOFF
```

- [ ] **Step 7: 테스트 파일 절차 R로 개명**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp tests/test_trainer.py tests/test_trainer.py.bak
git show HEAD:tests/test_trainer.py > tests/test_trainer.py
```
되돌려진 사본에서 `core.trainer` 참조(파일 상단 1곳 + 함수 내부 지연
import 1곳, 총 2곳)를 `core.train_models`로 교체:
```bash
python -c "
p = 'tests/test_trainer.py'
s = open(p, encoding='utf-8').read()
n = s.count('core.trainer')
s2 = s.replace('core.trainer', 'core.train_models')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced(HEAD copy)', n)
"
git mv tests/test_trainer.py tests/test_train_models.py
```
백업본에도 동일 치환 적용 후 새 경로로 복원:
```bash
python -c "
p = 'tests/test_trainer.py.bak'
s = open(p, encoding='utf-8').read()
s2 = s.replace('core.trainer', 'core.train_models')
open('tests/test_train_models.py', 'w', encoding='utf-8').write(s2)
"
rm tests/test_trainer.py.bak
git diff --cached --stat -- tests/test_train_models.py
```

- [ ] **Step 8: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.train_models; import main; import scripts._backtest_common; import core.report_feature_importance"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_train_models.py -v`
Expected: 전부 PASS

- [ ] **Step 9: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.trainer\b" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 10: 전체 테스트**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 11: 커밋**

```bash
git add core/train_models.py tests/test_train_models.py core/report_feature_importance.py scripts/backtest_win_tax_adjusted.py scripts/diagnose_calibration.py
git commit -m "refactor: core/trainer.py를 core/train_models.py로 개명"
```

(`main.py`, `scripts/_backtest_common.py`는 Step 2/3의 절차 I에서 이미
스테이징됨.)

---

## Task 21: `core/model_manager.py` → `core/manage_models.py`

**Files:**
- Move(절차 R — WIP 있음): `core/model_manager.py` → `core/manage_models.py`
- Modify(절차 I — WIP 있음): `main.py`
- Modify(절차 I — WIP 있음): `scripts/_backtest_common.py`
- Modify(절차 N — 깨끗함): `core/report_feature_importance.py`
- Modify(절차 N — 깨끗함): `scripts/backtest_win_tax_adjusted.py`
- Modify(절차 N — 깨끗함): `scripts/diagnose_calibration.py`
- Move + Modify(절차 R — WIP 있음): `tests/test_model_manager.py` → `tests/test_manage_models.py`

이번이 A-4의 마지막이자 이 계획 전체에서 개명하는 마지막 파일이다.

- [ ] **Step 1: `core/model_manager.py` 절차 R로 개명**

```bash
cd "F:/Project/경마 자동 투자 봇"
cp core/model_manager.py core/model_manager.py.bak
git show HEAD:core/model_manager.py > core/model_manager.py
git mv core/model_manager.py core/manage_models.py
cp core/model_manager.py.bak core/manage_models.py
rm core/model_manager.py.bak
git diff --cached --stat -- core/manage_models.py
```
Expected: 순수 rename

- [ ] **Step 2: `main.py` 절차 I로 import 수정**

```python
from core.model_manager import ModelManager
```
→
```python
from core.manage_models import ModelManager
```

- [ ] **Step 3: `scripts/_backtest_common.py` 절차 I로 import 수정**

```python
from core.model_manager import ModelManager
```
→
```python
from core.manage_models import ModelManager
```

- [ ] **Step 4: `core/report_feature_importance.py` 절차 N으로 import 수정**

```python
from core.model_manager import ModelManager
```
→
```python
from core.manage_models import ModelManager
```

- [ ] **Step 5: `scripts/backtest_win_tax_adjusted.py` 절차 N으로 import 수정**

```python
from core.model_manager import ModelManager
```
→
```python
from core.manage_models import ModelManager
```

- [ ] **Step 6: `scripts/diagnose_calibration.py` 절차 N으로 import 수정**

```python
from core.model_manager import ModelManager
```
→
```python
from core.manage_models import ModelManager
```

- [ ] **Step 7: 테스트 파일 절차 R로 개명**

이 파일은 `from core.model_manager import ModelManager` 외에도
`mocker.patch("core.model_manager.MODEL_DIR", tmp_path)` 형태의 mock
패치 대상 문자열이 4곳 더 있다 — 반드시 `core.model_manager` 전체를
부분 문자열로 일괄 치환해야 한다(단순 import 줄 하나만 바꾸면 나머지
4곳이 깨진다).

```bash
cd "F:/Project/경마 자동 투자 봇"
cp tests/test_model_manager.py tests/test_model_manager.py.bak
git show HEAD:tests/test_model_manager.py > tests/test_model_manager.py
python -c "
p = 'tests/test_model_manager.py'
s = open(p, encoding='utf-8').read()
n = s.count('core.model_manager')
s2 = s.replace('core.model_manager', 'core.manage_models')
open(p, 'w', encoding='utf-8').write(s2)
print('replaced(HEAD copy)', n)
"
git mv tests/test_model_manager.py tests/test_manage_models.py
```
백업본에도 동일 치환을 적용해 새 경로로 복원:
```bash
python -c "
p = 'tests/test_model_manager.py.bak'
s = open(p, encoding='utf-8').read()
n = s.count('core.model_manager')
s2 = s.replace('core.model_manager', 'core.manage_models')
open('tests/test_manage_models.py', 'w', encoding='utf-8').write(s2)
print('replaced(WIP backup)', n)
"
rm tests/test_model_manager.py.bak
git diff --cached --stat -- tests/test_manage_models.py
```
Expected: 두 `replaced` 출력 모두 5(import 1곳 + patch 4곳)

- [ ] **Step 8: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import core.manage_models; import main; import scripts._backtest_common; import core.report_feature_importance"`
Expected: 에러 없음

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest tests/test_manage_models.py -v`
Expected: 전부 PASS

- [ ] **Step 9: 옛 이름 잔존 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -rn "core\.model_manager" --include=*.py .`
Expected: 결과 없음

- [ ] **Step 10: 전체 테스트**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 11: 커밋**

```bash
git add core/manage_models.py tests/test_manage_models.py core/report_feature_importance.py scripts/backtest_win_tax_adjusted.py scripts/diagnose_calibration.py
git commit -m "refactor: core/model_manager.py를 core/manage_models.py로 개명"
```

(`main.py`, `scripts/_backtest_common.py`는 Step 2/3의 절차 I에서 이미
스테이징됨.)

---

## Task 22: `docs/architecture.html` + `.claude/agents/architecture-expert.md` 갱신

**Files:**
- Modify: `docs/architecture.html`
- Modify: `.claude/agents/architecture-expert.md`

Task 1~21에서 개명된 21개 파일의 새 이름을 두 문서에 전부 반영한다.

- [ ] **Step 1: `docs/architecture.html`의 파이프라인 다이어그램 갱신**

`<code>` 태그로 감싼 아래 옛 파일명들을 새 이름으로 교체(다이어그램
박스, 단계별 설명, 모듈 인벤토리 표, 알려진 구조적 이슈 섹션 전부
포함):

| 옛 이름 | 새 이름|
|---|---|
| `collector_horse.py` | `collect_horse.py` |
| `collector_jockeys.py` | `collect_jockeys.py` |
| `collector_trainers.py` | `collect_trainers.py` |
| `collector_races.py` | `collect_races.py` |
| `collector_race_detail.py` | `collect_race_detail.py` |
| `collector_section_records.py` | `collect_section_records.py` |
| `scraper_odds.py` | `scrape_odds.py` |
| `scraper_score.py` | `scrape_score.py` |
| `scraper_weights.py` | `scrape_weights.py` |
| `race_filter.py` | `filter_races.py` |
| `history_rebuilder.py` | `rebuild_history.py` |
| `ev_calculator.py` | `calculate_ev.py` |
| `investor.py` | `decide_bets.py` |
| `predictor.py` | `predict_races.py` |
| `preprocessor.py` | `preprocess_features.py` |
| `feature_builder.py` | `build_features.py` |
| `trainer.py` | `train_models.py` |
| `model_manager.py` | `manage_models.py` |
| `prediction_report.py` | `report_predictions.py` |
| `feature_importance_report.py` | `report_feature_importance.py` |
| `rank_distribution.py` | `calculate_rank_probabilities.py` |

또한 "1. 파이프라인 개요"의 `library/`(경로만 표기된 박스)는 그대로
두고, 각 단계 박스 안 개별 파일명(`core/preprocessor.py` 등)을 위
표에 따라 치환한다.

"5. 알려진 구조적 이슈" 섹션에 다음 문구를 추가한다(맨 위, 기존
"B단계 완료" 안내 바로 아래):

```html
    <p class="note">
      2026-08-19 A단계(네이밍 정리) 완료로 core/library의 21개 파일이
      명사형에서 동사형으로 개명됨(예: feature_builder.py &rarr; build_features.py).
      클래스명(OddsCollector, RaceFeatureBuilder 등)과 메서드명은 이번
      단계에서 변경하지 않았다 — 파일명(동작 축)과 클래스명(행위자 축)을
      의도적으로 분리하는 것으로 확정.
    </p>
```

- [ ] **Step 2: `.claude/agents/architecture-expert.md`의 레이어 맵 갱신**

`## 이 프로젝트의 실제 레이어 구조` 아래 코드 블록 안의 모든 파일명을
위 표에 따라 교체한다(`library/`, `core/` 섹션 전부).

- [ ] **Step 3: 검증**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack=[]
    def handle_starttag(self,tag,attrs):
        if tag not in ('meta','br','link','img','hr'):
            self.stack.append(tag)
    def handle_endtag(self,tag):
        if self.stack and self.stack[-1]==tag:
            self.stack.pop()
p=P()
with open('docs/architecture.html', encoding='utf-8') as f:
    p.feed(f.read())
print('remaining stack:', p.stack)
"`
Expected: `remaining stack: []`

Run: `cd "F:/Project/경마 자동 투자 봇" && grep -c "collector_\|scraper_\|race_filter\|history_rebuilder\|ev_calculator\|prediction_report\.py\|feature_importance_report\.py\|rank_distribution\|>predictor\.py\|>preprocessor\.py\|>feature_builder\.py\|>trainer\.py\|>model_manager\.py\|>investor\.py" docs/architecture.html`
Expected: `0`

- [ ] **Step 4: 커밋**

```bash
git add docs/architecture.html .claude/agents/architecture-expert.md
git commit -m "docs: architecture.html/architecture-expert.md를 A단계(네이밍 정리) 완료 상태로 갱신"
```

---

## Task 23: 최종 점검

**Files:**
- Read only: 전체 저장소

- [ ] **Step 1: 옛 이름 전수 재확인**

Run:
```bash
cd "F:/Project/경마 자동 투자 봇"
grep -rn "collector_horse\|collector_jockeys\|collector_trainers\|collector_races\|collector_race_detail\|collector_section_records\|scraper_odds\|scraper_score\|scraper_weights\|core\.race_filter\|core\.history_rebuilder\|core\.ev_calculator\|core\.investor\b\|core\.predictor\b\|core\.preprocessor\|core\.feature_builder\|core\.trainer\b\|core\.model_manager\|core\.prediction_report\|core\.feature_importance_report\|core\.rank_distribution" --include=*.py .
```
Expected: 결과 없음

- [ ] **Step 2: 21개 새 파일이 전부 존재하는지 확인**

Run:
```bash
cd "F:/Project/경마 자동 투자 봇"
python -c "
import pathlib
expected_library = ['collect_horse.py','collect_jockeys.py','collect_trainers.py','collect_races.py','collect_race_detail.py','collect_section_records.py','scrape_odds.py','scrape_score.py','scrape_weights.py','parsing.py','__init__.py']
expected_core = ['filter_races.py','rebuild_history.py','calculate_ev.py','decide_bets.py','predict_races.py','preprocess_features.py','build_features.py','train_models.py','manage_models.py','report_predictions.py','report_feature_importance.py','calculate_rank_probabilities.py','select_odds.py']
actual_library = sorted(p.name for p in pathlib.Path('library').glob('*.py'))
actual_core = sorted(p.name for p in pathlib.Path('core').glob('*.py'))
print('library missing:', set(expected_library) - set(actual_library))
print('library extra:', set(actual_library) - set(expected_library))
print('core missing:', set(expected_core) - set(actual_core))
print('core extra:', set(actual_core) - set(expected_core))
"
```
Expected: 4줄 전부 빈 집합(`set()`)

- [ ] **Step 3: 전체 테스트 스위트 최종 실행**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -m pytest -q`
Expected: 전부 PASS("알려진 예외" 섹션에 기록된, 이 계획 시작 전부터
있던 이슈 외에 새로운 실패가 없어야 함)

- [ ] **Step 4: 주요 진입점 임포트 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && python -c "import main; import scheduler; print('OK')"`
Expected: `OK`

- [ ] **Step 5: `git status`로 누락된 변경 없는지 확인**

Run: `cd "F:/Project/경마 자동 투자 봇" && git status --short`
Expected: "현재 WIP 파일 목록" 섹션에 나열된 파일들(및 그 파일들이
개명된 새 경로)만 `M`으로 남아있어야 함 — 이 계획과 무관한 새로운
미스테이징 변경이 없어야 함

---

## 이 계획에 포함되지 않은 것

- 클래스명 변경(`OddsCollector`, `RaceFeatureBuilder`, `ModelManager`,
  `EVCalculator`, `Investor`, `RacePredictor`, `RacePreprocessor`,
  `RaceFilter` 등) — 스펙에서 명시적으로 범위 제외.
- 메서드/함수 레벨 네이밍 정리(`collecting_race_results` 등 동명사형
  함수명) — 별도 후속 작업.
- B단계에서 발견된 별도 이슈(배당 조회 로직 중복, 쿼리 로직 중복,
  `collector_trainers.py`/현 `collect_trainers.py`의 데이터 유실 버그,
  `tests/test_collect_races_payouts.py`의 커밋 상태 실패 문제) — 전부
  범위 밖, `docs/architecture.html`에 이미 기록됨.

## 다음 단계

이 계획 완료로 구조 정리 3단계(C 문서화 → B 경계 재정립 → A 네이밍
정리) 전체가 끝난다. 이후 작업은 별도로 브레인스토밍한다.
