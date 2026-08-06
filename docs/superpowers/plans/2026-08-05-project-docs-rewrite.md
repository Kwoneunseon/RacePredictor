# 프로젝트 문서 재작성 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `README.md`를 표준 형식(프로젝트 설명+사용법)으로 재작성하고,
상세 구조/파이프라인 문서를 `docs/architecture.html`로 신규 작성한다.

**Architecture:** 정적 파일 2개만 다룬다. 코드 변경은 없다. `docs/architecture.html`은
외부 의존성 없는 단일 HTML(인라인 CSS, 순수 CSS 다이어그램, 라이트/다크
모드 대응, 반응형)이다. `README.md`는 마크다운 표준 섹션 구성.

**Tech Stack:** HTML5, CSS(flexbox, `prefers-color-scheme`), Markdown. 빌드
도구 없음 — 파일을 그대로 브라우저/GitHub에서 연다.

**Spec:** `docs/superpowers/specs/2026-08-05-project-docs-rewrite-design.md`

---

## Task 1: `docs/architecture.html` — 뼈대 + 스타일 시스템 + 섹션 골격

**Files:**
- Create: `docs/architecture.html`

- [ ] **Step 1: 파일 생성**

전체 HTML 뼈대, CSS 변수(라이트/다크), 반응형 기본기, 그리고 이후 태스크가
채울 5개 섹션의 빈 `<section>`(제목만 있음)을 만든다.

```html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>경마 자동 투자 봇 — 아키텍처</title>
<style>
  :root {
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #62626a;
    --card-bg: #f5f5f7;
    --border: #d8d8dc;
    --accent: #2563eb;
    --accent-bg: #eff6ff;
    --warn-bg: #fff7ed;
    --warn-border: #fb923c;
    --code-bg: #f0f0f3;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14151a;
      --fg: #e6e6e9;
      --muted: #9a9aa2;
      --card-bg: #1e2027;
      --border: #33343d;
      --accent: #60a5fa;
      --accent-bg: #1b2536;
      --warn-bg: #2a1f14;
      --warn-border: #f59e0b;
      --code-bg: #1c1d24;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
    line-height: 1.6;
  }
  main {
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem 1.25rem 4rem;
  }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  h1 + p.subtitle { color: var(--muted); margin-top: 0; }
  h2 {
    font-size: 1.2rem;
    margin-top: 3rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }
  code {
    background: var(--code-bg);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    font-size: 0.9em;
  }
  .table-scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; }
  th, td {
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  th { color: var(--muted); font-weight: 600; font-size: 0.85rem; }
  td code { white-space: nowrap; }
</style>
</head>
<body>
<main>
  <h1>경마 자동 투자 봇 — 아키텍처</h1>
  <p class="subtitle">
    코드 파이프라인 구조 스냅샷. 이 문서는 작성 시점의 구조를 담고 있으며,
    구조 정리 작업(경계 재정립·네이밍 정리)이 끝나면 갱신됩니다.
  </p>

  <section id="diagram">
    <h2>1. 파이프라인 개요</h2>
  </section>

  <section id="stages">
    <h2>2. 단계별 설명</h2>
  </section>

  <section id="inventory">
    <h2>3. 모듈 인벤토리</h2>
  </section>

  <section id="migrations">
    <h2>4. DB 마이그레이션 이력</h2>
  </section>

  <section id="issues">
    <h2>5. 알려진 구조적 이슈</h2>
  </section>
</main>
</body>
</html>
```

- [ ] **Step 2: 브라우저에서 열어 기본 레이아웃 확인**

Run: `start docs/architecture.html` (Windows) — 또는 파일 탐색기에서
더블클릭.
Expected: 제목/부제 아래 5개의 빈 섹션 제목(1~5)이 순서대로 보임. OS를
다크 모드로 바꿔도 배경/글자색이 반전되어 읽기 가능해야 함.

- [ ] **Step 3: 커밋**

```bash
git add docs/architecture.html
git commit -m "docs: architecture.html 뼈대 및 스타일 시스템 생성"
```

---

## Task 2: 파이프라인 다이어그램 섹션

**Files:**
- Modify: `docs/architecture.html` (`#diagram` 섹션 내부, Task 1에서 만든
  `<section id="diagram"><h2>...</h2></section>`의 `</h2>` 뒤에 삽입)

- [ ] **Step 1: 다이어그램 CSS 추가**

`</style>` 태그 바로 앞에 추가:

```css
  .pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: stretch;
    margin: 1rem 0;
  }
  .stage {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    flex: 1 1 110px;
    min-width: 110px;
  }
  .stage.entry { border-color: var(--accent); }
  .stage h3 { margin: 0 0 0.35rem; font-size: 0.95rem; }
  .stage p { margin: 0; font-size: 0.8rem; color: var(--muted); }
  .stage code { font-size: 0.75rem; }
  .arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 1.2rem;
    flex: 0 0 auto;
  }
  .cross-layer {
    margin-top: 0.75rem;
    text-align: center;
    background: var(--accent-bg);
    border: 1px dashed var(--accent);
    border-radius: 8px;
    padding: 0.6rem;
    font-size: 0.85rem;
  }
  .exception-note {
    margin-top: 0.75rem;
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.85rem;
  }
  @media (max-width: 700px) {
    .pipeline { flex-direction: column; }
    .arrow { transform: rotate(90deg); }
  }
```

- [ ] **Step 2: 다이어그램 HTML 삽입**

`<section id="diagram"><h2>1. 파이프라인 개요</h2>` 바로 뒤에 추가:

```html
    <div class="pipeline">
      <div class="stage"><h3>수집</h3><p><code>library/</code></p></div>
      <div class="arrow">&rarr;</div>
      <div class="stage"><h3>전처리&middot;피처</h3><p><code>core/preprocessor.py</code><br><code>core/feature_builder.py</code><br><code>core/history_rebuilder.py</code><br><code>core/race_filter.py</code></p></div>
      <div class="arrow">&rarr;</div>
      <div class="stage"><h3>학습&middot;캘리브레이션</h3><p><code>core/trainer.py</code><br><code>core/model_manager.py</code></p></div>
      <div class="arrow">&rarr;</div>
      <div class="stage"><h3>확률 모델</h3><p><code>core/rank_distribution.py</code></p></div>
      <div class="arrow">&rarr;</div>
      <div class="stage"><h3>예측</h3><p><code>core/predictor.py</code></p></div>
      <div class="arrow">&rarr;</div>
      <div class="stage"><h3>베팅</h3><p><code>core/ev_calculator.py</code><br><code>core/odds_collector.py</code><br><code>core/investor.py</code></p></div>
      <div class="arrow">&rarr;</div>
      <div class="stage"><h3>리포트</h3><p><code>core/prediction_report.py</code></p></div>
      <div class="arrow">&rarr;</div>
      <div class="stage entry"><h3>실행 진입점</h3><p><code>main.py</code><br><code>scheduler.py</code></p></div>
    </div>
    <div class="cross-layer">
      <code>Utils/</code> — 전 레이어 공용 (DB 싱글톤&middot;로거&middot;카카오톡 알림)
    </div>
    <div class="exception-note">
      <strong>예외:</strong> <code>scripts/</code>는 원칙적으로 core/library를 소비만 하는
      독립 브랜치이지만, <code>scripts/feature_importance_report.py</code>는
      <code>main.py</code>의 <code>cmd_train()</code>이 직접 import해서 사용합니다
      (레이어 원칙 위반 — 구조 정리 B단계 후보).
    </div>
```

- [ ] **Step 3: 확인**

Run: `start docs/architecture.html`
Expected: 8개 단계 박스가 화살표로 연결되어 가로로 나열되고, 그 아래 파란
점선 박스(Utils)와 주황 경고 박스(scripts 예외)가 순서대로 보임. 브라우저
창을 좁혀 700px 이하로 만들면 박스가 세로로 쌓이고 화살표가 아래쪽을
가리키게 회전함.

- [ ] **Step 4: 커밋**

```bash
git add docs/architecture.html
git commit -m "docs: architecture.html 파이프라인 다이어그램 추가"
```

---

## Task 3: 단계별 설명 섹션

**Files:**
- Modify: `docs/architecture.html` (`#stages` 섹션)

- [ ] **Step 1: 설명 CSS 추가**

`</style>` 앞에 추가:

```css
  .stage-block { margin: 1.25rem 0; }
  .stage-block h3 { margin: 0 0 0.3rem; font-size: 1rem; }
  .stage-block ul { margin: 0.3rem 0 0; padding-left: 1.25rem; }
  .stage-block li { margin: 0.15rem 0; font-size: 0.92rem; }
```

- [ ] **Step 2: 본문 삽입**

`<section id="stages"><h2>2. 단계별 설명</h2>` 뒤에 추가:

```html
    <div class="stage-block">
      <h3>데이터 수집</h3>
      <ul>
        <li><code>library/collector_*.py</code> — 말/기수/조교사/경주/경주결과/구간기록, 공공데이터포털 API</li>
        <li><code>library/scraper_*.py</code> — 배당/마체중, KRA 웹 스크래핑</li>
      </ul>
    </div>
    <div class="stage-block">
      <h3>전처리 &middot; 피처 엔지니어링</h3>
      <ul>
        <li><code>core/preprocessor.py</code> — 라벨 인코딩 등 학습 전처리</li>
        <li><code>core/feature_builder.py</code> — DB에서 피처를 모아 DataFrame으로 변환</li>
        <li><code>core/history_rebuilder.py</code> — <code>horse_race_history</code> 테이블 재구성</li>
        <li><code>core/race_filter.py</code> — 베팅 대상 경주 필터(거리/등급/두수)</li>
      </ul>
    </div>
    <div class="stage-block">
      <h3>학습 &middot; 캘리브레이션</h3>
      <ul>
        <li><code>core/trainer.py</code> — XGBoost/LightGBM top3 분류기+ranker 학습, isotonic 캘리브레이션</li>
        <li><code>core/model_manager.py</code> — 학습된 모델 저장/로드</li>
      </ul>
    </div>
    <div class="stage-block">
      <h3>확률 모델</h3>
      <ul>
        <li><code>core/rank_distribution.py</code> — Harville(Plackett-Luce) 방식으로 top3/복연승 확률 유도</li>
      </ul>
    </div>
    <div class="stage-block">
      <h3>예측</h3>
      <ul>
        <li><code>core/predictor.py</code> — 오늘자 경주 예측 실행, 카카오톡 메시지 포맷 생성</li>
      </ul>
    </div>
    <div class="stage-block">
      <h3>베팅</h3>
      <ul>
        <li><code>core/ev_calculator.py</code> — EV(기대값)/켈리 비율 계산</li>
        <li><code>core/odds_collector.py</code> — 실시간 배당 조회</li>
        <li><code>core/investor.py</code> — 베팅 실행(EV 계산과 배당 조회 연결)</li>
      </ul>
    </div>
    <div class="stage-block">
      <h3>리포트</h3>
      <ul>
        <li><code>core/prediction_report.py</code> — 예측 결과 HTML 리포트 생성</li>
        <li><code>scripts/feature_importance_report.py</code> — 변수 중요도 HTML 리포트(학습 시마다 갱신)</li>
      </ul>
    </div>
    <div class="stage-block">
      <h3>실행 진입점</h3>
      <ul>
        <li><code>main.py</code> — <code>--mode {train,predict,update-results,tune}</code> CLI</li>
        <li><code>scheduler.py</code> — APScheduler: 시간별/평일 06:00 데이터 수집, 매일 10:00 예측+카카오톡 전송</li>
      </ul>
    </div>
```

- [ ] **Step 3: 확인**

Run: `start docs/architecture.html`
Expected: "2. 단계별 설명" 아래 8개 소제목(데이터 수집 ~ 실행 진입점)과 각각의
글머리 목록이 순서대로 보임.

- [ ] **Step 4: 커밋**

```bash
git add docs/architecture.html
git commit -m "docs: architecture.html 단계별 설명 섹션 추가"
```

---

## Task 4: 모듈 인벤토리 표

**Files:**
- Modify: `docs/architecture.html` (`#inventory` 섹션)

- [ ] **Step 1: 표 삽입**

`<section id="inventory"><h2>3. 모듈 인벤토리</h2></section>`의 `</h2>` 뒤에
추가. 내용은 각 파일의 실제 docstring/코드를 조사해 확정한 값(추측 아님):

```html
    <h3>library/</h3>
    <div class="table-scroll">
    <table>
      <tr><th>파일</th><th>책임</th></tr>
      <tr><td><code>collector_horse.py</code></td><td>말 기본정보 수집(공공데이터포털 API)</td></tr>
      <tr><td><code>collector_jockeys.py</code></td><td>기수 정보 수집</td></tr>
      <tr><td><code>collector_trainers.py</code></td><td>조교사 정보 수집</td></tr>
      <tr><td><code>collector_races.py</code></td><td>경주/출전마 정보 수집(달력 기반)</td></tr>
      <tr><td><code>collector_race_detail.py</code></td><td>경주결과종합(부담중량&middot;성적&middot;통과순위) 수집 — API299</td></tr>
      <tr><td><code>collector_section_records.py</code></td><td>마필 구간별 경주기록 수집 — API37_1</td></tr>
      <tr><td><code>scraper_odds.py</code></td><td>현재 단승배당 스크래핑(KRA 경마속보)</td></tr>
      <tr><td><code>scraper_score.py</code></td><td>확정 배당(단승/연승/복승/쌍승/복연승/삼복승/삼쌍승) 스크래핑 — ScoreInfo.do</td></tr>
      <tr><td><code>scraper_weights.py</code></td><td>마체중 스크래핑</td></tr>
      <tr><td><code>__init__.py</code></td><td>패키지 공개 함수 재노출</td></tr>
    </table>
    </div>

    <h3>core/</h3>
    <div class="table-scroll">
    <table>
      <tr><th>파일</th><th>책임</th></tr>
      <tr><td><code>history_rebuilder.py</code></td><td><code>horse_race_history</code> 테이블 재구성</td></tr>
      <tr><td><code>odds_collector.py</code></td><td>실시간 배당 스크래핑(실패 시 DB 중앙값 fallback)</td></tr>
      <tr><td><code>race_filter.py</code></td><td>베팅 대상 경주 필터링(거리/등급/두수 조건)</td></tr>
      <tr><td><code>preprocessor.py</code></td><td>학습용 피처 전처리(라벨 인코딩 등)</td></tr>
      <tr><td><code>feature_builder.py</code></td><td>DB에서 피처를 수집&middot;계산해 DataFrame으로 반환</td></tr>
      <tr><td><code>trainer.py</code></td><td>XGBoost/LightGBM top3 분류기+ranker 학습, isotonic 캘리브레이션</td></tr>
      <tr><td><code>model_manager.py</code></td><td>학습된 모델 저장/로드 관리</td></tr>
      <tr><td><code>rank_distribution.py</code></td><td>Harville 방식 top3/복연승 확률 유도</td></tr>
      <tr><td><code>ev_calculator.py</code></td><td>EV(기대값)/켈리 비율 베팅 계산</td></tr>
      <tr><td><code>investor.py</code></td><td>베팅 판단&middot;기록(<code>betting_records</code> 저장)과 결과 정산</td></tr>
      <tr><td><code>predictor.py</code></td><td>오늘자 경주 예측 실행, 카카오톡 메시지 포맷</td></tr>
      <tr><td><code>prediction_report.py</code></td><td>예측 결과 HTML 리포트 생성</td></tr>
    </table>
    </div>

    <h3>Utils/</h3>
    <div class="table-scroll">
    <table>
      <tr><th>파일</th><th>책임</th></tr>
      <tr><td><code>function.py</code></td><td>공용 헬퍼(날짜 파싱, <code>safe_int</code>/<code>safe_float</code> 등)</td></tr>
      <tr><td><code>logger.py</code></td><td>로거 설정</td></tr>
      <tr><td><code>database.py</code></td><td>Supabase 클라이언트 싱글톤 + <code>upsert</code> 헬퍼</td></tr>
      <tr><td><code>kakaotalk.py</code></td><td>카카오톡 메시지 전송(토큰 만료 시 자동 갱신)</td></tr>
      <tr><td><code>kakao_auth.py</code></td><td>카카오톡 토큰 최초 발급(1회성 스크립트)</td></tr>
    </table>
    </div>

    <h3>scripts/</h3>
    <div class="table-scroll">
    <table>
      <tr><th>파일</th><th>책임</th></tr>
      <tr><td><code>_backtest_common.py</code></td><td>백테스트 공용 로딩/예측 로직</td></tr>
      <tr><td><code>backtest_top3.py</code></td><td>연승(3등 이내) 백테스트</td></tr>
      <tr><td><code>backtest_win.py</code></td><td>단승(1착) 백테스트</td></tr>
      <tr><td><code>backtest_win_tax_adjusted.py</code></td><td>단승 세후(稅後) ROI 재검증</td></tr>
      <tr><td><code>backtest_quinella_place_heuristic.py</code></td><td>복연승 휴리스틱 백테스트(정식 Harville 모델 구현 전 임시 대체)</td></tr>
      <tr><td><code>backfill_place_payouts.py</code></td><td>제주&middot;부산경남 연승 확정배당 소급 백필</td></tr>
      <tr><td><code>diagnose_calibration.py</code></td><td>확률 캘리브레이션(isotonic) 진단</td></tr>
      <tr><td><code>feature_importance_report.py</code></td><td>변수 중요도 HTML 리포트 생성(<code>main.py</code>가 직접 호출)</td></tr>
    </table>
    </div>
```

- [ ] **Step 2: 실제 코드와 대조 검증**

Run: `python -c "import pathlib; print(sorted(p.name for p in pathlib.Path('library').glob('*.py')))"`
Run: `python -c "import pathlib; print(sorted(p.name for p in pathlib.Path('core').glob('*.py')))"`
Run: `python -c "import pathlib; print(sorted(p.name for p in pathlib.Path('Utils').glob('*.py')))"`
Run: `python -c "import pathlib; print(sorted(p.name for p in pathlib.Path('scripts').glob('*.py')))"`
Expected: 각 출력의 파일 목록이 표에 나열된 파일과 정확히 일치(단,
`__init__.py`는 core/scripts에는 없을 수 있음 — library에만 존재).
일치하지 않으면(새 파일이 생겼거나 빠진 경우) 표를 실제 목록에 맞게 수정.

- [ ] **Step 3: 커밋**

```bash
git add docs/architecture.html
git commit -m "docs: architecture.html 모듈 인벤토리 표 추가"
```

---

## Task 5: migrations 이력 + 알려진 구조적 이슈 섹션

**Files:**
- Modify: `docs/architecture.html` (`#migrations`, `#issues` 섹션)

- [ ] **Step 1: 현재 migrations 파일 목록 확인**

Run: `python -c "import pathlib; print(sorted(p.name for p in pathlib.Path('migrations').glob('*.sql')))"`
Expected: `001_*.sql` ~ `004_*.sql` 형태의 4개 파일명 출력(정확한 번호/이름은
출력 결과 그대로 사용 — 지어내지 않음).

- [ ] **Step 2: migrations 섹션 삽입**

`<section id="migrations"><h2>4. DB 마이그레이션 이력</h2></section>`의
`</h2>` 뒤에 추가(파일명은 `migrations/001_init_schema.sql`로 확인됨):

```html
    <p style="color:var(--muted); font-size:0.9rem;">
      스키마 상세는 각 SQL 파일 자체가 source of truth입니다. 여기서는
      목록과 배경만 요약합니다.
    </p>
    <ul>
      <li><code>001_init_schema.sql</code> — 최초 전체 스키마(horses/races/jockeys/trainers/race_entries 등 기본 테이블)</li>
      <li><code>002_retroactive_new_signal_tables.sql</code> — 신호 피처 실험용 테이블(실험은 기각되었으나 재검토 가능성을 남기기 위해 스키마 유지)</li>
      <li><code>003_race_payouts.sql</code> — 경주 확정배당(연승/복승/쌍승/복연승/삼복승/삼쌍승) 저장 테이블</li>
      <li><code>004_betting_records_bet_type_check.sql</code> — <code>betting_records.bet_type</code> 제약조건 정리</li>
    </ul>
```

- [ ] **Step 3: 알려진 구조적 이슈 섹션 삽입**

`<section id="issues"><h2>5. 알려진 구조적 이슈</h2></section>`의 `</h2>`
뒤에 추가:

```html
    <p style="color:var(--muted); font-size:0.9rem;">
      아래 항목은 다음 단계(경계 재정립 &middot; 네이밍 정리)의 입력 자료입니다.
    </p>
    <ul>
      <li><code>scripts/feature_importance_report.py</code>를 <code>main.py</code>가 직접 import — "scripts는 core/library를 소비만 한다"는 레이어 원칙과 반대 방향</li>
      <li><code>Utils/function.py</code> — 이름만으로 내용을 알 수 없는 그랩백(grab-bag) 유틸리티 파일</li>
      <li><code>core/odds_collector.py</code>와 <code>library/scraper_odds.py</code> — 이름이 비슷하지만 레이어와 역할이 다름(전자는 core의 배당 조회 클래스, 후자는 library의 스크래핑 함수) — 새로 온 사람이 혼동하기 쉬움</li>
    </ul>
```

- [ ] **Step 4: 확인**

Run: `start docs/architecture.html`
Expected: "4. DB 마이그레이션 이력"에 4개 항목, "5. 알려진 구조적 이슈"에
3개 항목이 보임. 전체 문서를 위에서 아래로 스크롤하며 레이아웃 깨짐(글자
겹침, 표 삐져나감)이 없는지 확인. 브라우저 개발자도구로 다크모드
에뮬레이션(`Ctrl+Shift+P` → "Rendering" → "prefers-color-scheme: dark")도
확인.

- [ ] **Step 5: 커밋**

```bash
git add docs/architecture.html
git commit -m "docs: architecture.html migrations 이력 및 구조적 이슈 섹션 추가"
```

---

## Task 6: `README.md` 재작성

**Files:**
- Modify: `README.md` (전체 교체)

- [ ] **Step 1: 사용법에 쓸 실제 CLI/환경변수 재확인**

Run: `grep -n "add_argument\|choices=" main.py`
Expected: `--mode`(choices: train/predict/update-results/tune),
`--start-date`, `--end-date`, `--date`, `--meet` 가 출력됨. 이 값과
아래 Step 2의 사용법 예시가 다르면 예시를 실제 출력에 맞게 수정.

Run: `grep -n "os.getenv" config.py`
Expected: `API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `KAKAO_REST_API_KEY`,
`KAKAO_ACCESS_TOKEN`, `MODEL_DIR`, `LOG_DIR` 출력. 이 값과 아래 Step 2의
`.env` 예시가 다르면 수정.

- [ ] **Step 2: README.md 전체 교체**

기존 `README.md` 내용을 아래로 완전히 대체한다:

```markdown
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
```

- [ ] **Step 3: 링크 유효성 확인**

Run: `python -c "import pathlib; print(pathlib.Path('docs/architecture.html').exists())"`
Expected: `True` (Task 1~5가 먼저 완료되어 있어야 함)

- [ ] **Step 4: 커밋**

```bash
git add README.md
git commit -m "docs: README.md를 실제 코드 기준으로 재작성"
```

---

## Task 7: 최종 점검

**Files:**
- Read only: `README.md`, `docs/architecture.html`

- [ ] **Step 1: 링크/앵커 전수 확인**

Run: `grep -n "architecture.html" README.md`
Expected: 2곳(핵심 기능 아래, 프로젝트 구조 아래)에서 링크가 보임.

- [ ] **Step 2: architecture.html 전체를 처음부터 끝까지 브라우저로 재확인**

Run: `start docs/architecture.html`
Expected: 5개 섹션(파이프라인 개요/단계별 설명/모듈 인벤토리/마이그레이션
이력/알려진 구조적 이슈)이 모두 내용으로 채워져 있고 빈 섹션이 없음.

- [ ] **Step 3: README.md를 GitHub 미리보기 스타일로 확인(선택)**

VSCode 등에서 마크다운 프리뷰로 열어 코드블록/표/링크 렌더링이 깨지지
않는지 확인.

- [ ] **Step 4: 최종 커밋(변경 사항이 남아있다면)**

```bash
git status --short
```

수정한 게 있다면 해당 파일만 추가해서 커밋. 없다면 이 태스크는 확인만
하고 종료.

---

## 다음 단계 (이 계획 범위 밖)

이 계획 완료 후 구조 정리의 B단계(`core`/`library`/`Utils`/`scripts` 경계
재정립)를 별도 스펙으로 브레인스토밍한다. B/A 단계가 끝나면 이 문서들
(`README.md`, `docs/architecture.html`)을 새 구조에 맞게 갱신하는 작업을
각 단계의 계획에 포함시킨다.
