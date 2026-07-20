"""
변수 중요도 HTML 리포트 생성.

모델을 재학습할 때마다 다시 실행하면 최신 모델 기준으로 리포트가 갱신된다.
    python -m scripts.feature_importance_report [--version latest]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import MODEL_DIR
from core.model_manager import ModelManager
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "feature_importance.html"

FEATURE_LABELS_KO = {
    "horse_age": "말 나이",
    "horse_class_num": "말 등급(숫자화)",
    "horse_weight": "말 체중",
    "race_distance": "경주 거리",
    "race_grade_enc": "경주 등급(인코딩)",
    "jockey_win_rate": "기수 통산 승률",
    "trainer_win_rate": "조교사 통산 승률",
    "horse_win_rate": "말 통산 승률",
    "horse_top3_rate": "말 통산 연승률(3착 이내)",
    "prev_total_races": "통산 출전 횟수",
    "prev_5_avg_rank": "최근 5경주 평균 순위",
    "prev_total_avg_rank": "통산 평균 순위",
    "horse_class_vs_mean": "경주 내 평균 대비 등급 우위",
    "win_rate_vs_mean": "경주 내 평균 대비 승률 우위",
    "jockey_recent_win_rate": "기수 최근 30일 승률",
    "trainer_recent_win_rate": "조교사 최근 30일 승률",
    "training_grade_enc": "조교 등급(인코딩)",
    "days_since_last_race": "직전 경주 이후 경과일",
    "horse_weight_diff": "직전 대비 체중 변화량",
    "horse_win_rate_at_distance": "해당 거리 승률",
    "horse_track_win_rate": "해당 경마장 승률",
    "jockey_horse_win_rate": "기수-말 조합 승률",
    "jockey_track_win_rate": "기수 해당 경마장 승률",
    "prev_speed_figure": "직전 경주 속도(거리/기록시간)",
    "speed_figure_trend": "최근 3경주 속도 추세",
    "horse_speed_at_distance": "해당 거리 과거 평균 속도",
    "recent_rank_trend": "최근 순위 추세",
    "weight_trend": "체중 변화 추세",
    "entry_number": "출발 게이트 번호",
    "prev_win_odds": "직전 경주 단승 배당",
    "recent_form": "최근 폼 지수(5경주 평균순위 기반)",
    "experience_score": "경험치 점수(통산 출전 로그변환)",
    "track_condition_enc": "마장 상태(인코딩)",
    "weather_enc": "날씨(인코딩)",
    "is_male": "성별(수컷 여부)",
    "implied_prob_vs_mean": "경주 내 평균 대비 배당 내재확률",
    "race_implied_prob_rank": "경주 내 배당 내재확률 순위(인기순)",
}


def _importance(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float)).reshape(-1)
    else:
        return None
    total = raw.sum()
    return raw / total if total > 0 else raw


def _load_meta(version: str) -> dict:
    if version == "latest":
        files = sorted(MODEL_DIR.glob("model_*_meta.json"))
        if not files:
            return {}
        path = files[-1]
    else:
        path = MODEL_DIR / f"model_{version}_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_series(trainer: RaceModelTrainer) -> dict[str, dict[str, np.ndarray]]:
    win_models = {
        "RandomForest": trainer.models.get("RandomForest"),
        "LogisticRegression": trainer.models.get("LogisticRegression"),
        "XGBoost (Ranker)": trainer.models.get("XGBoost"),
        "LightGBM (Ranker)": trainer.models.get("LightGBM"),
    }
    top3_models = {
        "RandomForest": trainer.top3_models.get("RandomForest"),
        "LogisticRegression": trainer.top3_models.get("LogisticRegression"),
        "XGBoost (Clf)": trainer.top3_models.get("XGBoostClf"),
        "LightGBM (Clf)": trainer.top3_models.get("LightGBMClf"),
        "XGBoost (Ranker)": trainer.top3_ranker_models.get("XGBoost"),
        "LightGBM (Ranker)": trainer.top3_ranker_models.get("LightGBM"),
    }

    groups = {"단승 (Win)": win_models, "연승 (Top3)": top3_models}
    result: dict[str, dict[str, np.ndarray]] = {}
    for group_label, models in groups.items():
        group_series = {}
        for name, model in models.items():
            if model is None:
                continue
            imp = _importance(model)
            if imp is None:
                continue
            group_series[name] = imp
        if group_series:
            avg = np.mean(list(group_series.values()), axis=0)
            group_series = {"평균": avg, **group_series}
        result[group_label] = group_series
    return result


def render_html(feature_columns: list[str], series: dict, meta: dict, metrics: dict) -> str:
    payload = {
        "features": feature_columns,
        "labels": {f: FEATURE_LABELS_KO.get(f, f) for f in feature_columns},
        "series": {
            group: {name: vals.tolist() for name, vals in models.items()}
            for group, models in series.items()
        },
        "meta": meta,
        "metrics": metrics,
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>변수 중요도 리포트</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  .viz-root {{
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --series-1-track: #cde2fb;
    --chip-bg: #f0efec;
  }}
  @media (prefers-color-scheme: dark) {{
    .viz-root {{
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-1-track: #184f95;
      --chip-bg: #26261f;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #898781; --gridline: #2c2c2a;
    --baseline: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-1-track: #184f95; --chip-bg: #26261f;
  }}
  :root[data-theme="light"] .viz-root {{
    --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --text-muted: #898781; --gridline: #e1e0d9;
    --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6; --series-1-track: #cde2fb; --chip-bg: #f0efec;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .viz-root {{ max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }}

  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }}

  .meta-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }}
  .chip {{
    background: var(--chip-bg); color: var(--text-secondary);
    border-radius: 999px; padding: 5px 12px; font-size: 12px;
  }}

  .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 20px; }}
  select {{
    font: inherit; font-size: 13px; padding: 7px 10px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary);
  }}
  button.tablebtn {{
    font: inherit; font-size: 13px; padding: 7px 12px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
    cursor: pointer;
  }}
  button.tablebtn:hover {{ color: var(--text-primary); }}
  button.tablebtn.active {{ color: var(--series-1); border-color: var(--series-1); }}

  .card {{
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 20px 8px;
  }}

  .bar-row {{ margin-bottom: 10px; }}
  .bar-label-line {{
    display: flex; justify-content: space-between; font-size: 12.5px;
    color: var(--text-secondary); margin-bottom: 4px;
  }}
  .bar-label-line .rank {{ color: var(--text-muted); margin-right: 6px; }}
  .bar-label-line .value {{ color: var(--text-primary); font-variant-numeric: tabular-nums; }}
  .code {{ color: var(--text-muted); font-size: 11.5px; font-family: ui-monospace, Consolas, monospace; }}
  td.code {{ color: var(--text-muted); }}
  .bar-track {{
    position: relative; height: 16px; border-radius: 4px;
    background: var(--series-1-track); overflow: hidden;
  }}
  .bar-fill {{
    height: 100%; border-radius: 4px; background: var(--series-1);
    transition: width .25s ease;
  }}
  .bar-row.top .bar-label-line .value {{ color: var(--series-1); font-weight: 600; }}

  table.data-table {{
    width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 12px;
  }}
  table.data-table th, table.data-table td {{
    text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--gridline);
  }}
  table.data-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  table.data-table th {{ color: var(--text-muted); font-weight: 500; }}

  .hidden {{ display: none; }}

  details.metrics {{ margin-top: 20px; }}
  details.metrics summary {{
    cursor: pointer; font-size: 13px; color: var(--text-secondary); padding: 4px 0;
  }}
  table.metrics-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }}
  table.metrics-table th, table.metrics-table td {{
    text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--gridline);
  }}
  table.metrics-table th:first-child, table.metrics-table td:first-child {{ text-align: left; }}
  table.metrics-table th {{ color: var(--text-muted); font-weight: 500; }}
</style>
</head>
<body>
<div class="viz-root">
  <h1>변수 중요도 리포트</h1>
  <p class="subtitle" id="subtitle">모델 로딩 중...</p>
  <div class="meta-row" id="metaRow"></div>

  <div class="controls">
    <select id="seriesSelect"></select>
    <button class="tablebtn" id="toggleView">테이블로 보기</button>
  </div>

  <div class="card">
    <div id="chartView"></div>
    <div id="tableView" class="hidden"></div>
  </div>

  <details class="metrics" id="metricsPanel">
    <summary>이 모델의 성능 지표</summary>
    <div id="metricsBody"></div>
  </details>
</div>

<script>
const DATA = {data_json};

function fmtPct(v) {{ return (v * 100).toFixed(1) + '%'; }}

function renderMeta() {{
  const m = DATA.meta || {{}};
  document.getElementById('subtitle').textContent =
    m.version ? `모델 버전 ${{m.version}} · 특성 ${{m.feature_count ?? DATA.features.length}}개` : `특성 ${{DATA.features.length}}개`;
  const row = document.getElementById('metaRow');
  row.innerHTML = '';
  if (m.created_at) {{
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = '학습일 ' + m.created_at.slice(0, 19).replace('T', ' ');
    row.appendChild(chip);
  }}
}}

function populateSelect() {{
  const sel = document.getElementById('seriesSelect');
  for (const [group, models] of Object.entries(DATA.series)) {{
    const og = document.createElement('optgroup');
    og.label = group;
    for (const name of Object.keys(models)) {{
      const opt = document.createElement('option');
      const value = group + '::' + name;
      opt.value = value;
      opt.textContent = name === '평균' ? `${{group}} · 전체 평균` : `${{group}} · ${{name}}`;
      og.appendChild(opt);
    }}
    sel.appendChild(og);
  }}
}}

function currentValues() {{
  const [group, name] = document.getElementById('seriesSelect').value.split('::');
  return DATA.series[group][name];
}}

function renderChart() {{
  const values = currentValues();
  const rows = DATA.features.map((f, i) => ({{ feature: f, value: values[i] }}))
    .sort((a, b) => b.value - a.value);
  const max = rows[0].value || 1;

  const chart = document.getElementById('chartView');
  chart.innerHTML = '';
  rows.forEach((r, idx) => {{
    const row = document.createElement('div');
    row.className = 'bar-row' + (idx === 0 ? ' top' : '');
    const pct = Math.max((r.value / max) * 100, 1.5);
    const label = DATA.labels[r.feature] || r.feature;
    row.innerHTML = `
      <div class="bar-label-line">
        <span><span class="rank">${{idx + 1}}</span>${{label}} <span class="code">${{r.feature}}</span></span>
        <span class="value">${{fmtPct(r.value)}}</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%"></div></div>
    `;
    row.title = `${{label}} (${{r.feature}}): ${{fmtPct(r.value)}} (전체 중 ${{idx + 1}}위)`;
    chart.appendChild(row);
  }});

  const table = document.getElementById('tableView');
  table.innerHTML = `
    <table class="data-table">
      <thead><tr><th>순위</th><th>특성 설명</th><th>컬럼명</th><th>중요도</th></tr></thead>
      <tbody>
        ${{rows.map((r, idx) => `<tr><td>${{idx + 1}}</td><td>${{DATA.labels[r.feature] || r.feature}}</td><td class="code">${{r.feature}}</td><td class="num">${{fmtPct(r.value)}}</td></tr>`).join('')}}
      </tbody>
    </table>
  `;
}}

function renderMetrics() {{
  const body = document.getElementById('metricsBody');
  const entries = Object.entries(DATA.metrics || {{}});
  if (!entries.length) {{ body.textContent = '지표 없음'; return; }}
  const cols = new Set();
  entries.forEach(([, v]) => Object.keys(v).forEach(k => cols.add(k)));
  const colList = [...cols];
  const table = document.createElement('table');
  table.className = 'metrics-table';
  table.innerHTML = `
    <thead><tr><th>모델</th>${{colList.map(c => `<th>${{c}}</th>`).join('')}}</tr></thead>
    <tbody>
      ${{entries.map(([name, v]) => `<tr><td>${{name}}</td>${{colList.map(c =>
        `<td>${{v[c] !== undefined ? v[c].toFixed(4) : '-'}}</td>`).join('')}}</tr>`).join('')}}
    </tbody>
  `;
  body.innerHTML = '';
  body.appendChild(table);
}}

document.getElementById('seriesSelect').addEventListener('change', renderChart);
document.getElementById('toggleView').addEventListener('click', (e) => {{
  const chart = document.getElementById('chartView');
  const table = document.getElementById('tableView');
  const showingTable = !table.classList.contains('hidden');
  chart.classList.toggle('hidden', !showingTable);
  table.classList.toggle('hidden', showingTable);
  e.target.textContent = showingTable ? '테이블로 보기' : '차트로 보기';
  e.target.classList.toggle('active', !showingTable);
}});

renderMeta();
populateSelect();
renderChart();
renderMetrics();
</script>
</body>
</html>
"""


def generate_report(trainer, meta: dict) -> Path | None:
    """trainer(학습 직후의 메모리 객체 또는 로드된 객체)로부터 리포트를 생성해 REPORT_PATH에 저장.
    중요도를 추출할 수 있는 모델이 하나도 없으면 아무것도 쓰지 않고 None 반환."""
    series = build_series(trainer)
    if not any(series.values()):
        return None

    html = render_html(trainer.feature_columns, series, meta, meta.get("metrics", {}))
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    return REPORT_PATH


def main():
    parser = argparse.ArgumentParser(description="변수 중요도 HTML 리포트 생성")
    parser.add_argument("--version", default="latest")
    args = parser.parse_args()

    trainer = RaceModelTrainer()
    preprocessor = RacePreprocessor()
    mm = ModelManager()
    if not mm.load(trainer, preprocessor, version=args.version):
        print("로드할 모델이 없습니다. 먼저 모델을 학습하세요.")
        return

    meta = _load_meta(args.version)
    report_path = generate_report(trainer, meta)
    if report_path is None:
        print("중요도를 추출할 수 있는 모델이 없습니다.")
        return
    print(f"리포트 생성 완료: {report_path}")


if __name__ == "__main__":
    main()
