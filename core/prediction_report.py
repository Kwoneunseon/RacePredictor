"""예측 결과 HTML 리포트 생성. cmd_predict가 계산을 마친 결과 데이터를 받아 렌더링만 담당한다."""
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _fmt_amount(v) -> str:
    return f"{int(v):,}"


def render_html(payload: dict) -> str:
    date = payload["date"]
    threshold = payload["threshold"]
    total_races = payload["total_races"]
    eligible_races = payload["eligible_races"]
    race_rows = payload.get("race_rows", [])
    bet_races = payload.get("bet_races", [])
    candidates = payload.get("candidates", [])
    decisions = payload.get("decisions", [])

    def elig_mark(r):
        return "" if r.get("eligible", True) else ' <span class="tag-out">필터제외</span>'

    def _actual_ranks(r):
        result = r.get("actual_result") or ""
        return result.split("-") if result else []

    def win_mark(r):
        ranks = _actual_ranks(r)
        no = r.get("win_no")
        if not ranks or no in (None, "", "?"):
            return ""
        return " ✅" if str(no) == ranks[0] else " ❌"

    def top3_mark(r):
        ranks = _actual_ranks(r)
        no = r.get("t3_no")
        if not ranks or no in (None, "", "?"):
            return ""
        return " ✅" if str(no) in ranks[:3] else " ❌"

    show_result_col = any(r.get("actual_result") for r in race_rows)
    result_header = "<th>실제 결과</th>" if show_result_col else ""
    result_col_count = 7 if show_result_col else 6

    def result_cell(r):
        if not show_result_col:
            return ""
        return f'<td class="code">{r.get("actual_result") or "-"}</td>'

    race_rows_html = "".join(f"""
        <tr>
          <td>{r['label']}</td>
          <td>{r['win_no']}번 {r['win_name']}{win_mark(r)}</td>
          <td class="num">{r['win_prob']:.3f}</td>
          <td>{r['t3_no']}번 {r['t3_name']}{top3_mark(r)}</td>
          <td class="num">{r['t3_prob']:.3f}</td>
          <td>{elig_mark(r)}</td>
          {result_cell(r)}
        </tr>""" for r in race_rows) or f'<tr><td colspan="{result_col_count}" class="empty">경주 없음</td></tr>'

    if bet_races:
        bet_html = "".join(f"""
        <div class="pick-row">
          <div class="pick-main">{r['label']} → <strong>{r['horse_no']}번 {r['horse']}</strong>{elig_mark(r)}</div>
          <div class="pick-sub">top3_prob={r['top3_prob']:.3f} · 배당={r['odds']:.2f}배 · {r['reason']}</div>
        </div>""" for r in bet_races)
    else:
        bet_html = f'<p class="empty">연승 추천: 없음 (top3_prob &ge; {threshold:.2f} + 배당 조건 미달)</p>'

    if candidates:
        cand_html = "".join(f"""
        <div class="pick-row">
          <div class="pick-main">{c['label']} → <strong>{
              (str(c['t3_no']) + '번 ' + c['t3_name']) if c.get('same')
              else f"단승:{c['win_no']}번 {c['win_name']} / 연승:{c['t3_no']}번 {c['t3_name']}"
          }</strong>{elig_mark(c)}</div>
          <div class="pick-sub">prob={c['t3_prob']:.3f} · 실시간 배당 없음 — 직접 확인 필요</div>
        </div>""" for c in candidates)
    else:
        cand_html = ""

    if decisions:
        dec_html = "".join(f"""
        <div class="pick-row">
          <div class="pick-main">{d['label']} → <strong>{d['horse_no']}번 {d['horse_name']}</strong> · {d['bet_type']} {_fmt_amount(d['amount'])}원</div>
          <div class="pick-sub">EV={d['ev']:+.2f} · 확률={d['model_prob']:.3f} · 배당={d['odds']:.1f}배 · 출처={d['source']}</div>
        </div>""" for d in decisions)
    else:
        dec_html = '<p class="empty">베팅 추천: 없음 (EV 기준 미달)</p>'

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>예측 결과 리포트 - {date}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  .viz-root {{
    --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --text-muted: #898781; --gridline: #e1e0d9;
    --border: rgba(11,11,11,0.10); --series-1: #2a78d6; --chip-bg: #f0efec;
    --good: #0ca30c; --critical: #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    .viz-root {{
      --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
      --text-secondary: #c3c2b7; --text-muted: #898781; --gridline: #2c2c2a;
      --border: rgba(255,255,255,0.10); --series-1: #3987e5; --chip-bg: #26261f;
      --good: #0ca30c; --critical: #e66767;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #898781; --gridline: #2c2c2a;
    --border: rgba(255,255,255,0.10); --series-1: #3987e5; --chip-bg: #26261f;
  }}
  :root[data-theme="light"] .viz-root {{
    --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --text-muted: #898781; --gridline: #e1e0d9;
    --border: rgba(11,11,11,0.10); --series-1: #2a78d6; --chip-bg: #f0efec;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .viz-root {{ max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 28px 0 10px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }}
  .meta-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }}
  .chip {{ background: var(--chip-bg); color: var(--text-secondary); border-radius: 999px; padding: 5px 12px; font-size: 12px; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--gridline); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  th {{ color: var(--text-muted); font-weight: 500; }}
  td.empty, p.empty {{ color: var(--text-muted); }}
  .tag-out {{ color: var(--text-muted); font-size: 11px; }}
  .pick-row {{ padding: 10px 0; border-bottom: 1px solid var(--gridline); }}
  .pick-row:last-child {{ border-bottom: none; }}
  .pick-main {{ font-size: 13.5px; }}
  .pick-main strong {{ color: var(--series-1); }}
  .pick-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 2px; }}
</style>
</head>
<body>
<div class="viz-root">
  <h1>예측 결과 리포트</h1>
  <p class="subtitle">{date}</p>
  <div class="meta-row">
    <span class="chip">필터 통과 {eligible_races}/{total_races}경주</span>
    <span class="chip">연승 기준 top3_prob &ge; {threshold:.2f}</span>
  </div>

  <h2>연승 베팅 추천</h2>
  <div class="card">{bet_html}</div>

  {"<h2>연승 후보 (배당 직접 확인)</h2><div class='card'>" + cand_html + "</div>" if candidates else ""}

  <h2>EV 기반 베팅 추천</h2>
  <div class="card">{dec_html}</div>

  <h2>전체 경주 예측</h2>
  <div class="card">
    <table>
      <thead><tr><th>경주</th><th>단승 1위</th><th>확률</th><th>연승 1위</th><th>확률</th><th></th>{result_header}</tr></thead>
      <tbody>{race_rows_html}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""


def save_report(payload: dict) -> Path:
    html = render_html(payload)
    REPORTS_DIR.mkdir(exist_ok=True)
    base_name = f"prediction_{payload['date']}"
    path = REPORTS_DIR / f"{base_name}.html"
    n = 1
    while path.exists():
        path = REPORTS_DIR / f"{base_name}_{n}.html"
        n += 1
    path.write_text(html, encoding="utf-8")
    return path
