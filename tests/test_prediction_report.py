from core import prediction_report


def _payload(**overrides):
    base = {
        "date": "2026-07-12",
        "threshold": 0.65,
        "total_races": 5,
        "eligible_races": 3,
        "race_rows": [],
        "bet_races": [],
        "candidates": [],
        "decisions": [],
    }
    base.update(overrides)
    return base


def test_render_html_includes_date_and_race_counts():
    html = prediction_report.render_html(_payload(date="2026-07-12", total_races=5, eligible_races=3))
    assert "2026-07-12" in html
    assert "3" in html and "5" in html


def test_render_html_shows_bet_recommendation():
    html = prediction_report.render_html(_payload(bet_races=[{
        "label": "[서울 3경주]", "horse_no": "4", "horse": "테스트마",
        "top3_prob": 0.72, "odds": 1.8, "reason": "동일·1.80배", "eligible": True,
    }]))
    assert "테스트마" in html
    assert "1.8" in html


def test_render_html_no_bet_message_when_empty():
    html = prediction_report.render_html(_payload(bet_races=[]))
    assert "없음" in html


def test_render_html_shows_candidate_when_no_realtime_odds():
    html = prediction_report.render_html(_payload(candidates=[{
        "label": "[부산 4경주]", "win_no": "2", "win_name": "말C", "t3_no": "2",
        "t3_name": "말C", "t3_prob": 0.68, "same": True, "eligible": True,
    }]))
    assert "말C" in html


def test_render_html_shows_ev_decision():
    html = prediction_report.render_html(_payload(decisions=[{
        "label": "[제주 2경주]", "horse_no": "7", "horse_name": "말이름",
        "bet_type": "단승", "amount": 5000, "ev": 0.35, "model_prob": 0.42,
        "odds": 3.1, "source": "scraping",
    }]))
    assert "말이름" in html
    assert "5,000" in html


def test_render_html_shows_race_rows():
    html = prediction_report.render_html(_payload(race_rows=[{
        "label": "[서울 1경주]", "win_no": "3", "win_name": "말A", "win_prob": 0.31,
        "t3_no": "5", "t3_name": "말B", "t3_prob": 0.55, "eligible": True,
    }]))
    assert "말A" in html
    assert "말B" in html


def test_render_html_hides_result_column_when_no_results():
    html = prediction_report.render_html(_payload(race_rows=[{
        "label": "[서울 1경주]", "win_no": "3", "win_name": "말A", "win_prob": 0.31,
        "t3_no": "5", "t3_name": "말B", "t3_prob": 0.55, "eligible": True, "actual_result": "",
    }]))
    assert "실제 결과" not in html


def test_render_html_shows_result_column_when_available():
    html = prediction_report.render_html(_payload(race_rows=[{
        "label": "[서울 1경주]", "win_no": "3", "win_name": "말A", "win_prob": 0.31,
        "t3_no": "5", "t3_name": "말B", "t3_prob": 0.55, "eligible": True, "actual_result": "1-3-4-11",
    }]))
    assert "실제 결과" in html
    assert "1-3-4-11" in html


def test_render_html_marks_correct_win_and_top3_picks():
    html = prediction_report.render_html(_payload(race_rows=[{
        "label": "[서울 1경주]", "win_no": "1", "win_name": "말A", "win_prob": 0.31,
        "t3_no": "3", "t3_name": "말B", "t3_prob": 0.55, "eligible": True, "actual_result": "1-3-4-11",
    }]))
    assert "말A ✅" in html
    assert "말B ✅" in html


def test_render_html_marks_incorrect_win_and_top3_picks():
    html = prediction_report.render_html(_payload(race_rows=[{
        "label": "[서울 1경주]", "win_no": "3", "win_name": "말A", "win_prob": 0.31,
        "t3_no": "5", "t3_name": "말B", "t3_prob": 0.55, "eligible": True, "actual_result": "1-3-4-11",
    }]))
    assert "말A ❌" in html
    assert "말B ❌" in html


def test_render_html_no_mark_when_no_actual_result():
    html = prediction_report.render_html(_payload(race_rows=[{
        "label": "[서울 1경주]", "win_no": "3", "win_name": "말A", "win_prob": 0.31,
        "t3_no": "5", "t3_name": "말B", "t3_prob": 0.55, "eligible": True, "actual_result": "",
    }]))
    assert "✅" not in html
    assert "❌" not in html


def test_save_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_report, "REPORTS_DIR", tmp_path)
    path = prediction_report.save_report(_payload(date="2026-07-12"))
    assert path.exists()
    assert path.name == "prediction_2026-07-12.html"


def test_save_report_appends_suffix_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_report, "REPORTS_DIR", tmp_path)
    path1 = prediction_report.save_report(_payload(date="2026-07-12"))
    path2 = prediction_report.save_report(_payload(date="2026-07-12"))
    path3 = prediction_report.save_report(_payload(date="2026-07-12"))
    assert path1.name == "prediction_2026-07-12.html"
    assert path2.name == "prediction_2026-07-12_1.html"
    assert path3.name == "prediction_2026-07-12_2.html"
    assert path1.exists() and path2.exists() and path3.exists()
