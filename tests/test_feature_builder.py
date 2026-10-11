import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────
# get_flatten_entries 단위 테스트 (DB 불필요)
# ──────────────────────────────────────────────
def make_raw_entry(horse_id="H1", race_date="2026-06-07"):
    return {
        "id": 1,
        "race_id": f"{race_date}_서울_1",
        "race_date": race_date,
        "horse_id": horse_id,
        "entry_number": 1,
        "horse_weight": 480,
        "final_rank": None,
        "jk_no": "J001",
        "trainer_id": "T001",
        "win_odds": 0,
        "finish_time": None,
        "horses": {"name": "테스트마", "age": 9, "birth_year": 2022, "gender": "수", "gelded_since": None, "rank": "국5"},
        "races": {
            "race_no": 1, "meet_code": 1, "race_distance": 1200,
            "total_horses": 8, "race_grade": "일반",
            "track_condition": "양호", "weather": "맑음",
        },
        "jockeys": {"total_wins": 100, "total_races": 1000},
        "trainers": {"ord1_cnt_t": 50, "rc_cnt_t": 500},
    }


def test_flatten_entries_extracts_nested_fields():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = fb.get_flatten_entries([make_raw_entry()])
    assert df["horse_name"].iloc[0] == "테스트마"
    assert df["horse_age"].iloc[0] == 4
    assert df["meet_code"].iloc[0] == 1
    assert df["race_distance"].iloc[0] == 1200
    assert df["jockey_total_wins"].iloc[0] == 100
    assert df["trainer_total_wins"].iloc[0] == 50


def test_flatten_entries_multiple_rows():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = fb.get_flatten_entries([make_raw_entry("H1"), make_raw_entry("H2")])
    assert len(df) == 2
    assert set(df["horse_id"]) == {"H1", "H2"}


def _flatten_with_horse(race_date, horse):
    from core.feature_builder import RaceFeatureBuilder
    entry = make_raw_entry(race_date=race_date)
    entry["horses"] = horse
    return RaceFeatureBuilder.get_flatten_entries([entry]).iloc[0]


@pytest.mark.parametrize("race_date,expected", [
    ("2024-03-02", 2), ("2025-12-28", 3), ("2026-01-03", 4),
])
def test_flatten_horse_age_is_race_year_minus_birth_year(race_date, expected):
    """horses.age(수집일 스냅샷)가 아니라 경주 연도 기준 나이여야 한다"""
    row = _flatten_with_horse(race_date, {"age": 4, "birth_year": 2022, "gender": "암"})
    assert row["horse_age"] == expected


@pytest.mark.parametrize("horse", [
    {"age": 5, "gender": "수"},                       # 컬럼 자체가 없음(백필 전)
    {"age": 5, "gender": "수", "birth_year": None},   # 값이 NULL
    None,                                             # horses 조인 실패
])
def test_flatten_horse_age_missing_birth_year_is_nan_not_zero_or_snapshot(horse):
    row = _flatten_with_horse("2026-06-07", horse)
    assert pd.isna(row["horse_age"])


@pytest.mark.parametrize("race_date,gender,gelded_since,expected", [
    ("2025-02-28", "거", "2025-03-01", "수"),   # 거세 전 경주
    ("2025-03-01", "거", "2025-03-01", "거"),   # 거세 관측 당일
    ("2026-06-07", "거", "2025-03-01", "거"),
    ("2026-06-07", "거", None, "거"),           # 관측일 없으면 현재 값 그대로
    ("2026-06-07", "수", None, "수"),
    ("2026-06-07", "암", None, "암"),
])
def test_flatten_gender_is_gender_at_race_time(race_date, gender, gelded_since, expected):
    row = _flatten_with_horse(race_date, {"birth_year": 2021, "gender": gender, "gelded_since": gelded_since})
    assert row["gender"] == expected


def test_build_for_prediction_uses_race_year_age_and_race_time_gender():
    """예측 경로도 학습과 같은 정의(경주 연도 - birth_year, 경주 당시 성별)를 쓴다"""
    entry = make_raw_entry("H1", race_date="2026-06-07")
    entry["horses"] = {"name": "테스트마", "age": 9, "birth_year": 2021,
                       "gender": "거", "gelded_since": "2026-08-01", "rank": "국5"}
    with patch("core.feature_builder.get_client", return_value=make_mock_client([entry])), \
         patch("library.scraper_odds.fetch_race_win_odds", return_value={}):
        from core.feature_builder import RaceFeatureBuilder
        result = RaceFeatureBuilder().build_for_prediction("2026-06-07")
    assert result["horse_age"].iloc[0] == 5
    assert result["gender"].iloc[0] == "수"


# ──────────────────────────────────────────────
# build_for_prediction 통합 테스트 (DB mock)
# ──────────────────────────────────────────────
def make_mock_client(entry_data=None):
    """race_entries 첫 조회에만 데이터 반환, 나머지는 빈 리스트"""
    client = MagicMock()
    mock_chain = client.table.return_value.select.return_value

    # race_entries 첫 조회 (.eq("race_date", ...))
    mock_chain.eq.return_value.execute.return_value.data = entry_data or []

    # 이후 피처 조회들 → 빈 리스트 (MagicMock이 체인을 자동 생성하므로
    # execute().data만 설정하면 됨)
    mock_chain.in_.return_value.lt.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.lt.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.lt.return_value.gt.return_value.lt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.in_.return_value.gte.return_value.lt.return_value.execute.return_value.data = []
    mock_chain.gte.return_value.lt.return_value.not_.is_.return_value.execute.return_value.data = []
    mock_chain.gte.return_value.lt.return_value.gt.return_value.lt.return_value.execute.return_value.data = []
    # _fetch_assigned_weight: .select(...).in_("race_id", ...).execute() (중간 필터 없음)
    mock_chain.in_.return_value.execute.return_value.data = []

    return client


def test_build_for_prediction_empty_db():
    """DB에 데이터 없으면 빈 DataFrame 반환"""
    with patch("core.feature_builder.get_client", return_value=make_mock_client([])):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_build_for_prediction_returns_required_columns():
    """race_entries가 있으면 필수 컬럼을 포함한 DataFrame 반환"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    with patch("core.feature_builder.get_client", return_value=client), \
         patch("library.scraper_odds.fetch_race_win_odds", return_value={}):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    required = [
        "horse_id", "horse_name", "meet_code", "race_distance",
        "prev_win_odds", "current_win_odds",
        "days_since_last_race", "prev_speed_figure", "speed_figure_trend",
        "assigned_weight",
    ]
    for col in required:
        assert col in result.columns, f"{col} 컬럼 없음"


def test_build_for_prediction_fallback_values():
    """피처 DB 조회 결과 없을 때 폴백값이 설정된다"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    with patch("core.feature_builder.get_client", return_value=client), \
         patch("library.scraper_odds.fetch_race_win_odds", return_value={}):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    assert result["prev_win_odds"].iloc[0] == 15.0
    # 스크래핑 실패 시 current_win_odds = prev_win_odds
    assert result["current_win_odds"].iloc[0] == 15.0
    assert result["days_since_last_race"].iloc[0] == 60
    assert result["prev_speed_figure"].iloc[0] == 0.0
    # race_entry_detail에 매칭되는 행 없음 → 기본 폴백값 56.0
    assert result["assigned_weight"].iloc[0] == 56.0


# ──────────────────────────────────────────────
# _fetch_distance_rates 단위 테스트 (DB mock)
# ──────────────────────────────────────────────
def test_fetch_distance_rates_computes_speed_at_distance():
    """같은 거리 과거 경주들의 평균 속도(거리/기록시간)를 계산한다."""
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "race_distance": [1200]})

    client = MagicMock()
    chain = client.table.return_value.select.return_value.in_.return_value.lt.return_value.gt.return_value.lt.return_value
    chain.execute.return_value.data = [
        {"horse_id": "H1", "final_rank": 2, "finish_time": 80.0, "races": {"race_distance": 1200}},
        {"horse_id": "H1", "final_rank": 1, "finish_time": 75.0, "races": {"race_distance": 1200}},
    ]

    result = fb._fetch_distance_rates(df, "2026-06-07", client)
    assert "horse_speed_at_distance" in result.columns
    expected = (1200 / 80.0 + 1200 / 75.0) / 2
    assert result["horse_speed_at_distance"].iloc[0] == pytest.approx(expected)


def test_fetch_distance_rates_speed_defaults_to_zero_when_no_history():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "race_distance": [1200]})

    client = MagicMock()
    chain = client.table.return_value.select.return_value.in_.return_value.lt.return_value.gt.return_value.lt.return_value
    chain.execute.return_value.data = []

    result = fb._fetch_distance_rates(df, "2026-06-07", client)
    assert result["horse_speed_at_distance"].iloc[0] == 0.0


def test_fetch_assigned_weight_for_training_fills_median_when_missing():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "race_id": ["R1", "R2", "R3"],
        "horse_id": ["H1", "H2", "H3"],
    })
    detail_rows = [
        {"race_id": "R1", "horse_id": "H1", "assigned_weight": 54.0},
        {"race_id": "R2", "horse_id": "H2", "assigned_weight": 58.0},
        # R3/H3는 race_entry_detail에 없음 (수집 실패/미백필 케이스)
    ]
    result = fb._fetch_assigned_weight_for_training(df, detail_rows)
    assert result.loc[result["race_id"] == "R1", "assigned_weight"].iloc[0] == 54.0
    assert result.loc[result["race_id"] == "R2", "assigned_weight"].iloc[0] == 58.0
    # 결측은 중앙값(54, 58의 중앙값=56)으로 대체
    assert result.loc[result["race_id"] == "R3", "assigned_weight"].iloc[0] == 56.0


def test_build_for_prediction_uses_scraped_odds():
    """스크래핑 성공 시 current_win_odds에 실시간 배당이 반영된다"""
    entry_data = [make_raw_entry("H1")]
    client = make_mock_client(entry_data)

    # entry_number=1인 말에 배당 5.3 반환
    with patch("core.feature_builder.get_client", return_value=client), \
         patch("library.scraper_odds.fetch_race_win_odds", return_value={1: 5.3}):
        from core.feature_builder import RaceFeatureBuilder
        fb = RaceFeatureBuilder()
        result = fb.build_for_prediction("2026-06-07")

    assert result["current_win_odds"].iloc[0] == pytest.approx(5.3)


# ──────────────────────────────────────────────
# 2단계: 구간기록·통산/최근1년 성적
# ──────────────────────────────────────────────
def test_fetch_section_features_treats_zero_corner_as_not_applicable():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"],
        "race_date": ["2026-07-12"],
        "race_distance": [1000],  # 임계값 미만 → 1C/2C 미해당
    })
    section_rows = [{
        "horse_id": "H1", "race_distance": 1000, "snapshot_race_date": "2026-07-01",
        "c1_avg": 0.0, "c2_avg": 0.0, "c3_avg": 30.0, "c4_avg": 49.0,
        "s1f_avg": 13.9, "g3f_avg": 37.8, "g1f_avg": 13.5,
    }]
    result = fb._join_section_records(df, section_rows)
    assert result.loc[0, "c3_avg"] == 30.0
    assert result.loc[0, "g1f_avg"] == 13.5


def test_horse_career_win_rate_uses_starts_as_denominator():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "race_id": ["R1"], "horse_id": ["H1"],
    })
    detail_rows = [{
        "race_id": "R1", "horse_id": "H1",
        "horse_career_wins": 3, "horse_career_starts": 10,
        "jockey_career_wins": 50, "jockey_career_starts": 500,
        "jockey_recent1y_wins": 5, "jockey_recent1y_starts": 40,
        "trainer_career_wins": 20, "trainer_career_starts": 300,
        "trainer_recent1y_wins": 2, "trainer_recent1y_starts": 30,
        "assigned_weight": 56.0,
    }]
    result = fb._join_career_stats(df, detail_rows)
    assert result.loc[0, "horse_career_win_rate"] == 0.3
    assert result.loc[0, "jockey_recent1y_win_rate"] == 0.125


# ──────────────────────────────────────────────
# 3단계: 러닝스타일 (코너별 통과순위)
# ──────────────────────────────────────────────
def test_running_style_score_positive_means_closer():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"], "race_date": ["2026-07-12"],
    })
    # 과거 2경주: 1코너 5위→최종 2위(3계단 추입), 1코너 6위→최종 1위(5계단 추입)
    past_rows = [
        {"horse_id": "H1", "race_date": "2026-06-01", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 2},
        {"horse_id": "H1", "race_date": "2026-06-15", "sj_1c_ord": 6, "sj_3c_ord": 5, "final_rank": 1},
    ]
    result = fb._compute_running_style(df, past_rows)
    assert result.loc[0, "running_style_score"] == pytest.approx((5 - 2 + 6 - 1) / 2)


def test_running_style_score_falls_back_to_3c_when_1c_not_applicable():
    import pandas as pd
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({
        "horse_id": ["H1"], "race_date": ["2026-07-12"],
    })
    # 짧은 거리라 1C 없음(None) → 3C(4위) 기준으로 계산해야 함: 4 - 2 = 2
    past_rows = [
        {"horse_id": "H1", "race_date": "2026-06-01", "sj_1c_ord": None, "sj_3c_ord": 4, "final_rank": 2},
    ]
    result = fb._compute_running_style(df, past_rows)
    assert result.loc[0, "running_style_score"] == pytest.approx(2.0)


# ──────────────────────────────────────────────
# 90번대(비착순 상태 코드) 필터
# ──────────────────────────────────────────────
class _RecordingQuery:
    """PostgREST 체인 호출을 순서 무관하게 기록하는 가짜 쿼리 빌더 (DB 미접속)."""

    def __init__(self, data, calls):
        self._data = data
        self.calls = calls

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args))
            return self
        return _record

    @property
    def not_(self):
        return self

    def execute(self):
        return MagicMock(data=self._data)


class _RecordingClient:
    def __init__(self, data_by_table):
        self.data_by_table = data_by_table
        self.calls = []

    def table(self, name):
        self.calls.append(("table", (name,)))
        return _RecordingQuery(self.data_by_table.get(name, []), self.calls)


def _has_valid_rank_filter(calls):
    from config import NON_FINISH_RANK_MIN
    return ("gt", ("final_rank", 0)) in calls and ("lt", ("final_rank", NON_FINISH_RANK_MIN)) in calls


def test_fetch_recent_rates_query_filters_valid_rank_range():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"jk_no": ["J1"], "trainer_id": ["T1"]})
    client = _RecordingClient({"race_entries": [{"jk_no": "J1", "trainer_id": "T1", "final_rank": 1}]})
    fb._fetch_recent_rates(df, "2026-06-07", client)
    assert _has_valid_rank_filter(client.calls)


def test_fetch_last_race_features_query_filters_valid_rank_range():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "horse_weight": [480]})
    client = _RecordingClient({"race_entries": []})
    fb._fetch_last_race_features(df, "2026-06-07", client)
    assert _has_valid_rank_filter(client.calls)


def test_fetch_speed_features_query_filters_valid_rank_range():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"]})
    client = _RecordingClient({"race_entries": []})
    fb._fetch_speed_features(df, "2026-06-07", client)
    assert _has_valid_rank_filter(client.calls)


def test_fetch_distance_rates_query_filters_valid_rank_range():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "race_distance": [1200]})
    client = _RecordingClient({"race_entries": []})
    fb._fetch_distance_rates(df, "2026-06-07", client)
    assert _has_valid_rank_filter(client.calls)


def test_fetch_combo_features_query_filters_valid_rank_range():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "meet_code": [1]})
    client = _RecordingClient({"race_entries": []})
    fb._fetch_combo_features(df, "2026-06-07", client)
    assert _has_valid_rank_filter(client.calls)


def test_compute_running_style_excludes_non_finish_ranks():
    """final_rank 99(경주취소)/93(제외)/0(미기입) 행은 러닝스타일 평균에 들어가지 않는다."""
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1", "H2"], "race_date": ["2026-07-12"] * 2})
    past_rows = [
        {"horse_id": "H1", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 2},   # delta 3
        {"horse_id": "H1", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 99},  # 제외돼야 함 (-94)
        {"horse_id": "H1", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 93},
        {"horse_id": "H1", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 0},
        {"horse_id": "H2", "sj_1c_ord": 5, "sj_3c_ord": 4, "final_rank": 99},  # 유효 행 없음 -> 0.0
    ]
    result = fb._compute_running_style(df, past_rows).set_index("horse_id")
    assert result.loc["H1", "running_style_score"] == pytest.approx(3.0)
    assert result.loc["H2", "running_style_score"] == 0.0


def test_fetch_running_style_for_training_excludes_non_finish_ranks():
    from core.feature_builder import RaceFeatureBuilder
    fb = RaceFeatureBuilder()
    df = pd.DataFrame({"horse_id": ["H1"], "race_date": ["2026-07-12"]})
    detail = [
        {"horse_id": "H1", "race_id": "R1", "sj_1c_ord": 5, "sj_3c_ord": 4},
        {"horse_id": "H1", "race_id": "R2", "sj_1c_ord": 5, "sj_3c_ord": 4},
    ]
    ranks = [
        {"race_id": "R1", "horse_id": "H1", "race_date": "2026-06-01", "final_rank": 2},
        {"race_id": "R2", "horse_id": "H1", "race_date": "2026-06-15", "final_rank": 99},
    ]
    client = _RecordingClient({"race_entry_detail": detail, "race_entries": ranks})
    with patch("core.feature_builder.get_client", return_value=client):
        result = fb._fetch_running_style_for_training(df)
    assert result["running_style_score"].iloc[0] == pytest.approx(3.0)


# ──────────────────────────────────────────────
# 속도 피처 학습/예측 경로 정의 일치
# ──────────────────────────────────────────────
def _speed_history(finish_times, target_finish_time=70.0):
    """H1의 과거 경주(2026-01-01부터 하루 간격) + 대상 경주(2026-06-07) 학습용 df."""
    dates = [f"2026-01-{i + 1:02d}" for i in range(len(finish_times))] + ["2026-06-07"]
    return pd.DataFrame({
        "horse_id": ["H1"] * len(dates),
        "race_date": dates,
        "finish_time": list(finish_times) + [target_finish_time],
        "race_distance": [1200] * len(dates),
    })


def _serving_speed_features(train_df):
    from core.feature_builder import RaceFeatureBuilder
    past = train_df[train_df["race_date"] < "2026-06-07"]
    rows = [{
        "horse_id": r.horse_id, "race_date": r.race_date,
        "finish_time": None if pd.isna(r.finish_time) else r.finish_time,
        "races": {"race_distance": r.race_distance},
    } for r in past.itertuples()]
    client = _RecordingClient({"race_entries": rows})
    return RaceFeatureBuilder()._fetch_speed_features(
        pd.DataFrame({"horse_id": ["H1"]}), "2026-06-07", client).iloc[0]


def _training_speed_features(train_df):
    from core.feature_builder import RaceFeatureBuilder
    result = RaceFeatureBuilder()._fetch_speed_features_for_training(train_df.copy())
    return result[result["race_date"] == "2026-06-07"].iloc[0]


@pytest.mark.parametrize("finish_times", [
    [80.0, None],                     # 직전 경주 기록 없음, 그 전 경주 기록 있음
    [80.0, 78.0, None, 75.0, None],   # 기록 없는 경주가 사이사이 낀 경우
    [80.0, 78.0, 76.0, 0],            # 0은 기록 없음
    [80.0, 78.0, 76.0, 74.0],         # 결측 없음
    [None, None],                     # 기록 전무
    [],                               # 데뷔전
])
def test_speed_features_training_matches_serving(finish_times):
    train_df = _speed_history(finish_times)
    serving = _serving_speed_features(train_df)
    training = _training_speed_features(train_df)
    assert training["prev_speed_figure"] == pytest.approx(serving["prev_speed_figure"])
    assert training["speed_figure_trend"] == pytest.approx(serving["speed_figure_trend"])


def test_speed_features_training_uses_latest_recorded_race():
    training = _training_speed_features(_speed_history([80.0, 78.0, None, 75.0, None]))
    assert training["prev_speed_figure"] == pytest.approx(1200 / 75.0)
    expected = np.polyfit(np.arange(3), [1200 / 80.0, 1200 / 78.0, 1200 / 75.0], 1)[0]
    assert training["speed_figure_trend"] == pytest.approx(expected)


def test_speed_features_training_ignores_own_and_later_races():
    """대상 경주 자신과 이후 경주의 기록이 바뀌어도 값이 같아야 한다(누수 방지)."""
    base = _speed_history([80.0, None], target_finish_time=70.0)
    later = pd.DataFrame({"horse_id": ["H1"], "race_date": ["2026-07-01"],
                          "finish_time": [60.0], "race_distance": [1200]})
    changed = pd.concat([_speed_history([80.0, None], target_finish_time=50.0), later], ignore_index=True)
    a, b = _training_speed_features(base), _training_speed_features(changed)
    assert a["prev_speed_figure"] == pytest.approx(1200 / 80.0)
    assert b["prev_speed_figure"] == pytest.approx(1200 / 80.0)
    assert a["speed_figure_trend"] == b["speed_figure_trend"] == 0.0
