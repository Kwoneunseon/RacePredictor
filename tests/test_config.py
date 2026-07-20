from config import FEATURE_COLUMNS


def test_feature_columns_includes_speed_features():
    for col in ["prev_speed_figure", "speed_figure_trend", "horse_speed_at_distance"]:
        assert col in FEATURE_COLUMNS, f"{col} 누락"
