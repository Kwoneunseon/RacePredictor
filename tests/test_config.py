from config import FEATURE_COLUMNS


def test_feature_columns_includes_speed_features():
    for col in ["prev_speed_figure", "speed_figure_trend", "horse_speed_at_distance"]:
        assert col in FEATURE_COLUMNS, f"{col} 누락"


def test_feature_columns_excludes_leaky_horse_class_features():
    # horses.rank는 말당 1행 덮어쓰기 스냅샷(2026-05-25 일괄 갱신)이라 과거 경주에 미래 class가 새어 들어간다.
    # 경주 시점 class 이력이 생기기 전까지 이 파생 피처는 학습·예측 피처에서 제외한다.
    for col in ["horse_class_num", "horse_class_vs_mean"]:
        assert col not in FEATURE_COLUMNS, f"{col}는 누수 피처라 FEATURE_COLUMNS에 있으면 안 됨"
