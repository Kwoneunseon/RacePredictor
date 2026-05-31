import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from core.model_manager import ModelManager
import pandas as pd
import numpy as np

preprocessor = RacePreprocessor()
trainer = RaceModelTrainer()
mm = ModelManager()
mm.load(trainer, preprocessor)
predictor = RacePredictor(trainer, preprocessor)

results = {}
for date in ["2026-05-30", "2026-05-31"]:
    df = predictor.predict_from_db(date)
    results[date] = df

print("=" * 60)
print("날짜별 확률 분포 비교")
print("=" * 60)
for date, df in results.items():
    probs = df["win_probability"]
    print(f"\n[{date}]")
    print(f"  출전마 수    : {len(df)}")
    print(f"  최고 확률    : {probs.max():.3f}")
    print(f"  중앙값       : {probs.median():.3f}")
    print(f"  평균         : {probs.mean():.3f}")
    print(f"  표준편차     : {probs.std():.3f}")
    print(f"  0.30 이상    : {(probs >= 0.30).sum()}마")
    print(f"  0.20 이상    : {(probs >= 0.20).sum()}마")

# 핵심 피처 평균 비교
from config import FEATURE_COLUMNS
print("\n" + "=" * 60)
print("핵심 피처 평균값 비교 (30일 vs 31일)")
print("=" * 60)

key_features = [
    "horse_win_rate", "prev_5_avg_rank", "prev_total_races",
    "horse_win_rate_at_distance", "days_since_last_race",
    "horse_weight", "horse_weight_diff",
    "jockey_win_rate", "trainer_win_rate",
    "jockey_recent_win_rate", "trainer_recent_win_rate",
    "training_grade_enc", "horse_class_num",
    "track_condition_enc", "weather_enc",
]

df30 = results["2026-05-30"]
df31 = results["2026-05-31"]

print(f"\n{'피처':30s} {'30일':>10} {'31일':>10} {'차이':>10}")
print("-" * 65)
for feat in key_features:
    if feat in df30.columns and feat in df31.columns:
        v30 = df30[feat].mean()
        v31 = df31[feat].mean()
        diff = v31 - v30
        flag = " !!!" if abs(diff) > 0.3 * max(abs(v30), abs(v31), 0.01) else ""
        print(f"  {feat:28s} {v30:10.3f} {v31:10.3f} {diff:+10.3f}{flag}")

# NULL/0 현황 비교
print("\n" + "=" * 60)
print("NULL/0 비율 비교")
print("=" * 60)
critical = ["horse_weight", "prev_total_races", "horse_win_rate_at_distance",
            "days_since_last_race", "training_grade_enc", "track_condition_enc", "weather_enc"]
print(f"\n{'피처':30s} {'30일 0비율':>12} {'31일 0비율':>12}")
print("-" * 58)
for feat in critical:
    if feat in df30.columns and feat in df31.columns:
        z30 = (df30[feat].fillna(0) == 0).mean()
        z31 = (df31[feat].fillna(0) == 0).mean()
        flag = " !!!" if abs(z30 - z31) > 0.1 else ""
        print(f"  {feat:28s} {z30:12.1%} {z31:12.1%}{flag}")

# 30일 최고확률 말 vs 31일 최고확률 말 피처 비교
print("\n" + "=" * 60)
print("최고확률 말 피처 상세 비교")
print("=" * 60)
top30 = df30.loc[df30["win_probability"].idxmax()]
top31 = df31.loc[df31["win_probability"].idxmax()]
print(f"\n  30일 1위: {top30.get('horse_name','?')} ({top30['win_probability']:.1%})")
print(f"  31일 1위: {top31.get('horse_name','?')} ({top31['win_probability']:.1%})")
print(f"\n{'피처':30s} {'30일 1위':>12} {'31일 1위':>12}")
print("-" * 58)
for feat in key_features:
    if feat in df30.columns and feat in df31.columns:
        v30 = top30.get(feat, np.nan)
        v31 = top31.get(feat, np.nan)
        flag = " <<<" if abs(float(v31 or 0) - float(v30 or 0)) > 0.1 else ""
        print(f"  {feat:28s} {str(round(v30,3) if pd.notna(v30) else 'N/A'):>12} {str(round(v31,3) if pd.notna(v31) else 'N/A'):>12}{flag}")
