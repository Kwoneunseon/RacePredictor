import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from core.model_manager import ModelManager
from Utils.database import get_client
import pandas as pd

preprocessor = RacePreprocessor()
trainer = RaceModelTrainer()
mm = ModelManager()
mm.load(trainer, preprocessor)
predictor = RacePredictor(trainer, preprocessor)
client = get_client()

# ── 날짜별 피처 분포 수집 ──
dates = ["2026-05-24", "2026-05-30", "2026-05-31"]
dfs = {}
for d in dates:
    dfs[d] = predictor.predict_from_db(d)

print("=" * 65)
print("날짜별 확률 분포")
print("=" * 65)
for d, df in dfs.items():
    p = df["win_probability"]
    print(f"\n[{d}]  최고={p.max():.3f}  평균={p.mean():.3f}  std={p.std():.3f}  0.5이상={( p>=0.5).sum()}마  0.3이상={(p>=0.3).sum()}마")

print("\n" + "=" * 65)
print("핵심 피처 평균 비교")
print("=" * 65)
feats = [
    "horse_win_rate", "prev_5_avg_rank", "prev_total_races",
    "horse_win_rate_at_distance", "days_since_last_race",
    "horse_weight", "jockey_win_rate", "jockey_recent_win_rate",
    "trainer_recent_win_rate", "training_grade_enc",
    "track_condition_enc", "weather_enc",
    "horse_class_vs_mean", "win_rate_vs_mean",
]
print(f"\n{'피처':32s} {'5/24':>8} {'5/30':>8} {'5/31':>8}")
print("-" * 62)
for f in feats:
    vals = []
    for d in dates:
        df = dfs[d]
        vals.append(df[f].mean() if f in df.columns else float("nan"))
    flag = " !!!" if max(vals) > 0 and (max(vals) - min(vals)) / max(abs(v) for v in vals if v == v) > 0.3 else ""
    print(f"  {f:30s} {vals[0]:8.3f} {vals[1]:8.3f} {vals[2]:8.3f}{flag}")

# ── track_condition/weather 실제값 확인 ──
print("\n" + "=" * 65)
print("5/24 vs 5/30 races 테이블 track_condition / weather 실제값")
print("=" * 65)
for d in ["2026-05-24", "2026-05-30"]:
    res = client.table("races").select("track_condition, weather").eq("race_date", d).execute()
    tc = set(r["track_condition"] for r in res.data)
    wt = set(r["weather"] for r in res.data)
    print(f"  [{d}]  track_condition={tc}  weather={wt}")

# ── horse_race_history as_of_date 확인 ──
print("\n" + "=" * 65)
print("horse_race_history as_of_date 분포 (5/24 출전마)")
print("=" * 65)
df24 = dfs["2026-05-24"]
horse_ids_24 = df24["horse_id"].dropna().unique().tolist()[:20]
hist = client.table("horse_race_history").select("horse_id, as_of_date")\
    .in_("horse_id", horse_ids_24).order("as_of_date", desc=True).execute()
if hist.data:
    dates_hist = [r["as_of_date"] for r in hist.data]
    from collections import Counter
    date_cnt = Counter(dates_hist)
    for dt, cnt in sorted(date_cnt.items(), reverse=True)[:10]:
        flag = " <-- 당일 포함!" if dt == "2026-05-24" else ""
        print(f"  {dt}: {cnt}건{flag}")

# ── 5/24 최고확률 말 피처 상세 ──
print("\n" + "=" * 65)
print("5/24 최고확률 말 피처 (안양빛의속도 87.6%)")
print("=" * 65)
top24 = df24.loc[df24["win_probability"].idxmax()]
print(f"  말 이름: {top24.get('horse_name','?')}")
for f in feats:
    if f in top24.index:
        print(f"  {f:32s} = {top24[f]}")

# ── 5/24 이 말의 horse_race_history ──
print("\n" + "=" * 65)
print("해당 말의 horse_race_history 전체 이력")
print("=" * 65)
h_id = top24.get("horse_id")
if h_id:
    hr = client.table("horse_race_history").select("*").eq("horse_id", h_id).order("as_of_date").execute()
    for row in hr.data:
        flag = " <-- 당일 포함?" if row["as_of_date"] >= "2026-05-24" else ""
        print(f"  as_of={row['as_of_date']}  wins={row['prev_wins']}  total={row['prev_total_races']}  avg5={row['prev_5_avg_rank']}{flag}")
