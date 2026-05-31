"""
당일 데이터 피처 값을 전처리 후 실제로 어떻게 되는지 확인
"""
from Utils.database import get_client
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.model_manager import ModelManager
import pandas as pd

client = get_client()

# ── 1. 당일 데이터 predict_from_db 흐름 그대로 재현 ──
today = "2026-05-31"
res = client.table("race_entries").select(
    "*, horses(*), races(*), jockeys(*), trainers(*)"
).eq("race_date", today).execute()

rows = []
for r in res.data:
    flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
    h  = r.get("horses")   or {}
    rc = r.get("races")    or {}
    j  = r.get("jockeys")  or {}
    t  = r.get("trainers") or {}

    flat["horse_name"]        = h.get("name")
    flat["horse_age"]         = h.get("age")
    flat["gender"]            = h.get("gender")
    flat["horse_class"]       = h.get("rank")
    flat["race_no"]           = rc.get("race_no")
    flat["meet_code"]         = rc.get("meet_code")
    flat["race_distance"]     = rc.get("race_distance")
    flat["total_horses"]      = rc.get("total_horses")
    flat["race_grade"]        = rc.get("race_grade")
    flat["track_condition"]   = rc.get("track_condition")
    flat["weather"]           = rc.get("weather")
    flat["jockey_total_wins"] = j.get("total_wins", 0) or 0
    flat["jockey_total_races"]= j.get("total_races", 0) or 0
    flat["trainer_total_wins"]= t.get("ord1_cnt_t", 0) or 0
    flat["trainer_total_races"]= t.get("rc_cnt_t", 0) or 0
    rows.append(flat)

df = pd.DataFrame(rows)
print(f"당일 출전마: {len(df)}두")

# ── 2. 핵심 컬럼 null/0 현황 ──
check_cols = ["horse_age", "gender", "horse_class", "horse_weight",
              "race_distance", "track_condition", "weather",
              "jockey_total_wins", "jockey_total_races",
              "trainer_total_wins", "trainer_total_races",
              "final_rank", "win_odds"]
print("\n=== 핵심 컬럼 null/0 현황 ===")
for col in check_cols:
    if col in df.columns:
        null_cnt  = df[col].isna().sum()
        zero_cnt  = (df[col] == 0).sum() if df[col].dtype != object else (df[col] == "").sum()
        sample    = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
        print(f"  {col:30s}  null={null_cnt:3d}  zero/blank={zero_cnt:3d}  sample={sample!r}")

# ── 3. track_condition, weather 고유값 ──
print("\n=== 범주형 값 ===")
print("  track_condition 고유값:", df["track_condition"].unique().tolist())
print("  weather         고유값:", df["weather"].unique().tolist())
print("  gender          고유값:", df["gender"].unique().tolist())
print("  horse_class 샘플:", df["horse_class"].head(5).tolist())

# ── 4. horse_race_history 없는 말 비율 ──
horse_ids = df["horse_id"].dropna().unique().tolist()
hist_res = client.table("horse_race_history").select("horse_id, as_of_date")\
    .in_("horse_id", horse_ids).lte("as_of_date", today).execute()
hist_ids = {r["horse_id"] for r in (hist_res.data or [])}
missing_hist = [h for h in horse_ids if h not in hist_ids]
print(f"\n=== horse_race_history 없는 말: {len(missing_hist)}/{len(horse_ids)} ===")
if missing_hist:
    print("  말 ID:", missing_hist[:5])

# ── 5. 학습 데이터와 track_condition, weather 비교 ──
print("\n=== 학습 데이터 track_condition/weather 고유값 ===")
train_res = client.table("races").select("track_condition, weather")\
    .not_.is_("track_condition", "null")\
    .limit(500).execute()
if train_res.data:
    tc_vals = list({r["track_condition"] for r in train_res.data})
    w_vals  = list({r["weather"] for r in train_res.data})
    print("  track_condition:", tc_vals)
    print("  weather        :", w_vals)

# ── 6. final_rank=0 학습 오염 확인 ──
print("\n=== final_rank 분포 (당일) ===")
print(df["final_rank"].value_counts().to_dict())

# 학습 데이터에 final_rank=0이 얼마나 있나
zero_rank = client.table("race_entries").select("id", count="exact")\
    .eq("final_rank", 0).execute()
null_rank = client.table("race_entries").select("id", count="exact")\
    .is_("final_rank", "null").execute()
print(f"\n=== DB 전체 final_rank 분포 ===")
print(f"  final_rank = 0    : {zero_rank.count}건")
print(f"  final_rank = NULL : {null_rank.count}건")
