import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from datetime import datetime
from config import MODEL_DIR
from Utils.database import get_client
from Utils.logger import get_logger
from core.preprocessor import RacePreprocessor
from core.trainer import RaceModelTrainer
from core.predictor import RacePredictor
from core.model_manager import ModelManager
from core.race_filter import RaceFilter

target_date = sys.argv[1] if len(sys.argv) > 1 else "2026-05-30"

preprocessor = RacePreprocessor()
trainer = RaceModelTrainer()
mm = ModelManager()
mm.load(trainer, preprocessor)

predictor = RacePredictor(trainer, preprocessor)
predictions = predictor.predict_from_db(target_date)

if predictions.empty:
    print(f"{target_date} 예측 데이터 없음")
    sys.exit()

client = get_client()
races_res = client.table("races").select(
    "race_id, race_distance, race_grade, total_horses, race_no, meet_code"
).eq("race_date", target_date).execute()
race_info = {r["race_id"]: r for r in (races_res.data or [])}

race_filter = RaceFilter()
eligible_ids = {rid for rid, info in race_info.items() if race_filter.is_eligible(info)}

MEET = {1: "서울", 2: "제주", 3: "부산", 0: "미상"}

print(f"\n=== 예측 결과 ({target_date}) ===")
print(f"필터 통과: {len(eligible_ids)}/{len(race_info)}경주\n")

all_groups = list(predictions.groupby("race_id", sort=False))
all_groups.sort(key=lambda x: (
    race_info.get(x[0], {}).get("meet_code", 0),
    race_info.get(x[0], {}).get("race_no", 0),
))

for race_id, group in all_groups:
    info = race_info.get(race_id, {})
    meet = MEET.get(info.get("meet_code"), "미상")
    race_no = info.get("race_no", "?")
    dist = info.get("race_distance", "?")
    total = info.get("total_horses", "?")

    top3 = group.sort_values("win_probability", ascending=False).head(3)
    eligible = "[O]" if race_id in eligible_ids else "[필터제외]"

    print(f"[{meet} {race_no}경주] {dist}m {total}두 {eligible}")
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        name = row.get("horse_name") or row.get("horse_id", "?")
        prob = row["win_probability"]
        print(f"    {rank}위 예측: {name} ({prob:.1%})")
    print()
