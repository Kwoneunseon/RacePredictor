from Utils.database import get_client
import pandas as pd

client = get_client()
res = client.table("race_entries").select(
    "race_id, horse_id, final_rank, horses(name), races(race_no, meet_code, race_distance)"
).eq("race_date", "2026-05-30").gt("final_rank", 0).execute()
# final_rank=1인 것만 필터
res.data = [r for r in res.data if r.get("final_rank") == 1]

rows = []
for r in res.data:
    h = r.get("horses") or {}
    rc = r.get("races") or {}
    rows.append({
        "race_no":  rc.get("race_no"),
        "meet":     rc.get("meet_code"),
        "dist":     rc.get("race_distance"),
        "winner":   h.get("name"),
    })

df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values(["meet", "race_no"])
print("=== 2026-05-30 실제 결과 ===")
for _, r in df.iterrows():
    meet = "서울" if r["meet"] == 1 else ("부산" if r["meet"] == 3 else "제주")
    print(f"  [{meet} {int(r['race_no'])}경주] 실제 1위: {r['winner']} ({int(r['dist'])}m)")
