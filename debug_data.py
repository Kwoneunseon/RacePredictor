from Utils.database import get_client
import pandas as pd

client = get_client()

# 최근 경주일 찾기
res_dates = client.table("race_entries").select("race_date").order("race_date", desc=True).limit(1).execute()
today = res_dates.data[0]["race_date"] if res_dates.data else "2026-05-25"
print(f"\n=== 당일 데이터: {today} ===")

res = client.table("race_entries").select(
    "*, horses(*), races(*), jockeys(*), trainers(*)"
).eq("race_date", today).limit(3).execute()
print(f"출전 기록 수: {len(res.data)}")

if res.data:
    r = res.data[0]
    flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
    print("\n[race_entries 필드]")
    for k, v in flat.items():
        print(f"  {k:30s} = {v}")

    h = r.get("horses") or {}
    print("\n[horses 필드]")
    for k, v in h.items():
        print(f"  {k:30s} = {v}")

    rc = r.get("races") or {}
    print("\n[races 필드]")
    for k, v in rc.items():
        print(f"  {k:30s} = {v}")

    j = r.get("jockeys") or {}
    print("\n[jockeys 주요]")
    print(f"  total_wins  = {j.get('total_wins')}")
    print(f"  total_races = {j.get('total_races')}")

    t = r.get("trainers") or {}
    print("\n[trainers 주요]")
    print(f"  ord1_cnt_t  = {t.get('ord1_cnt_t')}")
    print(f"  rc_cnt_t    = {t.get('rc_cnt_t')}")

# horse_race_history 확인
horse_id = res.data[0].get("horse_id") if res.data else None
if horse_id:
    hist = client.table("horse_race_history").select("*").eq("horse_id", horse_id).order("as_of_date", desc=True).limit(1).execute()
    print(f"\n[horse_race_history for {horse_id}]")
    if hist.data:
        for k, v in hist.data[0].items():
            print(f"  {k:30s} = {v}")
    else:
        print("  *** 없음 ***")

# 학습 데이터 샘플 비교
print("\n=== 학습 데이터 샘플 (final_rank 있는 과거 기록) ===")
res2 = client.table("race_entries").select(
    "*, horses(*), races(*), jockeys(*), trainers(*)"
).not_.is_("final_rank", "null").order("race_date", desc=True).limit(3).execute()
if res2.data:
    r2 = res2.data[0]
    flat2 = {k: v for k, v in r2.items() if not isinstance(v, dict)}
    print(f"\n날짜: {flat2.get('race_date')}")
    print("[race_entries 주요 필드]")
    key_fields = ["horse_weight", "win_odds", "final_rank"]
    for k in key_fields:
        print(f"  {k:30s} = {flat2.get(k)}")
    h2 = r2.get("horses") or {}
    print(f"  {'horses.age':30s} = {h2.get('age')}")
    print(f"  {'horses.gender':30s} = {h2.get('gender')}")
    print(f"  {'horses.rank':30s} = {h2.get('rank')}")
    j2 = r2.get("jockeys") or {}
    print(f"  {'jockeys.total_wins':30s} = {j2.get('total_wins')}")
    print(f"  {'jockeys.total_races':30s} = {j2.get('total_races')}")
