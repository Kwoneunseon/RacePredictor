"""
기존 작동 API로 오늘 날짜 호출 시 실제 반환 필드 확인
track_condition, weather, horse_weight가 있는지 체크
"""
import sys, io, requests, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from config import API_KEY

TODAY = "20260531"

endpoints = [
    ("서울", "https://apis.data.go.kr/B551015/API186_1/SeoulRace_1",
     {"serviceKey": API_KEY, "pageNo": 1, "numOfRows": 3, "rc_date_fr": TODAY, "rc_date_to": TODAY, "_type": "json"}),
    ("부산", "http://apis.data.go.kr/B551015/API4_3/raceResult_3",
     {"serviceKey": API_KEY, "pageNo": 1, "numOfRows": 3, "rc_month": "202605", "_type": "json"}),
    ("제주", "https://apis.data.go.kr/B551015/API4_2/raceResult_2",
     {"serviceKey": API_KEY, "pageNo": 1, "numOfRows": 3, "rc_month": "202605", "_type": "json"}),
]

for name, url, params in endpoints:
    print(f"\n{'='*60}")
    print(f"[{name}] {url.split('/')[-1]}")
    print("="*60)
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"  오류: {resp.status_code}")
        continue
    data = resp.json()
    body = data.get("response", {}).get("body", {})
    items = body.get("items") or {}
    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]
    if not item_list:
        print(f"  데이터 없음 (totalCount={body.get('totalCount')})")
        continue

    # 첫 번째 항목 전체 필드 출력
    item = item_list[0]
    print(f"  총 {body.get('totalCount')}건, 첫 번째 항목 필드:")
    for k, v in item.items():
        print(f"    {k:20s} = {v!r}")

    # 마체중/주로상태/날씨 필드 찾기
    key_fields = {
        "마체중": ["wgHr", "wg_hr", "hrWeight", "hrWt"],
        "주로상태": ["track", "trackCond", "jury", "juryStatus"],
        "날씨":   ["weath", "weather", "weather_cd"],
        "경주일": ["rcDate", "rc_date", "rcDt"],
        "경주번호": ["rcNo", "rc_no"],
        "말이름": ["hrName", "hr_name", "hrNm"],
        "출전번호": ["chulNo", "rcChul"],
    }
    print(f"\n  핵심 필드 유무:")
    for label, candidates in key_fields.items():
        found = [(c, item.get(c)) for c in candidates if c in item]
        if found:
            print(f"    {label}: {found}")
        else:
            print(f"    {label}: 없음")
