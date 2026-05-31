"""
KRA 출전정보 API 엔드포인트 탐색
data.go.kr B551015 계열에서 출전정보 관련 API를 찾는다
"""
import sys, io, requests, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import API_KEY

BASE = "https://apis.data.go.kr/B551015"
TODAY = "20260531"
PARAMS_BASE = {"serviceKey": API_KEY, "pageNo": 1, "numOfRows": 3, "_type": "json"}

# 출전정보/출주정보 관련 가능한 API 목록
CANDIDATES = [
    # 서울 출전정보
    ("API186_2", "SeoulRaceEntry_2",    {"rcDate": TODAY}),
    ("API217",   "SeoulEntry",          {"rcDate": TODAY}),
    ("API217",   "SeoulEntry_1",        {"rcDate": TODAY}),
    ("API217",   "ChaksunList_1",       {"rc_date": TODAY}),
    ("API217",   "chaksunList_1",       {"rc_date": TODAY}),
    # 일반 출전 정보
    ("API213",   "RaceDetailInfo_1",    {"rcDate": TODAY}),
    ("API213",   "raceDetailInfo_1",    {"rc_date": TODAY}),
    ("API213",   "SeoulRaceEntry_1",    {"rcDate": TODAY}),
    ("API214",   "SeoulRaceEntry_1",    {"rcDate": TODAY}),
    # 출전마 정보 (horses in race)
    ("API218",   "ChaksunList_1",       {"rc_date": TODAY}),
    ("API218",   "chaksunList_1",       {"rc_date": TODAY}),
    ("API218",   "SeoulEntry_1",        {"rcDate": TODAY}),
    # 기타 패턴
    ("API222",   "SeoulEntry_1",        {"rcDate": TODAY}),
    ("API223",   "SeoulEntry_1",        {"rcDate": TODAY}),
    ("API224",   "SeoulEntry_1",        {"rcDate": TODAY}),
    # 서울 경주 관련 다른 엔드포인트들
    ("API186_1", "SeoulRaceInfo_1",     {"rcDate": TODAY}),
    ("API186_3", "SeoulRaceEntry_3",    {"rcDate": TODAY}),
    # 출전정보 + 마체중
    ("API329",   "chaksunList",         {"rc_date": TODAY}),
    ("API217",   "raceEntry_1",         {"rc_date": TODAY}),
    ("API217",   "RaceEntry_1",         {"rcDate": TODAY}),
]

print(f"{'API':15s} {'Endpoint':25s} {'상태':8s} {'결과'}")
print("-" * 80)

found = []
for api_id, endpoint, extra_params in CANDIDATES:
    url = f"{BASE}/{api_id}/{endpoint}"
    params = {**PARAMS_BASE, **extra_params}
    try:
        resp = requests.get(url, params=params, timeout=8)
        code = resp.status_code
        if code == 200:
            try:
                data = resp.json()
                body = data.get("response", {}).get("body", {})
                total = body.get("totalCount", "?")
                items = body.get("items")
                if items and items != "":
                    item_list = items.get("item", [])
                    if isinstance(item_list, dict):
                        item_list = [item_list]
                    status = f"OK ({total}건)"
                    if item_list:
                        keys = list(item_list[0].keys()) if item_list else []
                        has_weight = any("wg" in k.lower() or "weight" in k.lower() for k in keys)
                        has_track = any("track" in k.lower() or "jury" in k.lower() for k in keys)
                        status += f" 필드:{keys[:5]}{'...' if len(keys)>5 else ''}"
                        if has_weight:
                            status += " [마체중O]"
                        found.append((api_id, endpoint, total, keys))
                    print(f"  {api_id:13s} {endpoint:25s} {str(code):8s} {status}")
                else:
                    print(f"  {api_id:13s} {endpoint:25s} {str(code):8s} 빈 응답 (totalCount={total})")
            except Exception as e:
                print(f"  {api_id:13s} {endpoint:25s} {str(code):8s} JSON파싱오류: {e}")
        elif code == 403:
            print(f"  {api_id:13s} {endpoint:25s} 403      이용 신청 필요")
        elif code == 404:
            print(f"  {api_id:13s} {endpoint:25s} 404      엔드포인트 없음")
        else:
            print(f"  {api_id:13s} {endpoint:25s} {str(code):8s}")
    except requests.exceptions.Timeout:
        print(f"  {api_id:13s} {endpoint:25s} TIMEOUT")
    except Exception as e:
        print(f"  {api_id:13s} {endpoint:25s} ERR      {e}")
    time.sleep(0.3)

print("\n" + "=" * 80)
if found:
    print(f"데이터 있는 API: {len(found)}개")
    for api_id, endpoint, total, keys in found:
        print(f"  {api_id}/{endpoint}  건수={total}")
        print(f"    필드: {keys}")
else:
    print("데이터 있는 API를 찾지 못했습니다.")
