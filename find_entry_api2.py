"""
KRA 출전정보 API 2차 탐색 - 다양한 파라미터/엔드포인트 시도
"""
import sys, io, requests, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import API_KEY

BASE = "https://apis.data.go.kr/B551015"
PARAMS_BASE = {"serviceKey": API_KEY, "pageNo": 1, "numOfRows": 3, "_type": "json"}

# KRA API 특성: meet(경마장코드) + rcDate 또는 rcMonth 가 일반적
# 서울=1, 제주=2, 부산경남=3
# 오늘 날짜 + 각 경마장 조합 테스트
DATE_YYYYMMDD = "20260531"
DATE_YYYYMM   = "202605"

CANDIDATES = [
    # -- 출전마 정보 계열 (Chaksun/Entry) --
    ("API217",  "SeoulChaksun_1",         {"rcDate": DATE_YYYYMMDD, "meet": 1}),
    ("API217",  "SeoulChaksun_1",         {"rcDate": DATE_YYYYMMDD}),
    ("API217",  "ChaksunInfo_1",          {"rcDate": DATE_YYYYMMDD}),
    ("API217",  "raceEntry_1",            {"rcDate": DATE_YYYYMMDD, "meet": 1}),
    ("API217",  "raceEntry_1",            {"rc_date": DATE_YYYYMMDD, "meet": 1}),
    ("API218",  "RaceChaksun_2",          {"rcMonth": DATE_YYYYMM, "meet": 2}),
    ("API218",  "raceChaksun_2",          {"rcMonth": DATE_YYYYMM, "meet": 2}),
    ("API219",  "jeju_chaksun_2",         {"rcMonth": DATE_YYYYMM}),
    ("API219",  "JejuChaksun_2",          {"rcMonth": DATE_YYYYMM}),
    ("API220",  "busan_chaksun_3",        {"rcMonth": DATE_YYYYMM, "meet": 3}),
    ("API220",  "BusanChaksun_3",         {"rcMonth": DATE_YYYYMM, "meet": 3}),
    # -- 다른 API ID 범위 --
    ("API221",  "SeoulChaksun_1",         {"rcDate": DATE_YYYYMMDD}),
    ("API225",  "SeoulChaksun_1",         {"rcDate": DATE_YYYYMMDD}),
    ("API226",  "SeoulEntry_1",           {"rcDate": DATE_YYYYMMDD}),
    ("API228",  "SeoulEntry_1",           {"rcDate": DATE_YYYYMMDD}),
    ("API230",  "SeoulEntry_1",           {"rcDate": DATE_YYYYMMDD}),
    # -- 출주정보 계열 --
    ("API215",  "SeoulJutsu_1",           {"rcDate": DATE_YYYYMMDD}),
    ("API216",  "SeoulJutsu_1",           {"rcDate": DATE_YYYYMMDD}),
    # -- 마체중 별도 API --
    ("API329",  "wgHrInfo",              {"rcDate": DATE_YYYYMMDD}),
    ("API329",  "textDataSeWgHr",        {"rc_date": DATE_YYYYMMDD}),
    # -- SeoulRace 변형 --
    ("API186_1","SeoulEntry_1",          {"rcDate": DATE_YYYYMMDD}),
    ("API186_1","SeoulRaceInfo_1",       {"rcDate": DATE_YYYYMMDD}),
    ("API186_1","SeoulRaceChaksun_1",    {"rcDate": DATE_YYYYMMDD}),
    # -- 경주 일정 (tomorrow's races) --
    ("API213",  "raceSchedule_1",        {"rcDate": DATE_YYYYMMDD}),
    ("API213",  "SeoulRaceSchedule_1",   {"rcDate": DATE_YYYYMMDD}),
    ("API213",  "RaceInfo_1",            {"rcDate": DATE_YYYYMMDD}),
    ("API213",  "raceInfo_1",            {"rc_date": DATE_YYYYMMDD}),
    ("API213",  "SeoulRaceInfo_1",       {"rcDate": DATE_YYYYMMDD}),
    # -- 오늘 날짜 다른 형식 --
    ("API217",  "SeoulEntry_1",          {"rcDate": "2026-05-31"}),
    ("API217",  "SeoulEntry_1",          {"rc_date": "2026-05-31"}),
]

def try_api(api_id, endpoint, extra_params):
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
                if items and items != "" and items is not None:
                    item_list = items.get("item", [])
                    if isinstance(item_list, dict):
                        item_list = [item_list]
                    if item_list:
                        keys = list(item_list[0].keys())
                        has_weight = any(k in ("wgHr", "wg_hr", "weight", "hrWeight") for k in keys)
                        has_track  = any("track" in k.lower() or "jury" in k.lower() for k in keys)
                        return code, f"OK ({total}건) 필드:{keys[:6]} {'[마체중]' if has_weight else ''}", True, keys
                    else:
                        return code, f"빈items (total={total})", False, []
                else:
                    # totalCount=0 이지만 API 자체는 동작
                    return code, f"응답OK (total={total})", total != 0 and total != "?", []
            except Exception as e:
                return code, f"JSON오류:{e}", False, []
        elif code == 403:
            return code, "이용신청필요", False, []
        elif code == 404:
            return code, "없음", False, []
        elif code == 500:
            # 500이어도 응답 메시지에 힌트가 있을 수 있음
            try:
                msg = resp.text[:120].replace("\n","")
            except:
                msg = ""
            return code, f"서버오류 {msg}", False, []
        else:
            return code, "", False, []
    except requests.exceptions.Timeout:
        return 0, "TIMEOUT", False, []
    except Exception as e:
        return 0, str(e)[:60], False, []

print(f"{'API':12s} {'Endpoint':28s} {'코드':5s} {'결과'}")
print("-" * 90)

found = []
for api_id, endpoint, extra in CANDIDATES:
    code, msg, success, keys = try_api(api_id, endpoint, extra)
    if code == 200 or code == 403:
        mark = ">>>" if success else "   "
        print(f"{mark} {api_id:10s} {endpoint:28s} {str(code):5s} {msg}")
        if success and keys:
            found.append((api_id, endpoint, extra, keys))
    time.sleep(0.25)

print("\n" + "=" * 90)
if found:
    print(f"유효한 API {len(found)}개 발견:")
    for api_id, endpoint, params, keys in found:
        print(f"\n  {api_id}/{endpoint}")
        print(f"  파라미터: {params}")
        print(f"  필드: {keys}")
else:
    print("유효한 출전정보 API를 찾지 못함")
    print("\n403 응답 목록 (이용신청 후 사용 가능):")
