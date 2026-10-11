# -*- coding: utf-8 -*-
import calendar
import requests
from datetime import datetime
import time

from library.parsing import parse_date, safe_int, safe_float, safe_str
from config import API_KEY
from Utils.database import upsert, get_client
from Utils.logger import get_logger
from .collector_horse import fetch_single_horse_data as horse_fetch_page, save_to_supabase_batch as save_horse_data, parse_horse_data
from .collector_jockeys import fetch_single_jockey_data as jockey_fetch_page, save_to_supabase_batch as save_jockey_data, parse_jockey_data
from .collector_trainers import fetch_single_trainer_data as trainer_fetch_page, save_to_supabase_batch as save_trainer_data, parse_trainer_data

logger = get_logger(__name__)

MEET_CODE_MAP = {'서울': 1, '제주': 2, '부산': 3, '부산경남': 3}
TABLES_WITHOUT_UPDATED_AT = {'races', 'race_entries', 'race_payouts'}
TABLE_ON_CONFLICT = {
    'races': 'race_id',
    'race_entries': 'race_id,horse_id',
    'race_payouts': 'race_id,bet_type,entry_numbers',
}

# plcOdds(연승 확정 배당)를 파싱하는 경마장. 말 개별 plcOdds 필드는 공용
# API(raceResult_3) 응답에만 있다. 서울 전용 SeoulRace_1 응답에는 없으므로
# (경주 단위 집계값 rcP1~8Odd뿐) 서울 연승 배당은 common_api=True로 받은
# 행에서만 나온다. 서울 값은 2026-10-11 ScoreInfo.do 확정 배당과 대조 검증.
PLC_ODDS_SUPPORTED_MEET_CODES = {1, 2, 3}

# 3개 경마장 공용 경주 결과 API - meet 파라미터로 경마장을 지정한다(생략하면
# 3곳이 섞여 온다). 말 개별 winOdds/plcOdds가 있는 유일한 소스.
COMMON_RACE_ENDPOINT = "http://apis.data.go.kr/B551015/API4_3/raceResult_3"

# 경마장별 API 엔드포인트. 제주 전용 API4_2/raceResult_2는 폐기됨(code 12).
MEET_RACE_ENDPOINTS = {
    1: "https://apis.data.go.kr/B551015/API186_1/SeoulRace_1",   # 서울 (winOdds 없음)
    2: COMMON_RACE_ENDPOINT,                                      # 제주
    3: COMMON_RACE_ENDPOINT,                                      # 부산경남
}

# HTTP 요청 로그 숨기기
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.getLogger("postgrest").setLevel(logging.WARNING)

def fetch_race_data(page=1, per_page=1000 ,start_date=None, end_date=None):
    """서울 경주 결과 API에서 데이터 가져오는 함수"""
    params = {
        "serviceKey": API_KEY,
        "pageNo": page,
        "numOfRows": per_page,
        # 시작일 (기본값: None)
        **({"rc_date_fr": start_date} if start_date else {}),
        "rc_date_to": end_date,  # 종료일
        "_type": 'json'

    }

    try:
        endpoint = "https://apis.data.go.kr/B551015/API186_1/SeoulRace_1"
        response = requests.get(endpoint, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data, page
        elif response.status_code == 429:
            logger.warning(f"API 호출 제한 - 페이지 {page}, 5초 대기 후 재시도")
            time.sleep(5)
            return fetch_race_data(page, per_page, start_date, end_date)
        else:
            logger.error(f"API 호출 실패 - 페이지 {page}: {response.status_code}")
            return None, page

    except Exception as e:
        logger.error(f"API 호출 중 오류 발생 - 페이지 {page}: {str(e)}")
        return None, page

def fetch_race_data_all(page=1, per_page=1000, meet_code=2, start_date=None, end_date=None, common_api=False):
    """경마장별 엔드포인트로 경주 결과를 가져오는 함수.

    서울(1): SeoulRace_1 → rc_date_fr / rc_date_to (YYYYMMDD)
    제주(2)·부산경남(3): raceResult_3 + meet → rc_month (YYYYMM) 또는 rc_date (YYYYMMDD)
    common_api=True면 서울도 raceResult_3(meet=1)로 조회한다 - SeoulRace_1에는
    말 개별 winOdds가 없어서, 단승 배당은 이 경로로만 받을 수 있다.
    """
    if common_api:
        endpoint = COMMON_RACE_ENDPOINT
    else:
        endpoint = MEET_RACE_ENDPOINTS.get(meet_code, COMMON_RACE_ENDPOINT)

    if meet_code == 1 and not common_api:
        # 서울 API: rc_date_to / rc_date_fr 방식
        # end_date가 YYYYMM(6자리)이면 해당 월의 1일~말일로 변환
        end_str = str(end_date) if end_date else datetime.now().strftime('%Y%m%d')
        if len(end_str) == 6:
            year, month = int(end_str[:4]), int(end_str[4:6])
            last_day = calendar.monthrange(year, month)[1]
            rc_date_to = f"{end_str}{last_day:02d}"
            rc_date_fr = f"{end_str}01"
        else:
            rc_date_to = end_str
            rc_date_fr = start_date

        params = {
            "serviceKey": API_KEY,
            "pageNo": page,
            "numOfRows": per_page,
            "rc_date_to": rc_date_to,
            "_type": "json",
        }
        if rc_date_fr:
            params["rc_date_fr"] = rc_date_fr
    else:
        # 공용 API: 8자리 날짜면 그 하루(rc_date), 아니면 월 전체(rc_month)
        end_str = str(end_date) if end_date else datetime.now().strftime('%Y%m')
        params = {
            "serviceKey": API_KEY,
            "pageNo": page,
            "numOfRows": per_page,
            "meet": meet_code,
            "_type": "json",
        }
        if len(end_str) == 8:
            params["rc_date"] = end_str
        else:
            params["rc_month"] = end_str[:6]

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = requests.Request('GET', endpoint, params=params).prepare()
            logger.info(f"API 요청 URL - 경마장{meet_code} 페이지 {page}: {req.url}")
            response = requests.Session().send(req, timeout=60)

            if response.status_code == 200:
                return response.json(), page
            elif response.status_code == 429:
                wait = 5 * attempt
                logger.warning(f"API 호출 제한 - 경마장{meet_code} 페이지 {page}, {wait}초 대기 후 재시도")
                time.sleep(wait)
                continue
            elif response.status_code in (502, 503, 504):
                wait = 15 * attempt
                logger.warning(f"API 서버 오류 {response.status_code} - 경마장{meet_code} 페이지 {page} (시도 {attempt}/{max_retries}), {wait}초 대기 후 재시도")
                if attempt < max_retries:
                    time.sleep(wait)
            else:
                logger.error(f"API 호출 실패 - 경마장{meet_code} 페이지 {page}: {response.status_code}")
                return None, page

        except requests.exceptions.Timeout:
            wait = 10 * attempt
            logger.warning(f"API 타임아웃 - 경마장{meet_code} 페이지 {page} (시도 {attempt}/{max_retries}), {wait}초 대기 후 재시도")
            if attempt < max_retries:
                time.sleep(wait)
        except Exception as e:
            logger.error(f"API 호출 중 오류 발생 - 경마장{meet_code} 페이지 {page}: {str(e)}")
            return None, page

    logger.error(f"API 최대 재시도 초과 - 경마장{meet_code} 페이지 {page}")
    return None, page

def get_existing_master_data():
    """기존 마스터 데이터 개수 확인"""
    supabase = get_client()
    try:
        existing_data = {
            'horses': set(),
            'jockeys': set(),
            'trainers': set(),
            'owners': set()
        }

        # 테이블별 설정
        table_configs = [
            ('horses', 'horse_id', 'horses'),
            ('jockeys', 'jk_no', 'jockeys'),
            ('trainers', 'trainer_id', 'trainers'),
        ]

        for table_name, id_column, key in table_configs:
            page_size = 1000
            start = 0

            while True:
                try:
                    response = supabase.table(table_name)\
                        .select(id_column)\
                        .range(start, start + page_size - 1)\
                        .execute()

                    if not response.data:
                        break

                    # 세트에 추가
                    for item in response.data:
                        existing_data[key].add(item[id_column])

                    logger.info(f"{table_name}: {len(response.data)}개 로드됨 (누적: {len(existing_data[key])}개)")

                    if len(response.data) < page_size:
                        break

                    start += page_size

                except Exception as e:
                    logger.error(f"{table_name} 페이지 {start//page_size + 1} 로드 실패: {str(e)}")
                    break

        total_loaded = sum(len(data) for data in existing_data.values())
        logger.info(f"총 {total_loaded}개 마스터 데이터 로드 완료")

        return existing_data

    except Exception as e:
        logger.error(f"마스터 데이터 캐시 생성 실패: {str(e)}")
        return {'horses': set(), 'jockeys': set(), 'trainers': set(), 'owners': set()}


def save_missing_master_data(missing_data):
    """누락된 마스터 데이터 저장"""
    try:
        saved_counts = {}

        # 말 정보 저장
        if missing_data['horses']:
            horse_datas=[]
            logger.info(f"새로운 말 {len(missing_data['horses'])}마리 추가")
            for batch_start in range(0, len(missing_data['horses'])):
                entry = missing_data['horses'][batch_start]
                horse_id = entry['horse_id']
                horse_data = horse_fetch_page(horse_id)
                if horse_data:
                    horse_datas.append(horse_data)
                else:
                    # API에 데이터 없어도 최소 레코드 저장 (race_entries FK 제약 충족)
                    horse_datas.append({'horse_id': horse_id, 'name': entry.get('name', f'Horse_{horse_id}')})
            save_horse_data(horse_datas)
            saved_counts['horses'] = len(missing_data['horses'])
        else:
            saved_counts['horses'] = 0

        # 기수 정보 저장
        if missing_data['jockeys']:
            jockey_datas = []
            logger.info(f"새로운 기수 {len(missing_data['jockeys'])}명 추가")
            for batch_start in range(0, len(missing_data['jockeys'])):
                entry = missing_data['jockeys'][batch_start]
                jockey_no = entry['jk_no']
                api_data, _ = jockey_fetch_page(jockey_no)
                jockey_data = parse_jockey_data(api_data)
                if jockey_data:
                    jockey_datas.append(jockey_data[0])
                else:
                    # API 데이터 없어도 최소 레코드 저장 (race_entries FK 제약 충족)
                    jockey_datas.append({'jk_no': jockey_no, 'name': entry.get('name', f'Jockey_{jockey_no}')})

            save_jockey_data(jockey_datas)
            saved_counts['jockeys'] = len(missing_data['jockeys'])
        else:
            saved_counts['jockeys'] = 0

        # 조교사 정보 저장
        if missing_data['trainers']:
            trainer_datas = []
            logger.info(f"새로운 조교사 {len(missing_data['trainers'])}명 추가")
            for batch_start in range(0, len(missing_data['trainers'])):
                entry = missing_data['trainers'][batch_start]
                trainer_id = entry['trainer_id']
                api_data, _ = trainer_fetch_page(trainer_id)
                trainer_data = parse_trainer_data(api_data)
                if trainer_data:
                    trainer_datas.append(trainer_data[0])
                else:
                    # API 데이터 없어도 최소 레코드 저장 (race_entries FK 제약 충족)
                    trainer_datas.append({'trainer_id': trainer_id, 'name': entry.get('name', f'Trainer_{trainer_id}')})

            save_trainer_data(trainer_datas)
            saved_counts['trainers'] = len(missing_data['trainers'])
        else:
            saved_counts['trainers'] = 0

        saved_counts['owners'] = 0

        return saved_counts

    except Exception as e:
        logger.error(f"마스터 데이터 저장 실패: {str(e)}")
        return {'horses': 0, 'jockeys': 0, 'trainers': 0, 'owners': 0}

def parse_and_normalize_race_data(api_response):
    """API 응답을 파싱하여 정규화된 테이블 구조로 변환"""
    races = []
    race_entries = []
    race_payouts = []  # 제주·부산경남 API의 plcOdds(연승 확정 배당) - bet_type='place'
    horse_obs = []  # 공용 API의 말 나이·성별 관측(경주 시점 기준)
    master_data = {
        'horses': [],
        'jockeys': [],
        'trainers': [],
        'owners': []
    }
    def parse_horse_weight_simple(wg_hr_str):
        if wg_hr_str is None or wg_hr_str == "" or wg_hr_str == "()":
            return 469, 0  # 평균 마체중

        wg_hr_str = str(wg_hr_str)

        # 괄호 없으면 순수 숫자 (서울 API가 int로 반환하는 경우)
        if '(' not in wg_hr_str:
            try:
                return int(wg_hr_str), 0
            except ValueError:
                return 469, 0

        parts = wg_hr_str.split('(')
        weight = int(parts[0]) if parts[0] else None

        change_part = parts[1].rstrip(')')
        if change_part == '':
            change = 0
        elif change_part:
            change = int(change_part)
        else:
            change = None

        return weight, change

    try:
        if 'response' in api_response and 'body' in api_response['response']:
            body = api_response['response']['body']
            items = body.get('items', []) if body else []

            if items:
                items = items.get('item', [])

                # 중복 제거를 위한 집합
                processed_races = set()
                processed_horses = set()
                processed_jockeys = set()
                processed_trainers = set()
                processed_owners = set()

                for item in items:
                    race_date = parse_date(item.get('rcDate')) #경마일자
                    meet_str = safe_str(item.get('meet'))  #경마장 명 (한글)
                    meet_code = MEET_CODE_MAP.get(meet_str, 0)  #경마장 코드 (정수)
                    race_no = safe_int(item.get('rcNo'))    #경주 번호
                    horse_id = safe_str(item.get('hrno')) if item.get('hrno') else safe_str(item.get('hrNo'))  #마번

                    # 마스터 데이터 ID들과 이름들
                    jk_no = safe_str(item.get('jkNo'))
                    jockey_name = safe_str(item.get('jkName'))
                    trainer_id = safe_str(item.get('prtr')) if item.get('prtr') else safe_str(item.get('trNo'))
                    trainer_name = safe_str(item.get('prtrName')) if item.get('prtrName') else safe_str(item.get('trName'))
                    owner_id = safe_str(item.get('prow')) if item.get('prow') else safe_str(item.get('owNo'))
                    owner_name = safe_str(item.get('prowName')) if item.get('prowName') else safe_str(item.get('owName'))
                    horse_name = safe_str(item.get('hrName')) if item.get('hrName') else safe_str(item.get('trName'))

                    if not all([race_date, meet_str, race_no, horse_id, jk_no, trainer_id, owner_id]):
                        continue

                    # 경주 기본 정보 (new schema: race_id is composite key string)
                    race_key = (race_date, meet_str, race_no)
                    race_id = f"{race_date}_{meet_str}_{race_no}"
                    if race_key not in processed_races:
                        races.append({
                            'race_id': race_id,
                            'race_date': race_date,
                            'meet_code': meet_code,
                            'race_no': race_no,
                            'race_distance': safe_int(item.get('rcDist')),  #경주거리
                            'race_grade': safe_str(item.get('rcGrade')) if item.get('rcGrade') else safe_str(item.get('rcName')), #경주등급
                            'track_condition': safe_str(item.get('track')) if item.get('track') else '', #경주로상태
                            'weather': safe_str(item.get('weath')) if item.get('weath') else safe_str(item.get('weather')),         #날씨
                            'total_horses': safe_int(item.get('rcVtdusu'))  if item.get('rcVtdusu') else 11, #총 출전마 수
                        })
                        processed_races.add(race_key)

                    # 마스터 데이터 수집 (API에서 가져온 실제 정보 사용)
                    if horse_id not in processed_horses:
                        master_data['horses'].append({
                            'horse_id': horse_id,
                            'name': horse_name or f'Horse_{horse_id}',
                        })
                        processed_horses.add(horse_id)

                    if jk_no not in processed_jockeys:
                        master_data['jockeys'].append({
                            'jk_no': jk_no,
                            'name': jockey_name or f'Jockey_{jk_no}'
                        })
                        processed_jockeys.add(jk_no)

                    if trainer_id not in processed_trainers:
                        master_data['trainers'].append({
                            'trainer_id': trainer_id,
                            'name': trainer_name or f'Trainer_{trainer_id}'
                        })
                        processed_trainers.add(trainer_id)

                    if owner_id and owner_id not in processed_owners:
                        master_data['owners'].append({
                            'owner_id': owner_id,
                            'owner_name': owner_name or f'Owner_{owner_id}'
                        })
                        processed_owners.add(owner_id)

                    # 경주 참가 기록 (new schema columns only)
                    horse_weight, horse_weight_diff = parse_horse_weight_simple(item.get('wgHr'))
                    entry_number = safe_int(item.get('rcChul')) if item.get('rcChul') else safe_int(item.get('chulNo'))

                    race_entries.append({
                        'race_id': race_id,
                        'horse_id': horse_id,
                        'race_date': race_date,
                        'entry_number': entry_number,   #출전번호
                        'horse_weight': horse_weight,     #마체중
                        'jk_no': jk_no,
                        'trainer_id': trainer_id,
                        # rcOrd(제주·부산) → ord → chaksun(서울 착순) 순으로 fallback
                        'final_rank': (safe_int(item.get('rcOrd')) or
                                       safe_int(item.get('ord')) or
                                       safe_int(item.get('chaksun'))),
                        'win_odds': safe_float(item.get('winOdds')),  #단승배당률
                        'finish_time': safe_float(item.get('rcTime')),  #완주기록(초)
                    })

                    age, sex = safe_int(item.get('age')), safe_str(item.get('sex'))
                    if age and sex:
                        horse_obs.append({
                            'horse_id': horse_id,
                            'race_date': race_date,
                            'age': age,
                            'sex': sex,
                        })

                    # 연승(place) 확정 배당 - 공용 API 응답에만 있는 plcOdds(말 개별
                    # 값). SeoulRace_1 응답에는 이 필드가 없어 자동으로 스킵된다.
                    plc_odds = safe_float(item.get('plcOdds'))
                    if (meet_code in PLC_ODDS_SUPPORTED_MEET_CODES
                            and entry_number and plc_odds and plc_odds > 0):
                        race_payouts.append({
                            'race_id': race_id,
                            'bet_type': 'place',
                            'entry_numbers': [entry_number],
                            'odds': plc_odds,
                        })


    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")

    return {
        'races': races,
        'race_entries': race_entries,
        'race_payouts': race_payouts,
        'horse_obs': horse_obs,
        'master_data': master_data
    }


def summarize_horse_attrs(observations):
    """말별 출생연도·성별·거세 확인일 요약. 성별은 수→거로만 바뀌므로 '거'가 한 번이라도 있으면 거세마다."""
    attrs = {}
    for o in observations:
        a = attrs.setdefault(o['horse_id'], {
            'horse_id': o['horse_id'],
            'birth_year': None, 'gender': None, 'gelded_since': None,
        })
        a['birth_year'] = int(o['race_date'][:4]) - o['age']
        if o['sex'] == '거':
            a['gender'] = '거'
            if a['gelded_since'] is None or o['race_date'] < a['gelded_since']:
                a['gelded_since'] = o['race_date']
        elif a['gender'] != '거':
            a['gender'] = o['sex']
    return list(attrs.values())


def save_horse_attrs(attrs, batch_size=200):
    """horses에 출생연도·성별·거세 확인일 반영. 이미 있는 말만 갱신하고, 거세 정보는 되돌리지 않는다."""
    supabase = get_client()
    saved = 0
    for i in range(0, len(attrs), batch_size):
        batch = attrs[i:i + batch_size]
        try:
            existing = supabase.table('horses').select('horse_id, name, gender, gelded_since')                .in_('horse_id', [a['horse_id'] for a in batch]).execute().data
            current = {r['horse_id']: r for r in existing}
            rows = []
            for a in batch:
                cur = current.get(a['horse_id'])
                if cur is None:
                    continue
                p = cur['gelded_since']
                if p and (a['gelded_since'] is None or p < a['gelded_since']):
                    a['gelded_since'] = p
                if cur['gender'] == '거' or a['gelded_since']:
                    a['gender'] = '거'
                # 교류경주에서는 이름에 [서울] 같은 접두가 붙어 오므로 DB 이름을 유지한다
                rows.append({**a, 'name': cur['name']})
            if rows:
                supabase.table('horses').upsert(rows, on_conflict='horse_id').execute()
            saved += len(rows)
        except Exception as e:
            logger.error(f"horses 나이·성별 저장 실패 (배치 {i // batch_size + 1}): {str(e)[:120]}")
    return saved

def filter_duplicates_in_race_and_entries(races, race_entries):
    """race, race_entries 리스트 내 중복 제거 (race_date + race_id + meet_code, entry_key 기준)"""
    filtered_races = []
    seen_race_keys = set()
    for race in races:
        race_key = race.get('race_id')
        if race_key not in seen_race_keys:
            filtered_races.append(race)
            seen_race_keys.add(race_key)

    filtered_entries = []
    seen_entries = {}
    for entry in race_entries:
        entry_key = (
            entry.get('race_id'),
            entry.get('horse_id')
        )
        first = seen_entries.get(entry_key)
        if first is None:
            filtered_entries.append(entry)
            seen_entries[entry_key] = entry
        elif not first.get('win_odds') and entry.get('win_odds'):
            # 서울은 SeoulRace_1 행(winOdds 없음 → 0)이 먼저 오므로, 뒤따르는
            # 공용 API 행의 단승 배당으로 채운다.
            first['win_odds'] = entry['win_odds']

    return filtered_races, filtered_entries


def filter_duplicates_in_payouts(race_payouts):
    """race_payouts 리스트 내 중복 제거 (race_id, bet_type, entry_numbers 기준).

    제주·부산경남 API 응답에서 동일 rcDate+rcNo에 다른 hrNo 포맷(부산·경남
    통합 이전 잔재로 추정, [[project_race_payouts_scraper]] 참고)의 중복 행이
    섞여 나오는 현상이 관찰된 바 있어, race_entries와 마찬가지로 선착순
    dedup을 거친다. 같은 키에 서로 다른 odds가 온 경우는 먼저 본 값을 유지
    (Supabase upsert도 동일 키 재전송 시 마지막 값으로 덮어쓰므로, 배치 내
    dedup은 순서에 따라 결과가 갈릴 수 있는 극히 드문 경우를 로그로 남기지
    않고 조용히 첫 값을 취하는 정도의 안전장치일 뿐 - 실사용 영향은 미미).
    """
    filtered = []
    seen = set()
    for p in race_payouts:
        key = (p.get('race_id'), p.get('bet_type'), tuple(p.get('entry_numbers') or []))
        if key not in seen:
            filtered.append(p)
            seen.add(key)
    return filtered


def filter_master_data_duplicates(parsed_master_data, existing_master_data):
    """파싱된 마스터 데이터에서 기존 데이터와 중복 제거"""
    filtered_data = {
        'horses': [],
        'jockeys': [],
        'trainers': [],
        'owners': []
    }

    # 말 데이터 필터링
    for horse in parsed_master_data.get('horses', []):
        if horse['horse_id'] not in existing_master_data['horses']:
            filtered_data['horses'].append(horse)
            existing_master_data['horses'].add(horse['horse_id'])

    # 기수 데이터 필터링
    for jockey in parsed_master_data.get('jockeys', []):
        if jockey['jk_no'] not in existing_master_data['jockeys']:
            filtered_data['jockeys'].append(jockey)
            existing_master_data['jockeys'].add(jockey['jk_no'])

    # 조교사 데이터 필터링
    for trainer in parsed_master_data.get('trainers', []):
        if trainer['trainer_id'] not in existing_master_data['trainers']:
            filtered_data['trainers'].append(trainer)
            existing_master_data['trainers'].add(trainer['trainer_id'])

    # 마주 데이터 필터링
    for owner in parsed_master_data.get('owners', []):
        if owner['owner_id'] not in existing_master_data['owners']:
            filtered_data['owners'].append(owner)
            existing_master_data['owners'].add(owner['owner_id'])

    return filtered_data

def save_normalized_data(data_dict, existing_master_data, batch_size=200):
    """정규화된 데이터를 각 테이블에 저장"""
    saved_counts = {}

    # 1. API에서 파싱된 마스터 데이터 저장 (기존 데이터와 중복 제거)
    parsed_master_data = data_dict.get('master_data', {})
    filtered_master_data = filter_master_data_duplicates(parsed_master_data, existing_master_data)
    master_saved_counts = save_missing_master_data(filtered_master_data)

    logger.info(f"마스터 데이터 추가 완료: 말 {master_saved_counts['horses']}마리, "
               f"기수 {master_saved_counts['jockeys']}명, "
               f"조교사 {master_saved_counts['trainers']}명")

    horse_attrs = summarize_horse_attrs(data_dict.get('horse_obs', []))
    if horse_attrs:
        logger.info(f"horses 나이·성별 갱신: {save_horse_attrs(horse_attrs)}마리")

    # 2. 경주 관련 데이터 저장
    races, race_entries = filter_duplicates_in_race_and_entries(
        data_dict.get('races', []), data_dict.get('race_entries', [])
    )
    race_payouts = filter_duplicates_in_payouts(data_dict.get('race_payouts', []))
    # 순서 중요: race_payouts.race_id가 races(race_id) FK라서 races가 먼저
    # upsert되어야 한다 (dict는 삽입 순서를 보존함).
    table_data = {
        'races': races,
        'race_entries': race_entries,
        'race_payouts': race_payouts,
    }

    for table_name, records in table_data.items():
        if not records:
            logger.info(f"{table_name}: 저장할 데이터 없음")
            saved_counts[table_name] = 0
            continue

        logger.info(f"{table_name} 저장 시작: {len(records)}개")
        saved_counts[table_name] = save_table_data(table_name, records, batch_size)
        time.sleep(0.2)  # API 제한 방지

    # 마스터 데이터 카운트도 포함
    saved_counts.update(master_saved_counts)
    return saved_counts

def save_table_data(table_name, data, batch_size=200):
    """특정 테이블에 데이터 저장"""
    supabase = get_client()
    try:
        total_saved = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                # updated_at 컬럼이 있는 테이블만 타임스탬프 추가
                if table_name not in TABLES_WITHOUT_UPDATED_AT:
                    current_time = datetime.now().isoformat()
                    for record in batch:
                        if 'updated_at' not in record:
                            record['updated_at'] = current_time

                # Supabase에 데이터 삽입 (upsert 사용)
                on_conflict = TABLE_ON_CONFLICT.get(table_name)
                if on_conflict:
                    result = supabase.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
                else:
                    result = supabase.table(table_name).upsert(batch).execute()

                batch_count = len(batch)
                total_saved += batch_count

                logger.info(f"{table_name} 배치 {batch_num}: {batch_count}개 저장 완료 (누적: {total_saved}개)")

                time.sleep(0.1)

            except Exception as batch_error:
                error_msg = str(batch_error)
                logger.warning(f"{table_name} 배치 {batch_num} 실패, 개별 저장 시도: {error_msg[:80]}")
                individual_saved = save_individual_records(table_name, batch)
                total_saved += individual_saved

        logger.info(f"{table_name} 저장 완료: 총 {total_saved}개")
        return total_saved

    except Exception as e:
        logger.error(f"{table_name} 저장 중 오류: {str(e)}")
        return 0

def save_individual_records(table_name, records):
    """개별 레코드 저장 (배치 실패 시 백업)"""
    supabase = get_client()
    saved_count = 0

    for record in records:
        try:
            if table_name not in TABLES_WITHOUT_UPDATED_AT:
                record['updated_at'] = datetime.now().isoformat()
            on_conflict = TABLE_ON_CONFLICT.get(table_name)
            if on_conflict:
                supabase.table(table_name).upsert(record, on_conflict=on_conflict).execute()
            else:
                supabase.table(table_name).upsert(record).execute()
            saved_count += 1

            time.sleep(0.02)
        except Exception as e:
            if "duplicate key" not in str(e).lower() and "foreign key" not in str(e).lower():
                logger.error(f"{table_name} 개별 저장 실패: {str(e)}")
            else:
                logger.debug(f"{table_name} 개별 저장 스킵 (중복 또는 외래키): {str(e)}")

    return saved_count

def fetch_pages_sequential(start_page=1, max_pages=20, start_date=None, end_date=datetime.now().strftime('%Y%m%d')):
    """순차적으로 경주 결과 데이터 수집"""
    all_race_data = []
    empty_pages = 0

    # 서울은 두 번 수집한다: SeoulRace_1(두수·주로 등 경주 정보) 다음에
    # 공용 API(말별 단승 배당). 중복 행은 저장 전에 win_odds만 병합된다.
    sources = [(1, False), (1, True), (2, False), (3, False)]
    for meet_code, common_api in sources:
        if meet_code == 2:
            print("제주 경주 결과 수집 시작")
        elif meet_code == 3:
            print("부산 경주 결과 수집 시작")
        elif common_api:
            print("서울 단승 배당 수집 시작")
        elif meet_code == 1:
            print("서울 경주 결과 수집 시작")

        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        while current <= end:
            for page in range(start_page, start_page + max_pages):
                try:
                    api_data, _ = fetch_race_data_all(page=page, meet_code=meet_code, end_date=current.strftime('%Y%m'), common_api=common_api)
                    if api_data:
                        parsed_data = parse_and_normalize_race_data(api_data)
                        race_entries = parsed_data.get('race_entries', [])

                        if race_entries:
                            all_race_data.append(parsed_data)
                            logger.info(f"페이지 {page}: {len(race_entries)}개 경주 기록 수집")
                            empty_pages = 0
                        else:
                            empty_pages += 1
                            logger.info(f"페이지 {page}: 데이터 없음")
                    else:
                        empty_pages += 1

                    if empty_pages >= 5:
                        logger.info("연속으로 빈 페이지가 많아 수집을 중단합니다.")
                        break

                    time.sleep(0.2)

                except Exception as e:
                    logger.error(f"페이지 {page} 처리 중 오류: {str(e)}")
                    continue

            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)


    # 모든 데이터 통합
    if all_race_data:
        combined_data = {
            'races': [],
            'race_entries': [],
            'race_payouts': [],
            'horse_obs': [],
            'master_data': {
                'horses': [],
                'jockeys': [],
                'trainers': [],
                'owners': []
            }
        }

        for data in all_race_data:
            for key in ['races', 'race_entries', 'race_payouts', 'horse_obs']:
                combined_data[key].extend(data.get(key, []))

            # 마스터 데이터도 통합
            master_data = data.get('master_data', {})
            for master_key in ['horses', 'jockeys', 'trainers', 'owners']:
                combined_data['master_data'][master_key].extend(master_data.get(master_key, []))

        return combined_data

    return {}

def collecting_race_results(start_page=1, max_pages=20, start_date=None, end_date=datetime.now().strftime('%Y%m%d')):
    """정규화된 구조로 경주 결과 수집 메인 함수. 수집한 가장 이른 race_date를 반환(없으면 None)."""
    logger.info("경주 결과 수집을 시작합니다...")

    # 기존 마스터 데이터 캐시 생성
    existing_master_data = get_existing_master_data()

    # 데이터 수집
    start_time = time.time()
    race_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages, start_date=start_date, end_date=end_date)

    if not race_data or not race_data.get('race_entries'):
        logger.warning("수집된 경주 결과가 없습니다.")
        return None

    total_entries = len(race_data.get('race_entries', []))
    total_races = len(race_data.get('races', []))
    logger.info(f"총 {total_races}개 경주, {total_entries}개 경주 기록을 수집했습니다.")

    # 데이터 저장
    saved_counts = save_normalized_data(race_data, existing_master_data, batch_size=150)

    if saved_counts:
        end_time = time.time()
        logger.info(f"경주 결과 수집 완료! 소요시간: {end_time - start_time:.2f}초")
    else:
        logger.error("경주 결과 저장 실패")
        return None

    return min(e['race_date'] for e in race_data['race_entries'])

if __name__ == "__main__":
    import sys
    from datetime import datetime

    def _parse_date_arg(s: str) -> str:
        try:
            datetime.strptime(s, "%Y%m%d")
        except ValueError:
            print(f"[오류] 날짜 형식이 잘못됐습니다: '{s}' → YYYYMMDD 형식으로 입력하세요. (예: 20260613)")
            sys.exit(1)
        return s

    today = datetime.now().strftime('%Y%m%d')
    # 사용법: python -m library.collector_races [시작일 종료일]
    # 예)    python -m library.collector_races 20240101 20241231
    if len(sys.argv) == 3:
        start, end = _parse_date_arg(sys.argv[1]), _parse_date_arg(sys.argv[2])
    elif len(sys.argv) == 2:
        start = end = _parse_date_arg(sys.argv[1])
    else:
        start = end = today
    logger.info(f"수집 기간: {start} ~ {end}")
    first_date = collecting_race_results(start_page=1, max_pages=50, start_date=start, end_date=end)
    from core.history_rebuilder import rebuild_after_collect
    rebuild_after_collect(first_date)
