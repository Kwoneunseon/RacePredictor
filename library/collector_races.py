# -*- coding: utf-8 -*-
import requests
from datetime import datetime
import time
import logging
from supabase import create_client, Client

from Utils import parse_date, safe_int, safe_float, safe_str
from _const import API_KEY, SUPABASE_URL, SUPABASE_KEY
from .collector_horse import fetch_single_horse_data as horse_fetch_page, save_to_supabase_batch as save_horse_data, parse_horse_data
from .collector_jockeys import fetch_single_jockey_data as jockey_fetch_page, save_to_supabase_batch as save_jockey_data, parse_jockey_data  
from .collector_trainers import fetch_single_trainer_data as trainer_fetch_page, save_to_supabase_batch as save_trainer_data, parse_trainer_data

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HTTP 요청 로그 숨기기
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.getLogger("postgrest").setLevel(logging.WARNING)

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    
def fetch_race_data_jeju(page=1, per_page=1000 ,meet_code = 2, start_date=None, end_date=None):
    """제주 부경 경주 결과 API에서 데이터 가져오는 함수"""
    params = {
        "serviceKey": API_KEY,
        "pageNo": page,
        "numOfRows": per_page,
        "meet": meet_code,
        "rc_month": end_date[:6],  # 종료일
        "_type": 'json'

    }

    try:
        endpoint = "http://apis.data.go.kr/B551015/API4_3/raceResult_3"
        response = requests.get(endpoint, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data, page
        elif response.status_code == 429:
            logger.warning(f"API 호출 제한 - 페이지 {page}, 5초 대기 후 재시도")
            time.sleep(5)
            return fetch_race_data_jeju(page, per_page, start_date, end_date)
        else:
            logger.error(f"API 호출 실패 - 페이지 {page}: {response.status_code}")
            return None, page

    except Exception as e:
        logger.error(f"API 호출 중 오류 발생 - 페이지 {page}: {str(e)}")
        return None, page

def get_existing_master_data():
    """기존 마스터 데이터 개수 확인"""
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
            ('owners', 'owner_id', 'owners')
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
                horse_id = missing_data['horses'][batch_start]['horse_id']
                horse_data = horse_fetch_page(horse_id)
                if horse_data:
                    horse_datas.append(horse_data)
            save_horse_data(horse_datas)
            saved_counts['horses'] = len(missing_data['horses'])
        else:
            saved_counts['horses'] = 0
            
        # 기수 정보 저장
        if missing_data['jockeys']:
            jockey_datas = []   
            logger.info(f"새로운 기수 {len(missing_data['jockeys'])}명 추가")
            for batch_start in range(0, len(missing_data['jockeys'])):
                jockey_no = missing_data['jockeys'][batch_start]['jk_no']
                api_data, _ = jockey_fetch_page(jockey_no)
                jockey_data = parse_jockey_data(api_data)
                if jockey_data:
                    jockey_datas.append(jockey_data[0])

            save_jockey_data(jockey_datas)
            saved_counts['jockeys'] = len(missing_data['jockeys'])
        else:
            saved_counts['jockeys'] = 0
            
        # 조교사 정보 저장
        if missing_data['trainers']:
            trainer_datas = []
            logger.info(f"새로운 조교사 {len(missing_data['trainers'])}명 추가")
            for batch_start in range(0, len(missing_data['trainers'])):
                trainer_id = missing_data['trainers'][batch_start]['trainer_id']
                api_data, _ = trainer_fetch_page(trainer_id)
                trainer_data = parse_trainer_data(api_data)
                if trainer_data:
                    trainer_datas.append(trainer_data[0])

            save_trainer_data(trainer_datas)
            saved_counts['trainers'] = len(missing_data['trainers'])
        else:
            saved_counts['trainers'] = 0
            
        # 마주 정보 저장
        # if missing_data['owners']:
        #     logger.info(f"새로운 마주 {len(missing_data['owners'])}명 추가")
        #     for batch_start in range(0, len(missing_data['owners']), 100):
        #         batch = missing_data['owners'][batch_start:batch_start + 100]
        #         try:
        #             result = supabase.table('owners').insert(batch).execute()
        #             time.sleep(0.1)
        #         except Exception as e:
        #             logger.error(f"마주 정보 배치 저장 실패: {str(e)}")
        #             # 개별 저장 시도
        #             for owner in batch:
        #                 try:
        #                     supabase.table('owners').insert(owner).execute()
        #                     time.sleep(0.02)
        #                 except Exception as individual_error:
        #                     logger.debug(f"마주 개별 저장 실패 (중복일 수 있음): {str(individual_error)}")
        #     saved_counts['owners'] = len(missing_data['owners'])
        # else:
        saved_counts['owners'] = 0
            
        return saved_counts
        
    except Exception as e:
        logger.error(f"마스터 데이터 저장 실패: {str(e)}")
        return {'horses': 0, 'jockeys': 0, 'trainers': 0, 'owners': 0}

def parse_and_normalize_race_data(api_response):
    """API 응답을 파싱하여 정규화된 테이블 구조로 변환"""
    races = []
    race_entries = []
    betting_odds = []
    master_data = {
        'horses': [],
        'jockeys': [],
        'trainers': [],
        'owners': []
    }
    def parse_horse_weight_simple(wg_hr_str):
        if not wg_hr_str or wg_hr_str == "()":
            return 469, 0 #평균 마체중
        
        # 괄호 분리
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
                processed_racetracks = set()
                processed_races = set()
                processed_horses = set()
                processed_jockeys = set()
                processed_trainers = set()
                processed_owners = set()
                
                for item in items:
                    race_date = parse_date(item.get('rcDate')) #경마일자
                    meet_code = safe_str(item.get('meet'))  #경마장 명  
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
                    
                    if not all([race_date, meet_code, race_no, horse_id, jk_no, trainer_id, owner_id]):
                        continue                    
                    
                    # 2. 경주 기본 정보
                    race_key = (race_date, meet_code, race_no)
                    if race_key not in processed_races:
                        races.append({
                            'race_date': race_date,
                            'meet_code': meet_code,
                            'race_id': race_no,
                            'race_distance': safe_int(item.get('rcDist')),  #경주거리
                            'race_grade': safe_str(item.get('rcGrade')) if item.get('rcGrade') else safe_str(item.get('rcName')), #경주등급
                            'race_age': safe_str(item.get('rcAge')) if item.get('rcAge') else safe_str(item.get('ageCond')), #연령조건
                            'race_sex': safe_str(item.get('rcSex')) if item.get('rcSex') else safe_str(item.get('sexCond')), #성별조건
                            'race_type': safe_str(item.get('rcCode')) if item.get('rcCode') else safe_str(item.get('rcName')), #대상경주명(일반, 특별, 오픈 등)
                            #'race_category': safe_str(item.get('rcRank')),
                            'race_kind': safe_str(item.get('rankKind')) if item.get('rankKind') else '0',    #경주종류
                            'race_flag': safe_str(item.get('rcFrflag')) if item.get('rcFrflag') else safe_str(item.get('name')),    #경주구분(국산, 혼합, 외산 등)
                            'night_race': safe_str(item.get('rcNrace')) if item.get('rcNrace') else '일반',    #야간경주 여부
                            'track_condition': safe_str(item.get('track')) if item.get('track') else '', #경주로상태
                            'weather': safe_str(item.get('weath')) if item.get('weath') else safe_str(item.get('weather')),         #날씨
                            'total_horses': safe_int(item.get('rcVtdusu'))  if item.get('rcVtdusu') else 11, #총 출전마 수
                            'planned_horses': safe_int(item.get('rcPlansu')) if item.get('rcPlansu') else 11,#계획 출전마 수
                            'weight_type': safe_int(item.get('rcBudam')) if item.get('rcBudam') else 2,   #부담구분(1:마령, 2:별정, 3:핸디캡)
                            'race_status': safe_str(item.get('noracefl')) if item.get('noracefl') else '정상' ,  #경주상태(정상/취소 등)
                            'is_divided': safe_int(item.get('divide')) if item.get('divide') else 0,     #분할경주 여부
                            'race_days': safe_int(item.get('rundayth')) if item.get('rundayth') else item.get('ilsu'),    #경주일수 (말 출전한 총 일수)
                            #'special_code_a': safe_str(item.get('rcSpcba')),
                            #'special_code_b': safe_str(item.get('rcSpcbu')),
                            'estimated_odds': safe_float(item.get('rc10dusu')) if item.get('rc10dusu') else item.get('winOdds') # 예상배당률
                        })
                        processed_races.add(race_key)                   

                    
                    # 마스터 데이터 수집 (API에서 가져온 실제 정보 사용)
                    if horse_id not in processed_horses:
                        master_data['horses'].append({
                            'horse_id': horse_id,
                            'name': horse_name or f'Horse_{horse_id}',  # name 컬럼으로 변경
                        })
                        processed_horses.add(horse_id)
                    
                    if jk_no not in processed_jockeys:
                        master_data['jockeys'].append({
                            'jk_no': jk_no,  # jk_no 컬럼 사용
                            'name': jockey_name or f'Jockey_{jk_no}'  # name 컬럼으로 변경
                        })
                        processed_jockeys.add(jk_no)
                    
                    if trainer_id not in processed_trainers:
                        master_data['trainers'].append({
                            'trainer_id': trainer_id,
                            'trainer_name': trainer_name or f'Trainer_{trainer_id}'  # trainer_name 컬럼 유지
                        })
                        processed_trainers.add(trainer_id)
                    
                    if owner_id and owner_id not in processed_owners:
                        master_data['owners'].append({
                            'owner_id': owner_id,
                            'owner_name': owner_name or f'Owner_{owner_id}'  # owner_name 컬럼 유지
                        })
                        processed_owners.add(owner_id)
                    
                    # 3. 경주 참가 기록
                    
                    horse_weight, horse_weight_idff = parse_horse_weight_simple(item.get('wgHr'))

                    race_entries.append({
                        'race_date': race_date,
                        'meet_code': meet_code,
                        'race_id': race_no,
                        'horse_id': horse_id,
                        'jk_no': jk_no,
                        'trainer_id': trainer_id,
                        #'owner_id': owner_id,
                        'entry_number': safe_int(item.get('rcChul')) if item.get('rcChul') else safe_int(item.get('chulNo')),   #출전번호
                        'horse_race_days': safe_int(item.get('rundayth')) if item.get('rundayth') else item.get('ilsu'),    #경주일수 (말 출전한 총 일수)
                        'horse_weight': horse_weight,     #마체중
                        'horse_weight_diff': horse_weight_idff,  #마체중 증감
                        'budam' : item.get('budam'), 
                        'budam_weight': safe_int(item.get('wgBudam')),  #부담중량
                        'horse_rating': safe_float(item.get('rating')),  #말평점
                        'final_rank': safe_int(item.get('rcOrd')) if item.get('rcOrd') else safe_int(item.get('ord')),      #순위
                        'finish_time': safe_float(item.get('rcTime')),  #경주시간(초단위)
                        #'diff_total': safe_float(item.get('diffTot')),  #1등과 시간차이 누적 
                        # 'diff_2nd': safe_float(item.get('rcDiff2')),
                        # 'diff_3rd': safe_float(item.get('rcDiff3')),
                        # 'diff_4th': safe_float(item.get('rcDiff4')),
                        # 'diff_5th': safe_float(item.get('rcDiff5')),
                        # 'prize_money': safe_int(item.get('chaksun'))
                    })
                    
                        
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")
        
    return {
        'races': races,
        'race_entries': race_entries,
        'master_data': master_data  # 새로 추가
    }

def filter_duplicates_in_race_and_entries(races, race_entries):
    """race, race_entries 리스트 내 중복 제거 (race_date + race_id + meet_code, entry_key 기준)"""
    filtered_races = []
    seen_race_keys = set()
    for race in races:
        race_key = (race.get('race_date'), race.get('race_id'), race.get('meet_code'))  # race_date + race_id + meet_code 조합
        if race_key not in seen_race_keys:
            filtered_races.append(race)
            seen_race_keys.add(race_key)

    filtered_entries = []
    seen_entry_keys = set()
    for entry in race_entries:
        entry_key = (
            entry.get('meet_code'),
            entry.get('race_date'),  
            entry.get('race_id'),
            entry.get('horse_id')
        )
        if entry_key not in seen_entry_keys:
            filtered_entries.append(entry)
            seen_entry_keys.add(entry_key)

    return filtered_races, filtered_entries


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
        if jockey['jk_no'] not in existing_master_data['jockeys']:  # jk_no로 변경
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
               f"조교사 {master_saved_counts['trainers']}명, "
               f"마주 {master_saved_counts['owners']}명")
    
    # 2. 경주 관련 데이터 저장
    # 중복 제거 및 저장할 테이블 정의
    races, race_entries = filter_duplicates_in_race_and_entries(
        data_dict.get('races', []), data_dict.get('race_entries', [])
    )
    table_data = {
        'races': races,
        'race_entries': race_entries
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
    try:
        total_saved = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # 각 배치에 타임스탬프 추가
                current_time = datetime.now().isoformat()
                for record in batch:
                    if 'updated_at' not in record:
                        record['updated_at'] = current_time
                
                # Supabase에 데이터 삽입 (upsert 사용)
                result = supabase.table(table_name).upsert(batch).execute()
                
                batch_count = len(batch)
                total_saved += batch_count
                
                logger.info(f"{table_name} 배치 {batch_num}: {batch_count}개 저장 완료 (누적: {total_saved}개)")
                
                time.sleep(0.1)
                
            except Exception as batch_error:
                error_msg = str(batch_error)
                if "duplicate key" in error_msg.lower():
                    logger.info(f"{table_name} 배치 {batch_num}: 중복 데이터 스킵")
                else:
                    logger.error(f"{table_name} 배치 {batch_num} 저장 실패: {error_msg}")
                    # 개별 저장 시도
                    individual_saved = save_individual_records(table_name, batch)
                    total_saved += individual_saved
        
        logger.info(f"{table_name} 저장 완료: 총 {total_saved}개")
        return total_saved
        
    except Exception as e:
        logger.error(f"{table_name} 저장 중 오류: {str(e)}")
        return 0

def save_individual_records(table_name, records):
    """개별 레코드 저장 (배치 실패 시 백업)"""
    saved_count = 0
    
    for record in records:
        try:
            record['updated_at'] = datetime.now().isoformat()
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
    
    #제주, 부산 경주 결과 수집    
    meet_codes = [1,2,3]
    for meet_code in meet_codes:
        if meet_code == 2:
            print("제주 경주 결과 수집 시작")
        elif meet_code == 3:
            print("부산 경주 결과 수집 시작")
        elif meet_code == 1:
            print("서울 경주 결과 수집 시작")
        
        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        while current <= end:        
            for page in range(start_page, start_page + max_pages):
                try:
                    api_data, _ = fetch_race_data_jeju(page=page, meet_code=meet_code, end_date=current.strftime('%Y%m'))                    
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
  
    # #서울 경주 결과 수집
    # for page in range(start_page, start_page + max_pages):
    #     try:
    #         api_data, _ = fetch_race_data(page=page, start_date=start_date, end_date=end_date)
            
    #         if api_data:
    #             parsed_data = parse_and_normalize_race_data(api_data)
    #             race_entries = parsed_data.get('race_entries', [])
                
    #             if race_entries:
    #                 all_race_data.append(parsed_data)
    #                 logger.info(f"페이지 {page}: {len(race_entries)}개 경주 기록 수집")
    #                 empty_pages = 0
    #             else:
    #                 empty_pages += 1
    #                 logger.info(f"페이지 {page}: 데이터 없음")
    #         else:
    #             empty_pages += 1
                
    #         if empty_pages >= 5:
    #             logger.info("연속으로 빈 페이지가 많아 수집을 중단합니다.")
    #             break
                
    #         time.sleep(0.2)
            
    #     except Exception as e:
    #         logger.error(f"페이지 {page} 처리 중 오류: {str(e)}")
    #         continue

    
    # 모든 데이터 통합
    if all_race_data:
        combined_data = {
            'races': [],
            'race_entries': [],
            'master_data': {
                'horses': [],
                'jockeys': [],
                'trainers': [],
                'owners': []
            }
        }
        
        for data in all_race_data:
            for key in ['races', 'race_entries']:
                combined_data[key].extend(data.get(key, []))
            
            # 마스터 데이터도 통합
            master_data = data.get('master_data', {})
            for master_key in ['horses', 'jockeys', 'trainers', 'owners']:
                combined_data['master_data'][master_key].extend(master_data.get(master_key, []))
        
        return combined_data
    
    return {}

def collecting_race_results(start_page=1, max_pages=20, start_date=None, end_date=datetime.now().strftime('%Y%m%d')):
    """정규화된 구조로 경주 결과 수집 메인 함수"""
    logger.info("정규화된 경주 결과 수집을 시작합니다...")
    logger.info(f"시작 페이지: {start_page}, 최대 페이지: {max_pages}")
    
    # 기존 마스터 데이터 캐시 생성 qqq
    existing_master_data = get_existing_master_data()
    start_time = time.time()
    
    # 데이터 수집
    race_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages, start_date= start_date, end_date=end_date)
    
    if not race_data or not race_data.get('race_entries'):
        logger.warning("수집된 경주 결과가 없습니다.")
        return
    
    total_entries = len(race_data.get('race_entries', []))
    total_races = len(race_data.get('races', []))
    logger.info(f"총 {total_races}개 경주, {total_entries}개 경주 기록을 수집했습니다.")
    
    # 데이터 저장
    saved_counts = save_normalized_data(race_data, existing_master_data, batch_size=150)
    
    if saved_counts:
        end_time = time.time()
        logger.info(f"경주 결과 수집 완료! 소요시간: {end_time - start_time:.2f}초")
        
        # 저장 결과 요약
        logger.info("=== 저장 결과 요약 ===")
        for table_name, count in saved_counts.items():
            logger.info(f"{table_name}: {count}개")
    else:
        logger.error("경주 결과 저장 실패")

def check_data_quality():
    """데이터 품질 확인"""
    try:
        # 경주 결과 통계
        race_entries = supabase.table('race_entries').select('*', count='exact').execute()
        races = supabase.table('races').select('*', count='exact').execute()
        
        entries_count = race_entries.count if hasattr(race_entries, 'count') else len(race_entries.data)
        races_count = races.count if hasattr(races, 'count') else len(races.data)
        
        logger.info(f"현재 DB 상태: {races_count}개 경주, {entries_count}개 경주 기록")
        
        # 마스터 데이터 확인
        horses_count = supabase.table('horses').select('*', count='exact').execute().count or 0
        jockeys_count = supabase.table('jockeys').select('*', count='exact').execute().count or 0
        trainers_count = supabase.table('trainers').select('*', count='exact').execute().count or 0
        owners_count = supabase.table('owners').select('*', count='exact').execute().count or 0
        
        logger.info(f"마스터 데이터: 말 {horses_count}마리, 기수 {jockeys_count}명, "
                   f"조교사 {trainers_count}명, 마주 {owners_count}명")
        
        # 최근 데이터 확인
        recent_races = supabase.table('race_entries').select(
            'race_date, meet_code, race_no, horse_id, final_rank'
        ).order('race_date', desc=True).limit(5).execute()
        
        if recent_races.data:
            logger.info("최근 경주 결과:")
            for entry in recent_races.data:
                logger.info(f"  - {entry['race_date']} {entry['meet_code']} {entry['race_no']}R: "
                          f"말 {entry['horse_id']} ({entry['final_rank']}위)")
        
    except Exception as e:
        logger.error(f"데이터 품질 확인 실패: {str(e)}")

# 실행 예시
if __name__ == "__main__":
    # 경주 결과 수집
    #collecting_race_results_normalized(start_page=1, max_pages=10)
    
    # 데이터 품질 확인
    check_data_quality()