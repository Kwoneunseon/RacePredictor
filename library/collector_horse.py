# -*- coding: utf-8 -*-
import requests
from datetime import datetime, date
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import API_KEY
from library.parsing import safe_int, safe_float
from Utils.database import upsert
from Utils.logger import get_logger
logger = get_logger(__name__)

def calculate_age(birth_str: str):
    '''
    생년월일 문자열을 받아서 나이를 계산하는 함수
    :param birth_str: 생년월일 문자열 (형식: YYYYMMDD)
    :return: 나이 (정수)
    '''
    try:
        birth_date = datetime.strptime(str(birth_str), "%Y%m%d").date()
        today = date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except:
        return 0

def fetch_horse_data(page=1, per_page=1000, options={}):
    '''
    경주마 정보 API에서 데이터 가져오는 함수 (최적화 버전)
    '''
    params = {
        "serviceKey": API_KEY,
        "page": page,
        "numOfRows": per_page,
        "_type": 'json',
        **options
    }

    try:
        endpoint = "https://apis.data.go.kr/B551015/API8_2/raceHorseInfo_2"
        response = requests.get(endpoint, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data, page
        elif response.status_code == 429:
            logger.warning(f"API 호출 제한 - 페이지 {page}, 5초 대기 후 재시도")
            time.sleep(5)
            return fetch_horse_data(page, per_page, options)
        else:
            logger.error(f"API 호출 실패 - 페이지 {page}: {response.status_code}")
            return None, page

    except Exception as e:
        logger.error(f"API 호출 중 오류 발생 - 페이지 {page}: {str(e)}")
        return None, page

def parse_horse_data(api_response):
    """
    API 응답을 파싱하여 테이블 구조에 맞는 데이터로 변환
    """
    horses_data = []

    try:
        if 'response' in api_response and 'body' in api_response['response']:
            items = api_response['response']['body']
            if items:
                items = items.get('items', [])
            if items:
                items = items.get('item', [])

                if isinstance(items, dict):
                    items = [items]

                for item in items:
                    horse_data = {
                        'horse_id': str(item.get('hrNo', 0)),  # TEXT 타입으로 변환
                        'name': item.get('hrName', ''),
                        'age': calculate_age(item.get('birthday', 0)),
                        'gender': item.get('sex', ''),
                        'rank': item.get('rank', '')
                    }
                    horses_data.append(horse_data)
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")

    return horses_data


def fetch_additional_horse_data(horse_id:str):
    '''
    경주마 성적 API에서 추가적인 데이터 가져오는 함수 (최적화 버전)
    '''
    params = {
        "serviceKey": API_KEY,
        "hr_no": horse_id,
        "_type": 'json',
        }

    try:
        endpoint = "https://apis.data.go.kr/B551015/API15_2/raceHorseResult_2"
        response = requests.get(endpoint, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"API 호출 실패 - 페이지 {horse_id}: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"API 호출 중 오류 발생 - 말 ID {horse_id} {str(e)}")
        return None


def parse_additional_horse_data(api_response):
    """
    API 응답을 파싱하여 테이블 구조에 맞는 데이터로 변환 (추가적인 정보)
    """
    additional_data = {
        'total_races' : 0,
        'total_win_rate': 0.0,
        'total_place_rate': 0.0,
        'year_races': 0,
        'year_win_rate': 0.0,
        'year_place_rate': 0.0
    }


    try:
        if 'response' in api_response and 'body' in api_response['response']:
            items = api_response['response']['body']
            if items:
                items = items.get('items', [])
            if items:
                items = items.get('item', [])

                if isinstance(items, dict):
                    items = [items]

                for item in items:
                    additional_data['total_races'] = safe_int(item.get('rcCntT', 0))
                    additional_data['total_win_rate'] = safe_float(item.get('winRateT', 0.0))
                    additional_data['total_place_rate'] = safe_float(item.get('qnlRateT', 0.0))

                    additional_data['year_races'] = safe_int(item.get('rcCntY', 0))
                    additional_data['year_win_rate'] = safe_float(item.get('winRateY', 0.0))
                    additional_data['year_place_rate'] = safe_float(item.get('qnlRateY', 0.0))

    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")

    return additional_data

def combine_horse_data_with_details(horse_basic_data):
    '''기본 말 정보에 추가 정보들을 조합하는 함수'''
    combined_data = []

    for horse in horse_basic_data:
        horse_id = horse['horse_id']
        logger.info(f"말 {horse_id}의 추가 정보 수집 중...")

        # 기본 정보로 시작
        combined_horse = horse.copy()

        # 경주 기록 정보 추가
        additional_data = fetch_additional_horse_data(horse_id)
        if additional_data:
            horse_additional_info = parse_additional_horse_data(additional_data)
            combined_horse.update(horse_additional_info)

        combined_data.append(combined_horse)

        # API 호출 제한 방지
        time.sleep(0.2)

    return combined_data


def save_to_supabase_batch(horses: list) -> int:
    if not horses:
        return 0
    # Only keep columns that exist in the new schema
    schema_keys = {"horse_id", "name", "age", "gender", "rank"}
    rows = [{k: v for k, v in h.items() if k in schema_keys} for h in horses if h and h.get("horse_id")]
    upsert("horses", rows, "horse_id")
    return len(rows)


def remove_duplicates_from_collected_data(horses_data):
    """
    수집된 데이터 내에서 중복 제거 (같은 horse_id가 여러 번 수집된 경우)
    """
    seen_ids = set()
    unique_horses = []
    duplicate_in_batch = 0

    for horse in horses_data:
        horse_id = horse.get('horse_id')
        if horse_id not in seen_ids:
            seen_ids.add(horse_id)
            unique_horses.append(horse)
        else:
            duplicate_in_batch += 1

    if duplicate_in_batch > 0:
        logger.info(f"수집 데이터 내 중복 제거: {duplicate_in_batch}건 제거, {len(unique_horses)}건 유지")

    return unique_horses

def fetch_page_data(page):
    """
    단일 페이지 데이터를 가져오는 헬퍼 함수 (멀티스레딩용)
    """
    api_data, page_num = fetch_horse_data(page=page)
    if api_data:
        horses_data = parse_horse_data(api_data)
        return horses_data, page_num
    return [], page_num

def fetch_pages_parallel(start_page=1, max_pages=100, max_workers=3):
    """
    병렬로 여러 페이지의 데이터를 가져오기 (Supabase용으로 worker 수 감소)
    """
    all_horses_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        page_range = range(start_page, min(start_page + max_pages, 1000))

        future_to_page = {executor.submit(fetch_page_data, page): page for page in page_range}

        empty_pages = 0
        for future in as_completed(future_to_page):
            try:
                horses_data, page_num = future.result()
                if horses_data:
                    all_horses_data.extend(horses_data)
                    logger.info(f"페이지 {page_num}: {len(horses_data)}마리 수집")
                    empty_pages = 0
                else:
                    empty_pages += 1
                    logger.info(f"페이지 {page_num}: 데이터 없음")

                if empty_pages >= 10:
                    logger.info("연속으로 빈 페이지가 많아 수집을 중단합니다.")
                    break

            except Exception as e:
                logger.error(f"페이지 처리 중 오류: {str(e)}")

    return all_horses_data

def fetch_pages_sequential(start_page=1, max_pages=50, options={}):
    """
    순차적으로 데이터를 가져오기 (안전한 버전)
    """
    all_horses_data = []
    empty_pages = 0

    for page in range(start_page, start_page + max_pages):
        try:
            api_data, _ = fetch_horse_data(page=page, options=options)

            if api_data:
                horses_data = parse_horse_data(api_data)
                if horses_data:
                    all_horses_data.extend(horses_data)
                    empty_pages = 0
                else:
                    empty_pages += 1
            else:
                empty_pages += 1

            if empty_pages >= 5:
                logger.info("연속으로 빈 페이지가 많아 수집을 중단합니다.")
                break

            time.sleep(0.1)

        except Exception as e:
            logger.error(f"페이지 {page} 처리 중 오류: {str(e)}")
            continue

    return all_horses_data

def collecting_horse_supabase(method='sequential', start_page=1, max_pages=50):
    """
    Supabase용 메인 실행 함수 (upsert 버전 - 업데이트/삽입)
    :param method: 'sequential' 또는 'parallel'
    :param start_page: 시작 페이지
    :param max_pages: 최대 페이지 수
    """
    logger.info("경주마 정보 수집을 시작합니다 (upsert 버전 - 업데이트/삽입)...")
    logger.info(f"수집 방법: {method}, 시작 페이지: {start_page}, 최대 페이지: {max_pages}")

    start_time = time.time()

    # 데이터 수집 방법 선택
    if method == 'parallel':
        horses_data = fetch_pages_parallel(start_page=start_page, max_pages=max_pages, max_workers=2)
    else:
        horses_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)

    if not horses_data:
        logger.warning("수집된 데이터가 없습니다.")
        return 0

    logger.info(f"총 {len(horses_data)}마리의 말 정보를 수집했습니다.")

    # 수집된 데이터 내에서 중복 제거
    horses_data = remove_duplicates_from_collected_data(horses_data)

    saved = save_to_supabase_batch(horses_data)
    end_time = time.time()
    logger.info(f"경주마 정보 수집 완료! 소요시간: {end_time - start_time:.2f}초, 저장: {saved}마리")
    return saved

def fetch_single_horse_data(horse_id):
    """
    특정 말 ID로 말 정보 API 호출 (fetch_horse_data 래퍼)
    """

    api_data, _ = fetch_horse_data(page=1, per_page=1, options={"hr_no": horse_id})
    horse_data = parse_horse_data(api_data)

    if not horse_data:
        logger.warning(f"말 ID {horse_id}에 대한 데이터가 없습니다.")
        return None

    horse_data = horse_data[0]  # 단일 말 정보이므로 첫 번째 요소만 사용

    additional_data = fetch_additional_horse_data(horse_id)
    if additional_data:
        horse_additional_info = parse_additional_horse_data(additional_data)
        horse_data.update(horse_additional_info)

    return horse_data

def fetch_all_horses(start_page: int = 1, max_pages: int = 50) -> int:
    horses = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)
    if not horses:
        return 0
    horses = remove_duplicates_from_collected_data(horses)
    return save_to_supabase_batch(horses)

# 실행 옵션들
if __name__ == "__main__":
    # 말 정보 수집 (upsert 방식)
    #collecting_horse_supabase(method='sequential', start_page=1, max_pages=100)

    # 특정 말 정보 업데이트 예시
    fetch_single_horse_data("0038052")
