# -*- coding: utf-8 -*-
import requests
from datetime import datetime, date
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import API_KEY
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

def fetch_trainer_data(page=1, per_page=1000, options={}):
    '''
    조교사 정보 API에서 데이터 가져오는 함수 (최적화 버전)
    '''
    params = {
        "serviceKey": API_KEY,
        "page": page,
        "numOfRows": per_page,
        "_type": 'json',
        **options
    }

    try:
        endpoint = "https://apis.data.go.kr/B551015/API19_1/trainerInfo_1"
        response = requests.get(endpoint, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data, page
        elif response.status_code == 429:
            logger.warning(f"API 호출 제한 - 페이지 {page}, 5초 대기 후 재시도")
            time.sleep(5)
            return fetch_trainer_data(page, per_page, options)
        else:
            logger.error(f"API 호출 실패 - 페이지 {page}: {response.status_code}")
            return None, page

    except Exception as e:
        logger.error(f"API 호출 중 오류 발생 - 페이지 {page}: {str(e)}")
        return None, page

def parse_trainer_data(api_response):
    """
    API 응답을 파싱하여 테이블 구조에 맞는 데이터로 변환
    """
    trainers_data = []

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
                    trainer_data = {
                        'trainer_id': str(item.get('trNo', '')),           # 조교사번호
                        'name': item.get('trName', ''),                    # 조교사명 (new schema: name)
                        # 통산 기록
                        'rc_cnt_t': int(item.get('rcCntT', 0)) if item.get('rcCntT') else 0,     # 통산출주횟수
                        'ord1_cnt_t': int(item.get('ord1CntT', 0)) if item.get('ord1CntT') else 0, # 통산1위횟수
                    }

                    # 데이터 검증 (필수 필드)
                    if not trainer_data['trainer_id'] or not trainer_data['name']:
                        print(f"필수 정보 누락: {trainer_data}")
                        continue

                    trainers_data.append(trainer_data)
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")

    return trainers_data

def save_to_supabase_batch(trainers: list) -> int:
    if not trainers:
        return 0
    schema_keys = {"trainer_id", "name", "rc_cnt_t", "ord1_cnt_t"}
    rows = [{k: v for k, v in t.items() if k in schema_keys} for t in trainers if t and t.get("trainer_id")]
    upsert("trainers", rows, "trainer_id")
    return len(rows)

def remove_duplicates_from_collected_data(trainers_data):
    """
    수집된 데이터 내에서 중복 제거 (같은 trainer_id가 여러 번 수집된 경우)
    """
    seen_ids = set()
    unique_trainers = []
    duplicate_in_batch = 0

    for trainer in trainers_data:
        trainer_id = trainer.get('trainer_id')
        if trainer_id not in seen_ids:
            seen_ids.add(trainer_id)
            unique_trainers.append(trainer)
        else:
            duplicate_in_batch += 1

    if duplicate_in_batch > 0:
        logger.info(f"수집 데이터 내 중복 제거: {duplicate_in_batch}건 제거, {len(unique_trainers)}건 유지")

    return unique_trainers

def fetch_page_data(page):
    """
    단일 페이지 데이터를 가져오는 헬퍼 함수 (멀티스레딩용)
    """
    api_data, page_num = fetch_trainer_data(page=page)
    if api_data:
        trainers_data = parse_trainer_data(api_data)
        return trainers_data, page_num
    return [], page_num

def fetch_pages_parallel(start_page=1, max_pages=100, max_workers=3):
    """
    병렬로 여러 페이지의 데이터를 가져오기 (Supabase용으로 worker 수 감소)
    """
    all_trainers_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        page_range = range(start_page, min(start_page + max_pages, 1000))

        future_to_page = {executor.submit(fetch_page_data, page): page for page in page_range}

        empty_pages = 0
        for future in as_completed(future_to_page):
            try:
                trainers_data, page_num = future.result()
                if trainers_data:
                    all_trainers_data.extend(trainers_data)
                    logger.info(f"페이지 {page_num}: {len(trainers_data)}마리 수집")
                    empty_pages = 0
                else:
                    empty_pages += 1
                    logger.info(f"페이지 {page_num}: 데이터 없음")

                if empty_pages >= 10:
                    logger.info("연속으로 빈 페이지가 많아 수집을 중단합니다.")
                    break

            except Exception as e:
                logger.error(f"페이지 처리 중 오류: {str(e)}")

    return all_trainers_data

def fetch_pages_sequential(start_page=1, max_pages=50, options={}):
    """
    순차적으로 데이터를 가져오기 (안전한 버전)
    """
    all_trainers_data = []
    empty_pages = 0

    for page in range(start_page, start_page + max_pages):
        try:
            api_data, _ = fetch_trainer_data(page=page, options=options)

            if api_data:
                trainers_data = parse_trainer_data(api_data)
                if trainers_data:
                    all_trainers_data.extend(trainers_data)
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

    return all_trainers_data

def collecting_trainer_supabase(method='sequential', start_page=1, max_pages=50):
    """
    Supabase용 메인 실행 함수 (upsert 버전 - 업데이트/삽입)
    :param method: 'sequential' 또는 'parallel'
    :param start_page: 시작 페이지
    :param max_pages: 최대 페이지 수
    """
    logger.info("조교사 정보 수집을 시작합니다 (upsert 버전 - 업데이트/삽입)...")
    logger.info(f"수집 방법: {method}, 시작 페이지: {start_page}, 최대 페이지: {max_pages}")

    start_time = time.time()

    # 데이터 수집 방법 선택
    if method == 'parallel':
        trainers_data = fetch_pages_parallel(start_page=start_page, max_pages=max_pages, max_workers=2)
    else:
        trainers_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)

    if not trainers_data:
        logger.warning("수집된 데이터가 없습니다.")
        return 0

    logger.info(f"총 {len(trainers_data)}명의 조교사 정보를 수집했습니다.")

    # 수집된 데이터 내에서 중복 제거
    trainers_data = remove_duplicates_from_collected_data(trainers_data)

    saved = save_to_supabase_batch(trainers_data)
    end_time = time.time()
    logger.info(f"조교사 정보 수집 완료! 소요시간: {end_time - start_time:.2f}초, 저장: {saved}명")
    return saved

def fetch_single_trainer_data(trainer_id):
    """
    특정 조교사ID로 조교사정보 API 호출 (fetch_trainer_data 래퍼)
    """
    return fetch_trainer_data(page=1, per_page=1, options={"tr_no": trainer_id})

def fetch_all_trainers(start_page: int = 1, max_pages: int = 50) -> int:
    trainers = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)
    if not trainers:
        return 0
    trainers = remove_duplicates_from_collected_data(trainers)
    return save_to_supabase_batch(trainers)

# 실행 옵션들
if __name__ == "__main__":
    # 조교사정보 수집 (upsert 방식)
    collecting_trainer_supabase(method='sequential', start_page=1, max_pages=10)
