# -*- coding: utf-8 -*-
import requests
from datetime import datetime, date
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import API_KEY
from Utils.database import upsert
from Utils.logger import get_logger
from library.parsing import safe_int, safe_str
logger = get_logger(__name__)

def remove_duplicates_from_collected_jockeys(jockeys_data):
    """
    수집된 기수 데이터 내에서 중복 제거 (같은 jk_no가 여러 번 수집된 경우)
    """
    seen_nos = set()
    unique_jockeys = []
    duplicate_in_batch = 0

    for jockey in jockeys_data:
        jk_no = jockey.get('jk_no')
        if jk_no not in seen_nos:
            seen_nos.add(jk_no)
            unique_jockeys.append(jockey)
        else:
            duplicate_in_batch += 1

    if duplicate_in_batch > 0:
        logger.info(f"수집 데이터 내 중복 제거: {duplicate_in_batch}건 제거, {len(unique_jockeys)}건 유지")

    return unique_jockeys

def fetch_jockey_data(page=1, per_page=1000, options={}):
    '''
    기수 정보 API에서 데이터 가져오는 함수 (최적화 버전)
    '''
    params = {
        "serviceKey": API_KEY,
        "pageNo": page,
        "numOfRows": per_page,
        "_type": 'json',
        **options
    }

    try:
        endpoint = "https://apis.data.go.kr/B551015/API12_1/jockeyInfo_1"
        response = requests.get(endpoint, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data, page
        elif response.status_code == 429:  # Too Many Requests
            logger.warning(f"API 호출 제한 - 페이지 {page}, 5초 대기 후 재시도")
            time.sleep(5)
            return fetch_jockey_data(page, per_page, options)
        else:
            logger.error(f"API 호출 실패 - 페이지 {page}: {response.status_code}")
            logger.error(f"응답 내용: {response.text}")
            return None, page

    except Exception as e:
        logger.error(f"API 호출 중 오류 발생 - 페이지 {page}: {str(e)}")
        return None, page

def parse_jockey_data(api_response):
    """
    API 응답을 파싱하여 테이블 구조에 맞는 데이터로 변환
    """
    jockeys_data = []

    try:
        if 'response' in api_response and 'body' in api_response['response']:
            body = api_response['response']['body']
            items = body.get('items', []) if body else []

            if items:
                items = items.get('item', [])

                if isinstance(items, dict):
                    items = [items]

                for item in items:
                    jockey_data = {
                        'jk_no': safe_str(item.get('jkNo')),           # TEXT 타입으로 변환
                        'name': safe_str(item.get('jkName')),          # 기수명
                        'total_races': safe_int(item.get('rcCntT', 0)),    # 통산출주횟수
                        'total_wins': safe_int(item.get('ord1CntT', 0)),   # 통산1위횟수
                    }
                    jockeys_data.append(jockey_data)
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")

    return jockeys_data

def save_to_supabase_batch(jockeys: list) -> int:
    if not jockeys:
        return 0
    schema_keys = {"jk_no", "name", "total_races", "total_wins"}
    rows = [{k: v for k, v in j.items() if k in schema_keys} for j in jockeys if j and j.get("jk_no")]
    upsert("jockeys", rows, "jk_no")
    return len(rows)

def fetch_page_data(page):
    """
    단일 페이지 기수 데이터를 가져오는 헬퍼 함수
    """
    api_data, page_num = fetch_jockey_data(page=page)
    if api_data:
        jockeys_data = parse_jockey_data(api_data)
        return jockeys_data, page_num
    return [], page_num

def fetch_pages_sequential(start_page=1, max_pages=10, options={}):
    """
    순차적으로 기수 데이터를 가져오기
    """
    all_jockeys_data = []
    empty_pages = 0

    for page in range(start_page, start_page + max_pages):
        try:
            api_data, _ = fetch_jockey_data(page=page, options=options)

            if api_data:
                jockeys_data = parse_jockey_data(api_data)
                if jockeys_data:
                    all_jockeys_data.extend(jockeys_data)
                    empty_pages = 0
                else:
                    empty_pages += 1
                    logger.info(f"페이지 {page}: 데이터 없음")
            else:
                empty_pages += 1

            # 연속으로 빈 페이지가 3개 이상이면 중단 (기수 데이터는 말 데이터보다 적음)
            if empty_pages >= 3:
                logger.info("연속으로 빈 페이지가 많아 수집을 중단합니다.")
                break

            time.sleep(0.1)

        except Exception as e:
            logger.error(f"페이지 {page} 처리 중 오류: {str(e)}")
            continue

    return all_jockeys_data

def fetch_pages_parallel(start_page=1, max_pages=10, max_workers=2):
    """
    병렬로 기수 데이터를 가져오기
    """
    all_jockeys_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        page_range = range(start_page, start_page + max_pages)

        future_to_page = {executor.submit(fetch_page_data, page): page for page in page_range}

        empty_pages = 0
        for future in as_completed(future_to_page):
            try:
                jockeys_data, page_num = future.result()
                if jockeys_data:
                    all_jockeys_data.extend(jockeys_data)
                    logger.info(f"페이지 {page_num}: {len(jockeys_data)}명 수집")
                    empty_pages = 0
                else:
                    empty_pages += 1
                    logger.info(f"페이지 {page_num}: 데이터 없음")

                if empty_pages >= 5:
                    logger.info("연속으로 빈 페이지가 많아 수집을 중단합니다.")
                    break

            except Exception as e:
                logger.error(f"페이지 처리 중 오류: {str(e)}")

    return all_jockeys_data

def collecting_jockey_supabase(method='sequential', start_page=1, max_pages=10):
    """
    Supabase용 기수 정보 수집 메인 함수 (upsert 버전 - 업데이트/삽입)
    :param method: 'sequential' 또는 'parallel'
    :param start_page: 시작 페이지
    :param max_pages: 최대 페이지 수
    """
    logger.info("기수 정보 수집을 시작합니다 (upsert 버전 - 업데이트/삽입)...")
    logger.info(f"수집 방법: {method}, 시작 페이지: {start_page}, 최대 페이지: {max_pages}")

    start_time = time.time()

    # 데이터 수집 방법 선택
    if method == 'parallel':
        jockeys_data = fetch_pages_parallel(start_page=start_page, max_pages=max_pages, max_workers=2)
    else:
        jockeys_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)

    if not jockeys_data:
        logger.warning("수집된 기수 데이터가 없습니다.")
        return 0

    logger.info(f"총 {len(jockeys_data)}명의 기수 정보를 수집했습니다.")

    # 수집된 데이터 내에서 중복 제거
    jockeys_data = remove_duplicates_from_collected_jockeys(jockeys_data)

    saved = save_to_supabase_batch(jockeys_data)
    end_time = time.time()
    logger.info(f"기수 정보 수집 완료! 소요시간: {end_time - start_time:.2f}초, 저장: {saved}명")
    return saved

def fetch_single_jockey_data(jockey_no):
    """
    특정 기수 번호로 기수 정보 API 호출
    """
    return fetch_jockey_data(page=1, per_page=1, options={"jk_no": jockey_no})

def fetch_all_jockeys(start_page: int = 1, max_pages: int = 50) -> int:
    jockeys = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)
    if not jockeys:
        return 0
    jockeys = remove_duplicates_from_collected_jockeys(jockeys)
    return save_to_supabase_batch(jockeys)

# 실행 옵션들
if __name__ == "__main__":
    # 기수 정보 수집 (upsert 방식)
    collecting_jockey_supabase(method='sequential', start_page=1, max_pages=50)
