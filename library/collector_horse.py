# -*- coding: utf-8 -*-
import requests
import pandas as pd
from datetime import datetime, date
from config import API_KEY, SUPABASE_URL, SUPABASE_KEY  # config 파일 수정 필요
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from supabase import create_client, Client

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def get_existing_horse_ids():
    """
    DB에 이미 존재하는 말 ID들을 모두 가져오는 함수
    """
    try:
        logger.info("기존 말 ID 목록을 가져오는 중...")
        
        # 모든 기존 horse_id를 가져오기 (페이징 처리)
        existing_ids = set()
        page_size = 1000
        offset = 0
        
        while True:
            result = supabase.table('horses').select('horse_id').range(offset, offset + page_size - 1).execute()
            
            if not result.data:
                break
                
            for item in result.data:
                existing_ids.add(item['horse_id'])
            
            if len(result.data) < page_size:
                break
                
            offset += page_size
            
        logger.info(f"기존 DB에서 {len(existing_ids)}개의 말 ID를 확인했습니다.")
        return existing_ids
        
    except Exception as e:
        logger.error(f"기존 데이터 확인 실패: {str(e)}")
        return set()

def filter_new_horses(horses_data, existing_ids):
    """
    새로운 말 데이터만 필터링하는 함수
    """
    new_horses = []
    duplicate_count = 0
    
    for horse in horses_data:
        horse_id = horse.get('horse_id')
        if horse_id not in existing_ids:
            new_horses.append(horse)
        else:
            duplicate_count += 1
    
    logger.info(f"중복 제거 결과: 전체 {len(horses_data)}마리 중 새로운 말 {len(new_horses)}마리, 중복 {duplicate_count}마리")
    return new_horses

def fetch_horse_data(page=1, per_page=1000):
    '''
    경주마 정보 API에서 데이터 가져오는 함수 (최적화 버전)
    '''
    params = {
        "serviceKey": API_KEY,
        "page": page,
        "numOfRows": per_page,
        "_type": 'json'
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
            return fetch_horse_data(page, per_page)
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
                
                for item in items:
                    horse_data = {
                        'horse_id': str(item.get('hrNo', 0)),  # TEXT 타입으로 변환
                        'name': item.get('hrName', ''),
                        'age': calculate_age(item.get('birthday', 0)),
                        'gender': item.get('sex', ''),
                        'breed': item.get('name', 'Thoroughbred'),
                        'rank': item.get('rank', '')
                    }
                    horses_data.append(horse_data)
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")
        
    return horses_data

def save_to_supabase_batch(horses_data, batch_size=500):
    """
    배치 단위로 Supabase DB에 저장 (중복 방지 버전)
    """
    try:
        if not horses_data:
            logger.info("저장할 새로운 데이터가 없습니다.")
            return True
            
        total_processed = 0
        successful_batches = 0
        
        logger.info(f"새로운 {len(horses_data)}마리 저장 시작...")
        
        # 배치 처리로 성능 개선
        for i in range(0, len(horses_data), batch_size):
            batch = horses_data[i:i + batch_size]
            batch_num = i//batch_size + 1
            
            try:
                # Supabase에 데이터 삽입 (insert 사용 - 새로운 데이터만 삽입)
                result = supabase.table('horses').insert(batch).execute()
                total_processed += len(batch)
                successful_batches += 1
                logger.info(f"배치 {batch_num} 저장 완료: {len(batch)}마리 (누적: {total_processed}마리)")
                
                # API 호출 제한 방지를 위한 딜레이
                time.sleep(0.2)
                
            except Exception as batch_error:
                logger.error(f"배치 {batch_num} 저장 실패: {str(batch_error)}")
                
                # 개별 레코드로 재시도
                logger.info(f"배치 {batch_num} 개별 저장으로 재시도...")
                saved_individually = save_individually(batch)
                total_processed += saved_individually
        
        logger.info(f"=== 저장 완료 ===")
        logger.info(f"총 처리된 레코드: {total_processed}마리")
        logger.info(f"성공한 배치: {successful_batches}개")
        
        return True
        
    except Exception as e:
        logger.error(f"Supabase 저장 중 오류 발생: {e}")
        return False

def save_individually(batch_data):
    """
    개별 레코드로 저장 (배치 실패 시 백업 방법)
    """
    saved_count = 0
    logger.info(f"개별 저장 시작: {len(batch_data)}마리")
    
    for horse in batch_data:
        try:
            # 개별 저장 시에도 insert 사용 (중복이 이미 필터링됨)
            result = supabase.table('horses').insert(horse).execute()
            saved_count += 1
            
            if saved_count % 50 == 0:  # 50마리마다 로그 출력
                logger.info(f"개별 저장 진행: {saved_count}/{len(batch_data)}")
                
            time.sleep(0.05)  # 개별 저장 시 더 짧은 딜레이
        except Exception as e:
            logger.error(f"개별 저장 실패 - {horse.get('horse_id', 'Unknown')}: {str(e)}")
    
    logger.info(f"개별 저장 완료: {saved_count}마리")
    return saved_count

def check_existing_data():
    """
    기존 데이터 확인 함수 (개선된 버전)
    """
    try:
        # 전체 레코드 수 확인
        result = supabase.table('horses').select('horse_id', count='exact').execute()
        count = result.count if hasattr(result, 'count') else len(result.data)
        
        logger.info(f"현재 DB에 저장된 말 정보: {count}마리")
        
        # 최근 데이터 몇 개 확인
        recent_data = supabase.table('horses').select('horse_id, name, inserted_at').order('inserted_at', desc=True).limit(5).execute()
        
        if recent_data.data:
            logger.info("최근 저장된 데이터:")
            for horse in recent_data.data:
                logger.info(f"  - {horse.get('horse_id')}: {horse.get('name')} ({horse.get('inserted_at')})")
        
        return count
    except Exception as e:
        logger.error(f"기존 데이터 확인 실패: {str(e)}")
        return 0

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

def fetch_pages_sequential(start_page=1, max_pages=50):
    """
    순차적으로 데이터를 가져오기 (안전한 버전)
    """
    all_horses_data = []
    empty_pages = 0
    
    for page in range(start_page, start_page + max_pages):
        try:
            api_data, _ = fetch_horse_data(page=page)
            
            if api_data:
                horses_data = parse_horse_data(api_data)
                if horses_data:
                    all_horses_data.extend(horses_data)
                    logger.info(f"페이지 {page}: {len(horses_data)}마리 수집 (총 {len(all_horses_data)}마리)")
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
    Supabase용 메인 실행 함수 (중복 방지 버전)
    :param method: 'sequential' 또는 'parallel'
    :param start_page: 시작 페이지
    :param max_pages: 최대 페이지 수
    """
    logger.info("경주마 정보 수집을 시작합니다 (중복 방지 버전)...")
    logger.info(f"수집 방법: {method}, 시작 페이지: {start_page}, 최대 페이지: {max_pages}")
    
    # 기존 데이터 확인
    initial_count = check_existing_data()
    
    # 기존 말 ID 목록 가져오기
    existing_horse_ids = get_existing_horse_ids()
    
    start_time = time.time()
    
    # 데이터 수집 방법 선택
    if method == 'parallel':
        horses_data = fetch_pages_parallel(start_page=start_page, max_pages=max_pages, max_workers=2)
    else:
        horses_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)
    
    if not horses_data:
        logger.warning("수집된 데이터가 없습니다.")
        return
    
    logger.info(f"총 {len(horses_data)}마리의 말 정보를 수집했습니다.")
    
    # 수집된 데이터 내에서 중복 제거
    horses_data = remove_duplicates_from_collected_data(horses_data)
    
    # 기존 DB 데이터와 비교하여 새로운 데이터만 필터링
    new_horses_data = filter_new_horses(horses_data, existing_horse_ids)
    
    if not new_horses_data:
        logger.info("새로 추가할 데이터가 없습니다. 모든 데이터가 이미 존재합니다.")
        return
    
    # Supabase DB 저장 (새로운 데이터만)
    if save_to_supabase_batch(new_horses_data, batch_size=300):
        end_time = time.time()
        logger.info(f"경주마 정보 수집 완료! 소요시간: {end_time - start_time:.2f}초")
        
        # 최종 데이터 확인
        final_count = check_existing_data()
        added_count = final_count - initial_count
        logger.info(f"데이터 추가 결과: 기존 {initial_count}마리 → 현재 {final_count}마리 (추가: {added_count}마리)")
    else:
        logger.error("데이터 저장 실패")

# 실행 옵션들
if __name__ == "__main__":
    collecting_horse_supabase(method='parallel', start_page=800, max_pages=100)