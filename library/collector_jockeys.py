# -*- coding: utf-8 -*-
import requests
import pandas as pd
from datetime import datetime, date
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from supabase import create_client, Client

from _const import API_KEY, SUPABASE_URL, SUPABASE_KEY
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def safe_int(value):
    """안전한 정수 변환"""
    try:
        return int(value) if value else 0
    except (ValueError, TypeError):
        return 0

def safe_str(value):
    """안전한 문자열 변환"""
    try:
        return str(value) if value else ''
    except:
        return ''

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
                        'jk_no': safe_str(item.get('jkNo', '')),           # TEXT 타입으로 변환
                        'name': safe_str(item.get('jkName', '')),          # 기수명
                        'total_races': safe_int(item.get('rcCntT', 0)),    # 통산출주횟수
                        'total_wins': safe_int(item.get('ord1CntT', 0)),   # 통산1위횟수
                        'total_seconds': safe_int(item.get('ord2CntT', 0)), # 통산2위횟수
                        'total_thirds': safe_int(item.get('ord3CntT', 0)),  # 통산3위횟수
                        'year_races': safe_int(item.get('rcCntY', 0)),     # 최근1년 출주횟수
                        'year_wins': safe_int(item.get('ord1CntY', 0)),    # 최근1년 1위횟수
                        'year_seconds': safe_int(item.get('ord2CntY', 0)), # 최근1년 2위횟수
                        'year_thirds': safe_int(item.get('ord3CntY', 0)),  # 최근1년 3위횟수
                    }
                    jockeys_data.append(jockey_data)
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")
        
    return jockeys_data

def save_to_supabase_batch(jockeys_data, batch_size=300):
    """
    배치 단위로 Supabase DB에 기수 정보 저장 (upsert 버전 - 업데이트/삽입)
    """
    try:
        if not jockeys_data:
            logger.info("처리할 기수 데이터가 없습니다.")
            return True
            
        total_processed = 0
        successful_batches = 0
        
        logger.info(f"{len(jockeys_data)}명의 기수 정보 처리 시작 (upsert)...")
        
        # 배치 처리로 성능 개선
        for i in range(0, len(jockeys_data), batch_size):
            batch = jockeys_data[i:i + batch_size]
            batch_num = i//batch_size + 1
            
            try:
                # Supabase에 데이터 upsert (있으면 업데이트, 없으면 삽입)
                result = supabase.table('jockeys').upsert(batch).execute()
                
                batch_count = len(batch)
                total_processed += batch_count
                successful_batches += 1
                
                logger.info(f"배치 {batch_num}: {batch_count}명 처리 완료 (누적: {total_processed}명)")
                
                # API 호출 제한 방지를 위한 딜레이
                time.sleep(0.2)
                
            except Exception as batch_error:
                logger.error(f"배치 {batch_num} 저장 실패: {str(batch_error)}")
                
                # 개별 레코드로 재시도
                logger.info(f"배치 {batch_num} 개별 저장으로 재시도...")
                saved_individually = save_jockeys_individually(batch)
                total_processed += saved_individually
        
        logger.info(f"=== 기수 정보 처리 완료 ===")
        logger.info(f"총 처리된 레코드: {total_processed}명")
        logger.info(f"성공한 배치: {successful_batches}개")
        
        return True
        
    except Exception as e:
        logger.error(f"Supabase 저장 중 오류 발생: {e}")
        return False

def save_jockeys_individually(batch_data):
    """
    개별 기수 레코드로 저장 (배치 실패 시 백업 방법)
    """
    saved_count = 0
    logger.info(f"개별 저장 시작: {len(batch_data)}명")
    
    for jockey in batch_data:
        try:
            # 개별 저장 시에도 upsert 사용 (있으면 업데이트, 없으면 삽입)
            result = supabase.table('jockeys').upsert(jockey).execute()
            saved_count += 1
            
            if saved_count % 20 == 0:  # 20명마다 로그 출력
                logger.info(f"개별 저장 진행: {saved_count}/{len(batch_data)}")
                
            time.sleep(0.05)
        except Exception as e:
            logger.error(f"개별 저장 실패 - {jockey.get('jk_no', 'Unknown')}: {str(e)}")
    
    logger.info(f"개별 저장 완료: {saved_count}명")
    return saved_count

def check_existing_jockeys():
    """
    기존 기수 데이터 확인 함수
    """
    try:
        # 전체 기수 수 확인
        result = supabase.table('jockeys').select('jk_no', count='exact').execute()
        count = result.count if hasattr(result, 'count') else len(result.data)
        
        logger.info(f"현재 DB에 저장된 기수 정보: {count}명")
        
        # 최근 데이터 몇 개 확인
        recent_data = supabase.table('jockeys').select('jk_no, name, total_wins, updated_at').order('updated_at', desc=True).limit(5).execute()
        
        if recent_data.data:
            logger.info("최근 업데이트된 기수 데이터:")
            for jockey in recent_data.data:
                logger.info(f"  - {jockey.get('jk_no')}: {jockey.get('name')} (승수: {jockey.get('total_wins')}) ({jockey.get('updated_at')})")
        
        return count
    except Exception as e:
        logger.error(f"기존 기수 데이터 확인 실패: {str(e)}")
        return 0

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
    
    # 기존 데이터 확인
    initial_count = check_existing_jockeys()
    
    start_time = time.time()
    
    # 데이터 수집 방법 선택
    if method == 'parallel':
        jockeys_data = fetch_pages_parallel(start_page=start_page, max_pages=max_pages, max_workers=2)
    else:
        jockeys_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)
    
    if not jockeys_data:
        logger.warning("수집된 기수 데이터가 없습니다.")
        return
    
    logger.info(f"총 {len(jockeys_data)}명의 기수 정보를 수집했습니다.")
    
    # 수집된 데이터 내에서 중복 제거
    jockeys_data = remove_duplicates_from_collected_jockeys(jockeys_data)
    
    # upsert 사용으로 모든 데이터 처리 (기존 필터링 로직 제거)
    if save_to_supabase_batch(jockeys_data, batch_size=200):
        end_time = time.time()
        logger.info(f"기수 정보 수집 완료! 소요시간: {end_time - start_time:.2f}초")
        
        # 최종 데이터 확인
        final_count = check_existing_jockeys()
        logger.info(f"데이터 처리 결과: 기존 {initial_count}명 → 현재 {final_count}명")
    else:
        logger.error("기수 데이터 저장 실패")

# 통계 정보 확인 함수
def get_jockey_statistics():
    """
    기수 통계 정보 확인
    """
    try:
        # 승수 상위 10명
        top_winners = supabase.table('jockeys').select('name, total_wins, total_races').order('total_wins', desc=True).limit(10).execute()
        
        if top_winners.data:
            logger.info("=== 통산 승수 상위 10명 ===")
            for i, jockey in enumerate(top_winners.data, 1):
                win_rate = (jockey['total_wins'] / max(jockey['total_races'], 1)) * 100
                logger.info(f"{i}. {jockey['name']}: {jockey['total_wins']}승/{jockey['total_races']}전 ({win_rate:.1f}%)")
        
        # 승률 상위 10명 (최소 100회 이상 출주)
        win_rate_query = supabase.table('jockeys').select('name, total_wins, total_races').gte('total_races', 100).execute()
        
        if win_rate_query.data:
            # 승률 계산하여 정렬
            jockeys_with_rate = []
            for jockey in win_rate_query.data:
                if jockey['total_races'] > 0:
                    win_rate = (jockey['total_wins'] / jockey['total_races']) * 100
                    jockeys_with_rate.append({
                        'name': jockey['name'],
                        'total_wins': jockey['total_wins'],
                        'total_races': jockey['total_races'],
                        'win_rate': win_rate
                    })
            
            # 승률 기준으로 정렬하여 상위 10명
            top_win_rates = sorted(jockeys_with_rate, key=lambda x: x['win_rate'], reverse=True)[:10]
            
            logger.info("\n=== 통산 승률 상위 10명 기수 (100회 이상 출주) ===")
            for i, jockey in enumerate(top_win_rates, 1):
                logger.info(f"{i}. {jockey['name']}: 승률 {jockey['win_rate']:.1f}% ({jockey['total_wins']}승/{jockey['total_races']}전)")
                
    except Exception as e:
        logger.error(f"통계 정보 확인 실패: {str(e)}")

def fetch_single_jockey_data(jockey_no):
    """
    특정 기수 번호로 기수 정보 API 호출
    """
    return fetch_jockey_data(page=1, per_page=1, options={"jk_no": jockey_no})

def update_single_jockey(jockey_no):
    """
    특정 기수의 정보를 최신으로 업데이트
    """
    try:
        logger.info(f"기수 {jockey_no} 정보 업데이트 중...")
        
        # API에서 최신 정보 가져오기
        api_data, _ = fetch_single_jockey_data(jockey_no)
        
        if api_data:
            jockeys_data = parse_jockey_data(api_data)
            if jockeys_data:
                # upsert로 업데이트
                result = supabase.table('jockeys').upsert(jockeys_data[0]).execute()
                logger.info(f"기수 {jockey_no} 정보 업데이트 완료")
                return True
        
        logger.warning(f"기수 {jockey_no} 정보를 찾을 수 없습니다.")
        return False
        
    except Exception as e:
        logger.error(f"기수 {jockey_no} 정보 업데이트 실패: {str(e)}")
        return False

def update_existing_jockeys():
    """
    기존 기수들의 최신 통계 정보를 업데이트하는 함수 (upsert 활용)
    """
    logger.info("기존 기수 정보 업데이트를 시작합니다...")
    
    try:
        # 기존 기수 번호들 가져오기
        existing_jockeys = supabase.table('jockeys').select('jk_no').execute()
        
        if not existing_jockeys.data:
            logger.info("업데이트할 기수 데이터가 없습니다.")
            return
        
        logger.info(f"{len(existing_jockeys.data)}명의 기수 정보 업데이트 중...")
        
        # 전체 기수 정보를 다시 수집해서 upsert
        collecting_jockey_supabase(method='sequential', start_page=1, max_pages=50)
        
    except Exception as e:
        logger.error(f"기수 정보 업데이트 실패: {str(e)}")

# 실행 옵션들
if __name__ == "__main__":
    # 기수 정보 수집 (upsert 방식)
    collecting_jockey_supabase(method='sequential', start_page=1, max_pages=50)
    
    # 통계 정보 확인
    get_jockey_statistics()
    
    # 특정 기수 정보 업데이트 예시
    # update_single_jockey("20240001")