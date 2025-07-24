# -*- coding: utf-8 -*-
import requests
from datetime import datetime, date
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from supabase import create_client, Client

from _const import API_KEY, SUPABASE_URL, SUPABASE_KEY  # config 파일 수정 필요

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
                        'trainer_name': item.get('trName', ''),            # 조교사명
                        'part': item.get('part', ''),                 # 소속조
                        # 생년월일
                        'age': int(item.get('age', 0)) if item.get('age') and str(item.get('age')).isdigit() else 0,  # 나이
                        'debut_date': item.get('stDate', ''),            # 데뷔일자
                        
                        # 통산 기록
                        'rc_cnt_t': int(item.get('rcCntT', 0)) if item.get('rcCntT') else 0,     # 통산출주횟수
                        'ord1_cnt_t': int(item.get('ord1CntT', 0)) if item.get('ord1CntT') else 0, # 통산1위횟수
                        'ord2_cnt_t': int(item.get('ord2CntT', 0)) if item.get('ord2CntT') else 0, # 통산2위횟수
                        'ord3_cnt_t': int(item.get('ord3CntT', 0)) if item.get('ord3CntT') else 0, # 통산3위횟수
                        'win_rate_t': float(item.get('winRateT', 0)) if item.get('winRateT') else 0.0,   # 통산승률
                        'qnl_rate_t': float(item.get('qnlRateT', 0)) if item.get('qnlRateT') else 0.0,   # 통산복승률
                        'plc_rate_t': float(item.get('plcRateT', 0)) if item.get('plcRateT') else 0.0,   # 통산연승률
                        
                        # 최근 1년 기록
                        'rc_cnt_y': int(item.get('rcCntY', 0)) if item.get('rcCntY') else 0,     # 최근1년출주횟수
                        'ord1_cnt_y': int(item.get('ord1CntY', 0)) if item.get('ord1CntY') else 0, # 최근1년1위횟수
                        'ord2_cnt_y': int(item.get('ord2CntY', 0)) if item.get('ord2CntY') else 0, # 최근1년2위횟수
                        'ord3_cnt_y': int(item.get('ord3CntY', 0)) if item.get('ord3CntY') else 0, # 최근1년3위횟수
                        'win_rate_y': float(item.get('winRateY', 0)) if item.get('winRateY') else 0.0,   # 최근1년승률
                        'qnl_rate_y': float(item.get('qnlRateY', 0)) if item.get('qnlRateY') else 0.0,   # 최근1년복승률
                        'plc_rate_y': float(item.get('plcRateY', 0)) if item.get('plcRateY') else 0.0,   # 최근1년연승률
                    }
                    
                    # 데이터 검증 (필수 필드)
                    if not trainer_data['trainer_id'] or not trainer_data['trainer_name']:
                        print(f"필수 정보 누락: {trainer_data}")
                        continue
                    
                    #print(f"조교사 정보 처리 완료: {trainer_data['tr_name']} ({trainer_data['tr_no']})")
                    trainers_data.append(trainer_data)
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {str(e)}")
        
    return trainers_data

def save_to_supabase_batch(trainers_data, batch_size=500):
    """
    배치 단위로 Supabase DB에 저장 (upsert 버전 - 업데이트/삽입)
    """
    try:
        if not trainers_data:
            logger.info("처리할 조교사데이터가 없습니다.")
            return True
            
        total_processed = 0
        successful_batches = 0
        
        logger.info(f"{len(trainers_data)}마리 조교사정보 처리 시작 (upsert)...")
        
        # 배치 처리로 성능 개선
        for i in range(0, len(trainers_data), batch_size):
            batch = trainers_data[i:i + batch_size]
            batch_num = i//batch_size + 1
            
            try:
                # Supabase에 데이터 upsert (있으면 업데이트, 없으면 삽입)
                result = supabase.table('trainers').upsert(batch).execute()
                total_processed += len(batch)
                successful_batches += 1
                logger.info(f"배치 {batch_num} 처리 완료: {len(batch)}마리 (누적: {total_processed}마리)")
                
                # API 호출 제한 방지를 위한 딜레이
                time.sleep(0.2)
                
            except Exception as batch_error:
                logger.error(f"배치 {batch_num} 저장 실패: {str(batch_error)}")
                
                # 개별 레코드로 재시도
                logger.info(f"배치 {batch_num} 개별 저장으로 재시도...")
                saved_individually = save_individually(batch)
                total_processed += saved_individually
        
        logger.info(f"=== 조교사정보 처리 완료 ===")
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
    
    for trainer in batch_data:
        try:
            # 개별 저장 시에도 upsert 사용
            result = supabase.table('trainers').upsert(trainer).execute()
            saved_count += 1
            
            if saved_count % 50 == 0:  # 50마리마다 로그 출력
                logger.info(f"개별 저장 진행: {saved_count}/{len(batch_data)}")
                
            time.sleep(0.05)  # 개별 저장 시 더 짧은 딜레이
        except Exception as e:
            logger.error(f"개별 저장 실패 - {trainer.get('trainer_id', 'Unknown')}: {str(e)}")
    
    logger.info(f"개별 저장 완료: {saved_count}마리")
    return saved_count

def check_existing_data():
    """
    기존 데이터 확인 함수 (개선된 버전)
    """
    try:
        # 전체 레코드 수 확인
        result = supabase.table('trainers').select('trainer_id', count='exact').execute()
        count = result.count if hasattr(result, 'count') else len(result.data)
        
        logger.info(f"현재 DB에 저장된 조교사정보: {count}마리")
        
        # 최근 데이터 몇 개 확인
        recent_data = supabase.table('trainers').select('trainer_id, trainer_name, updated_at').order('updated_at', desc=True).limit(5).execute()
        
        if recent_data.data:
            logger.info("최근 업데이트된 데이터:")
            for trainer in recent_data.data:
                logger.info(f"  - {trainer.get('trainer_id')}: {trainer.get('trainer_name')} ({trainer.get('updated_at')})")
        
        return count
    except Exception as e:
        logger.error(f"기존 데이터 확인 실패: {str(e)}")
        return 0

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
    logger.info("경주마 정보 수집을 시작합니다 (upsert 버전 - 업데이트/삽입)...")
    logger.info(f"수집 방법: {method}, 시작 페이지: {start_page}, 최대 페이지: {max_pages}")
    
    # 기존 데이터 확인
    initial_count = check_existing_data()
    
    start_time = time.time()
    
    # 데이터 수집 방법 선택
    if method == 'parallel':
        trainers_data = fetch_pages_parallel(start_page=start_page, max_pages=max_pages, max_workers=2)
    else:
        trainers_data = fetch_pages_sequential(start_page=start_page, max_pages=max_pages)
    
    if not trainers_data:
        logger.warning("수집된 데이터가 없습니다.")
        return
    
    logger.info(f"총 {len(trainers_data)}마리의 조교사정보를 수집했습니다.")
    
    # 수집된 데이터 내에서 중복 제거
    trainers_data = remove_duplicates_from_collected_data(trainers_data)
    
    # upsert 사용으로 모든 데이터 처리 (기존 필터링 로직 제거)
    if save_to_supabase_batch(trainers_data, batch_size=300):
        end_time = time.time()
        
        # 최종 데이터 확인
        #final_count = check_existing_data()
    else:
        logger.error("데이터 저장 실패")

def fetch_single_trainer_data(trainer_id):
    """
    특정 조교사ID로 조교사정보 API 호출 (fetch_trainer_data 래퍼)
    """
    return fetch_trainer_data(page=1, per_page=1, options={"tr_no": trainer_id})

def update_single_trainer(trainer_id):
    """
    특정 말의 정보를 최신으로 업데이트
    """
    try:
        logger.info(f"조교사{trainer_id} 정보 업데이트 중...")
        
        # API에서 최신 정보 가져오기
        api_data, _ = fetch_single_trainer_data(trainer_id)
        
        if api_data:
            trainers_data = parse_trainer_data(api_data)
            if trainers_data:
                # upsert로 업데이트
                result = supabase.table('trainers').upsert(trainers_data[0]).execute()
                logger.info(f"조교사{trainer_id} 정보 업데이트 완료")
                return True
        
        logger.warning(f"조교사{trainer_id} 정보를 찾을 수 없습니다.")
        return False
        
    except Exception as e:
        logger.error(f"조교사{trainer_id} 정보 업데이트 실패: {str(e)}")
        return False

# 실행 옵션들
if __name__ == "__main__":
    # 조교사정보 수집 (upsert 방식)
    collecting_trainer_supabase(method='sequential', start_page=1, max_pages=10)
    
    # 특정 조교사정보 업데이트 예시
    # update_single_trainer("20240001")