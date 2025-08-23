from datetime import datetime

def parse_date(date_str):
    """날짜 문자열을 변환 (YYYYMMDD -> YYYY-MM-DD 문자열)"""
    try:
        date_obj = datetime.strptime(str(date_str), "%Y%m%d").date()
        return date_obj.isoformat()
    except:
        return None

def safe_int(value, default=0):
    """안전한 정수 변환"""
    try:
        return int(value) if value and str(value).strip() != '' else default
    except:
        return default

def safe_float(value, default=0.0):
    """안전한 실수 변환"""
    try:
        return float(value) if value and str(value).strip() != '' else default
    except:
        return None

def safe_str(value, default=''):
    """안전한 문자열 변환"""
    try:
        return str(value).strip() if value else default
    except:
        return default
    


