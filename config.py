"""
경마 예측 시스템 설정 파일
"""

import os
from pathlib import Path


# 공공데이터 포털 API 키 (인코딩 되지 않은 키 입력)
API_KEY = "ofUTHp4Y5zAHr8I41vlXFYFcQ6k5Jc0hdiluFQVyKFEbJHZ8c64vVuZRtTNzYQhJUnMvalRmImgQsv4DhGSQHA=="


# ================================
# 기본 설정
# ================================
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# 디렉토리 생성
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ================================
# 데이터베이스 설정
# ================================
# 데이터베이스 연결 설정
DATABASE_CONFIG = {
    'timeout': 300,  # 5분
    'max_retries': 3,
    'retry_delay': 5  # 5초
}

SUPABASE_URL ='https://mxcblmhmcoslltjwvcrr.supabase.co'
SUPABASE_KEY ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im14Y2JsbWhtY29zbGx0and2Y3JyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA4NTI4MzIsImV4cCI6MjA2NjQyODgzMn0.9WuoUhsm624_QD1bS6emvtM-lB4kjU5Do_HowTgrkc4'


# ================================
# 데이터 추출 설정
# ================================
DATA_EXTRACTION = {
    'batch_size_months': 2,  # 배치 크기 (개월)
    'page_size': 500,        # 페이지 크기
    'min_horse_races': 3,    # 최소 경주 수
    'max_horses_per_race': 20,
    'min_horses_per_race': 5
}

# ================================
# 모델 설정
# ================================
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    
    # Random Forest
    'rf_n_estimators': 200,
    'rf_max_depth': 10,
    'rf_class_weight': 'balanced',
    
    # Gradient Boosting
    'gb_n_estimators': 200,
    'gb_max_depth': 6,
    'gb_learning_rate': 0.1,
    
    # Logistic Regression
    'lr_max_iter': 1000,
    'lr_class_weight': 'balanced'
}

# ================================
# 특성 설정
# ================================
FEATURE_CONFIG = {
    'categorical_features': [
        'horse_class', 'race_grade', 'track_condition', 'weather'
    ],
    'numeric_features': [
        'horse_age', 'is_male', 'race_distance', 'total_horses',
        'horse_weight', 'prev_total_races', 'prev_5_avg_rank',
        'prev_total_avg_rank', 'jockey_total_races', 'jockey_total_wins',
        'trainer_total_races', 'trainer_total_wins'
    ],
    'derived_features': [
        'jockey_win_rate', 'trainer_win_rate', 'horse_win_rate',
        'horse_top3_rate', 'experience_score', 'recent_form'
    ]
}

# ================================
# 예측 설정
# ================================
PREDICTION_CONFIG = {
    'confidence_threshold': 0.6,
    'ensemble_method': 'mean',  # 'mean', 'weighted'
    'top_n_predictions': 3
}

# ================================
# 백테스팅 설정
# ================================
BACKTEST_CONFIG = {
    'bet_amount': 1000,  # 베팅 금액
    'confidence_threshold': 0.6,
    'max_races_per_day': 50,
    'profit_multipliers': {
        1: 3.0,  # 1등: 3배
        2: 2.0,  # 2등: 2배
        3: 1.5   # 3등: 1.5배
    }
}

# ================================
# 로깅 설정
# ================================
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': str(LOG_DIR / 'app.log'),
            'formatter': 'detailed'
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': str(LOG_DIR / 'error.log'),
            'formatter': 'detailed'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'error': {
            'handlers': ['error_file'],
            'level': 'ERROR',
            'propagate': False
        }
    }
}

# ================================
# 모델 관리 설정
# ================================
MODEL_MANAGEMENT = {
    'auto_save': True,
    'version_format': '%Y%m%d_%H%M%S',
    'max_versions_to_keep': 5,
    'model_expiry_days': 30,
    'performance_threshold': {
        'min_auc': 0.65,
        'min_accuracy': 0.60
    }
}

# ================================
# API 설정 (향후 확장용)
# ================================
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'rate_limit': '100/minute'
}

# ================================
# 검증 함수
# ================================
def validate_config():
    """설정 유효성 검사"""
    required_env_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing environment variables: {missing_vars}")
    
    return True

def get_feature_columns():
    """전체 특성 컬럼 목록 반환"""
    return (FEATURE_CONFIG['numeric_features'] + 
            FEATURE_CONFIG['categorical_features'] + 
            FEATURE_CONFIG['derived_features'])

# ================================
# 환경별 설정 오버라이드
# ================================
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    # 프로덕션 환경 설정
    MODEL_CONFIG['cross_validation_folds'] = 3
    DATA_EXTRACTION['batch_size_months'] = 1
    LOGGING_CONFIG['handlers']['console']['level'] = 'WARNING'
    
elif ENVIRONMENT == 'testing':
    # 테스트 환경 설정
    DATA_EXTRACTION['page_size'] = 100
    MODEL_CONFIG['rf_n_estimators'] = 50
    MODEL_CONFIG['gb_n_estimators'] = 50