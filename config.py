from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "")
KAKAO_ACCESS_TOKEN = os.getenv("KAKAO_ACCESS_TOKEN", "")

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
LOG_DIR = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

DATABASE_CONFIG = {
    "timeout": 300,
    "max_retries": 3,
    "retry_delay": 5,
}

DATA_EXTRACTION = {
    "batch_size_months": 2,
    "page_size": 1000,
    "min_horse_races": 3,
}

MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "model_expiry_days": 30,
    "performance_threshold": {"min_auc": 0.65},
}

BANKROLL = 100_000
MAX_BET_FRACTION = 0.05
KELLY_FRACTION = 0.25
EV_THRESHOLD_SCRAPING = 0.10
EV_THRESHOLD_FALLBACK = 0.40
FILTER_DISTANCES = [1200, 1400]
FILTER_MIN_HORSES = 5
BET_PROB_THRESHOLD = 0.50  # fallback 시 배팅 최소 확률 (1위 예측마 기준)
TOP3_PROB_THRESHOLD = 0.65  # 연승 베팅 최소 확신도 (408경주 백테스트: 79.9% 적중/69.6% 커버, 2026-07-04 검증)
SOFTMAX_TEMPERATURE = 0.6  # 낮을수록 확률 분포 더 뾰족 (0.1~1.0), 1.0=기존 방식
LAMBDARANK_BLEND = 0.5    # LambdaRank 비중 (0.0=분류모델만, 1.0=LambdaRank만)
OPTUNA_N_TRIALS = 60      # 하이퍼파라미터 탐색 횟수 (많을수록 정확, 오래 걸림)

FEATURE_COLUMNS = [
    # 말 기본 정보
    "horse_age", "horse_class_num", "horse_weight",
    # 경주 정보
    "race_distance", "race_grade_enc",
    # 기수/조교사 통산 성적
    "jockey_win_rate", "trainer_win_rate",
    # 말 통산 성적
    "horse_win_rate", "horse_top3_rate",
    "prev_total_races", "prev_5_avg_rank", "prev_total_avg_rank",
    # 경주 내 상대적 특성
    "horse_class_vs_mean", "win_rate_vs_mean",
    # 최근 성적 (30일)
    "jockey_recent_win_rate", "trainer_recent_win_rate",
    # 조교 상태
    "training_grade_enc",
    # 컨디션/체중
    "days_since_last_race", "horse_weight_diff",
    # 거리·경마장·조합 특화 승률
    "horse_win_rate_at_distance", "horse_track_win_rate",
    "jockey_horse_win_rate", "jockey_track_win_rate",
    # 속도 지표 (거리/기록시간)
    "prev_speed_figure", "speed_figure_trend", "horse_speed_at_distance",
    # 추세
    "recent_rank_trend", "weight_trend",
    # 2026-07-20 최종 재검증: 최초 1~3단계 체크포인트는 두 가지 버그로 무효였음을 발견.
    # (1) 부산경남 경주(전체의 30.7%)가 race_id 트랙 라벨 불일치("부산경남" vs API299의
    #     "부경")로 신규 피처 조인이 처음부터 실패 → library/collector_race_detail.py에서
    #     meet 코드 기반 canonical 라벨로 수정.
    # (2) feature_builder의 학습용 배치 조회(500개씩 청크)가 PostgREST의 기본 1000행 응답
    #     제한에 걸려, 한 청크에 실제 일치하는 행이 1000건을 넘으면 나머지가 조용히
    #     잘려나감 → _fetch_in_batches 헬퍼로 청크 내부까지 range() 페이지네이션하도록 수정.
    # 두 버그 모두 고치고 재백필 후 3개 피처군을 한 번에 켜서 재검증한 결과도 개선 없음
    # (단승 42.9%→42.4%/-0.5pp, 연승@0.65 79.9%/69.6%→77.7%/73.5%) — 노이즈 범위 내이거나
    # 오히려 소폭 하락. 데이터 품질 문제 없이 검증된 결과이므로 이번엔 신뢰도 높게 제외.
    # "assigned_weight",
    # "s1f_avg", "c1_avg", "c2_avg", "c3_avg", "c4_avg", "g3f_avg", "g1f_avg",
    # "horse_career_win_rate",
    # "jockey_career_win_rate", "jockey_recent1y_win_rate",
    # "trainer_career_win_rate", "trainer_recent1y_win_rate",
    # "jockey_recent1y_starts", "trainer_recent1y_starts",
    # "running_style_score",
    # 출발 게이트 번호 (안쪽 레인 이점)
    "entry_number",
    # 직전 경주 배당 (경마 전날부터 이용 가능한 시장 신호)
    "prev_win_odds",
    # 최근 폼 및 경험 지표
    "recent_form",          # 최근 5경주 평균순위 기반 폼 지표 (6 - prev_5_avg_rank)
    "experience_score",     # 누적 레이스 수 log 변환
    # 마장·날씨 인코딩
    "track_condition_enc",  # 마장 상태 (양호/불량 등)
    "weather_enc",          # 날씨 인코딩
    # 성별
    "is_male",              # 수컷=1
    # 배당 상대 지표
    "implied_prob_vs_mean", # 경주 내 배당 확률 / 평균 배당 확률
    "race_implied_prob_rank",  # 경주 내 배당 확률 순위 (1=가장 낮은 배당=인기마)
]
