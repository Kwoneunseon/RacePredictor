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

FEATURE_COLUMNS = [
    "horse_age", "is_male", "horse_class_num", "race_distance", "total_horses",
    "horse_weight", "race_grade_enc", "track_condition_enc", "weather_enc",
    "prev_total_races", "prev_5_avg_rank", "prev_total_avg_rank",
    "jockey_win_rate", "trainer_win_rate", "horse_win_rate",
    "horse_top3_rate", "experience_score", "recent_form",
    "win_odds", "implied_prob",
    "race_implied_prob_rank", "implied_prob_vs_mean",
    "horse_class_vs_mean", "win_rate_vs_mean",
    "jockey_recent_win_rate", "trainer_recent_win_rate",
    "training_grade_enc",
]
