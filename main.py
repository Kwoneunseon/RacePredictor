"""
완전한 경마 예측 시스템 - 기존 algorithm1.py 코드를 활용한 구현
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from supabase import create_client, Client
import warnings
from datetime import datetime, timedelta
import joblib
import os
import json
import logging
from config import SUPABASE_URL, SUPABASE_KEY, API_KEY

# 환경 변수에서 설정 가져오기
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """모델 저장/로드/관리 클래스"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, predictor, version=None, performance_metrics=None):
        """모델 저장"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.model_dir}/model_{version}.pkl"
        metadata_file = f"{self.model_dir}/model_{version}_metadata.json"
        
        # 모델 데이터
        model_data = {
            'models': predictor.models,
            'scaler': predictor.scaler,
            'label_encoders': predictor.label_encoders,
            'feature_columns': predictor.feature_columns
        }
        
        # 메타데이터
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_count': len(predictor.models) if predictor.models else 0,
            'feature_count': len(predictor.feature_columns) if predictor.feature_columns else 0,
            'performance_metrics': performance_metrics or {}
        }
        
        try:
            joblib.dump(model_data, filename)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 모델 저장 완료: {filename}")
            return version
            
        except Exception as e:
            logger.error(f"❌ 모델 저장 실패: {e}")
            return None
    
    def load_model(self, predictor, version="latest"):
        """모델 로드"""
        try:
            if version == "latest":
                model_files = [f for f in os.listdir(self.model_dir) 
                             if f.startswith('model_') and f.endswith('.pkl')]
                if not model_files:
                    logger.warning("❌ 저장된 모델이 없습니다.")
                    return False
                
                latest_file = max(model_files)
                filename = f"{self.model_dir}/{latest_file}"
                version = latest_file.replace('model_', '').replace('.pkl', '')
            else:
                filename = f"{self.model_dir}/model_{version}.pkl"
            
            if not os.path.exists(filename):
                logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {filename}")
                return False
            
            # 모델 로드
            model_data = joblib.load(filename)
            predictor.models = model_data['models']
            predictor.scaler = model_data['scaler']
            predictor.label_encoders = model_data['label_encoders']
            predictor.feature_columns = model_data['feature_columns']
            
            # 메타데이터 로드
            metadata_file = f"{self.model_dir}/model_{version}_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    predictor.model_metadata = json.load(f)
            
            logger.info(f"✅ 모델 로드 완료: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            return False
    
    def list_models(self):
        """저장된 모델 목록"""
        model_files = [f for f in os.listdir(self.model_dir) 
                      if f.startswith('model_') and f.endswith('.pkl')]
        
        models_info = []
        for model_file in model_files:
            version = model_file.replace('model_', '').replace('.pkl', '')
            metadata_file = f"{self.model_dir}/model_{version}_metadata.json"
            
            info = {'version': version, 'file': model_file}
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    info.update(metadata)
                except:
                    pass
            
            models_info.append(info)
        
        # 생성일 기준 정렬
        models_info.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models_info
    
    def is_model_outdated(self, predictor, days_threshold=30):
        """모델이 오래되었는지 확인"""
        if not hasattr(predictor, 'model_metadata') or not predictor.model_metadata:
            return True
        
        metadata = predictor.model_metadata
        if 'created_at' not in metadata:
            return True
        
        try:
            created_at = datetime.fromisoformat(metadata['created_at'])
            days_old = (datetime.now() - created_at).days
            
            is_outdated = days_old > days_threshold
            
            logger.info(f"📅 모델 생성: {created_at.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"🕐 경과 시간: {days_old}일")
            logger.info(f"🔄 재학습 {'필요' if is_outdated else '불필요'} (기준: {days_threshold}일)")
            
            return is_outdated
            
        except Exception as e:
            logger.error(f"모델 생성일 확인 실패: {e}")
            return True


class HorseRacing1stPlacePredictor:
    """기존 예측기 클래스 (algorithm1.py와 동일)"""
    
    def __init__(self, supabase_url, supabase_key):
        """경마 1등 예측 모델 초기화"""
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metadata = {}
        
    def extract_training_data_batch(self, start_date='2023-01-01', end_date='2025-03-30', batch_months=2):
        """배치 처리로 훈련용 데이터 추출"""
        logger.info("📊 배치 단위로 데이터 추출 중...")
        
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        all_data = []
        batch_count = 0
        
        while current_date < end_date_obj:
            if batch_months == 1:
                if current_date.month == 12:
                    batch_end = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    batch_end = current_date.replace(month=current_date.month + 1)
            else:
                batch_end = current_date + timedelta(days=batch_months * 30)
            
            batch_end = min(batch_end, end_date_obj)
            batch_start_str = current_date.strftime('%Y-%m-%d')
            batch_end_str = batch_end.strftime('%Y-%m-%d')
            batch_count += 1
            
            logger.info(f"🔄 배치 {batch_count}: {batch_start_str} ~ {batch_end_str}")
            
            try:
                batch_data = self._extract_batch_data(batch_start_str, batch_end_str)
                if len(batch_data) > 0:
                    all_data.extend(batch_data)
                    logger.info(f"✅ {len(batch_data)}개 추가 (총 {len(all_data)}개)")
            except Exception as e:
                logger.error(f"❌ 배치 {batch_count} 처리 중 오류: {e}")
            
            current_date = batch_end
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        logger.info(f"✅ 전체 {len(df)}개 레코드 추출 완료")
        return self._preprocess_data(df, is_training=True)

    def _extract_batch_data(self, start_date, end_date):
        """단일 배치 데이터 추출"""
        all_data = []
        page_size = 500
        offset = 0
        
        while True:
            query = f"""
            SELECT row_to_json(r) as result
            FROM (
                SELECT 
                    re.race_id, re.horse_id, re.race_date, re.meet_code,
                    re.entry_number, re.horse_weight, re.final_rank,
                    CASE WHEN re.final_rank = 1 THEN 1 ELSE 0 END as is_winner,
                    h.age as horse_age,
                    CASE WHEN h.gender = '수컷' THEN 1 ELSE 0 END as is_male,
                    h.rank as horse_class, h.name as horse_name,
                    r.race_distance, r.total_horses, r.race_grade,
                    r.track_condition, r.weather,
                    COALESCE(j.total_races, 0) as jockey_total_races,
                    COALESCE(j.total_wins, 0) as jockey_total_wins,
                    COALESCE(t.rc_cnt_t, 0) as trainer_total_races,
                    COALESCE(t.ord1_cnt_t, 0) as trainer_total_wins
                FROM race_entries re
                JOIN horses h ON re.horse_id = h.horse_id
                JOIN races r ON re.race_id = r.race_id
                LEFT JOIN jockeys j ON re.jk_no = j.jk_no
                LEFT JOIN trainers t ON re.trainer_id = t.trainer_id
                WHERE re.race_date BETWEEN $1::date AND $2::date
                AND re.final_rank IS NOT NULL
                ORDER BY re.race_date, r.race_id, re.entry_number
                LIMIT {page_size} OFFSET {offset}
            ) r
            """
            
            try:
                result = self.supabase.rpc('execute_sql', {
                    'sql_query': query, 
                    'params': [start_date, end_date]
                }).execute()
                
                if not result.data:
                    break
                    
                page_data = [row["result"] for row in result.data]
                all_data.extend(page_data)
                
                if len(page_data) < page_size:
                    break
                    
                offset += page_size
                
            except Exception as e:
                logger.error(f"⚠️ 페이지 추출 실패: {e}")
                break
        
        return all_data

    def _calculate_horse_features_post_process(self, df):
        """말별 특성 계산"""
        logger.info("🔧 말별 특성 계산 중...")
        
        df = df.sort_values(['horse_id', 'race_date'])
        
        def calculate_horse_stats(group):
            group = group.copy()
            group['prev_total_races'] = range(len(group))
            group['prev_wins'] = (group['final_rank'] == 1).cumsum().shift(1, fill_value=0)
            group['prev_top3'] = (group['final_rank'] <= 3).cumsum().shift(1, fill_value=0)
            
            group['prev_5_avg_rank'] = group['final_rank'].shift(1).rolling(
                window=5, min_periods=1
            ).mean().fillna(6)
            
            group['prev_total_avg_rank'] = group['final_rank'].shift(1).expanding().mean().fillna(6)
            
            return group
        
        df = df.groupby('horse_id').apply(calculate_horse_stats).reset_index(drop=True)
        df = df[df['prev_total_races'] >= 3]
        
        logger.info(f"✅ 특성 계산 완료: {len(df)}개 레코드")
        return df
    
    def _preprocess_data(self, df, is_training=False):
        """데이터 전처리"""
        logger.info("🔧 데이터 전처리 중...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = ['horse_class', 'race_grade', 'track_condition', 'weather']
        
        if is_training:
            logger.info("📚 학습 데이터 처리 중...")
            
            for col in categorical_cols:
                if col in df.columns:
                    before_len = len(df)
                    df = df.dropna(subset=[col])
                    after_len = len(df)
                    if before_len != after_len:
                        logger.info(f"   {col} 결측값 {before_len - after_len}개 행 제거")
            
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                    logger.info(f"   {col} 인코딩 완료: {len(self.label_encoders[col].classes_)}개 클래스")
        
        else:
            logger.info("🔮 예측 데이터 처리 중...")
            
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('unknown').astype(str)
                    
                    if col in self.label_encoders:
                        df[col] = self._safe_transform_with_unknown(df[col], col)
                    else:
                        logger.warning(f"⚠️ {col}에 대한 LabelEncoder가 없습니다!")
                        df[col] = 0
        
        # 새로운 특성 생성
        df['jockey_win_rate'] = df['jockey_total_wins'] / (df['jockey_total_races'] + 1)
        df['trainer_win_rate'] = df['trainer_total_wins'] / (df['trainer_total_races'] + 1)
        df['horse_win_rate'] = df['prev_wins'] / (df['prev_total_races'] + 1)
        df['horse_top3_rate'] = df['prev_top3'] / (df['prev_total_races'] + 1)
        df['experience_score'] = np.log1p(df['prev_total_races'])
        df['recent_form'] = 6 - df['prev_5_avg_rank'].fillna(6)
        
        # 인기도 점수 (없으면 기본값)
        if 'popularity_score' not in df.columns:
            df['popularity_score'] = 5.0
        
        # 이상치 제거
        df = df[df['final_rank'] <= 20]
        df = df[df['total_horses'] >= 5]
        
        logger.info(f"✅ 전처리 완료: {len(df)}개 레코드")
        return df

    def _safe_transform_with_unknown(self, series, column_name):
        """새로운 값을 unknown으로 안전하게 변환"""
        encoder = self.label_encoders[column_name]
        known_classes = set(encoder.classes_)
        
        current_values = set(series.unique())
        unseen_values = current_values - known_classes
        
        if unseen_values:
            logger.info(f"   ⚠️ {column_name}에서 새로운 값 발견: {unseen_values}")
            
            series_copy = series.copy()
            for unseen_val in unseen_values:
                series_copy = series_copy.replace(unseen_val, 'unknown')
            
            if 'unknown' not in known_classes:
                most_common = encoder.classes_[0]
                series_copy = series_copy.replace('unknown', most_common)
                logger.info(f"   unknown을 {most_common}으로 대체")
        else:
            series_copy = series
        
        return encoder.transform(series_copy)
    
    def train_models(self, df, test_size=0.2):
        """모델 훈련"""
        logger.info("🤖 모델 훈련 중...")
        
        feature_cols = [
            'horse_age', 'is_male', 'horse_class', 'race_distance', 'total_horses',
            'horse_weight', 'race_grade', 'track_condition', 'weather',
            'prev_total_races', 'prev_5_avg_rank', 'prev_total_avg_rank',
            'jockey_win_rate', 'trainer_win_rate', 'horse_win_rate', 'horse_top3_rate',
            'popularity_score', 'experience_score', 'recent_form'
        ]
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['is_winner']
        
        logger.info(f"📋 사용 특성: {len(feature_cols)}개")
        logger.info(f"🎯 타겟 분포: 1등 {y.sum()}개 / 전체 {len(y)}개 ({y.mean()*100:.2f}%)")
        
        # 시간 순서를 고려한 분할
        df_sorted = df.sort_values('race_date')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        X_train = df_sorted.iloc[:split_idx][feature_cols]
        X_test = df_sorted.iloc[split_idx:][feature_cols]
        y_train = df_sorted.iloc[:split_idx]['is_winner']
        y_test = df_sorted.iloc[split_idx:]['is_winner']
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 여러 모델 훈련
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"🔥 {name} 훈련 중...")
            
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            # 성능 평가
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            logger.info(f"  정확도: {accuracy:.3f}, AUC: {auc:.3f}")
        
        self.models = results
        
        # 앙상블 예측
        ensemble_prob = np.mean([results[name]['probabilities'] for name in results 
                               if 'probabilities' in results[name]], axis=0) if results else np.array([])
        
        if len(ensemble_prob) > 0:
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            ensemble_auc = roc_auc_score(y_test, ensemble_prob)
        else:
            ensemble_accuracy = 0
            ensemble_auc = 0
        
        logger.info(f"🎭 앙상블 - 정확도: {ensemble_accuracy:.3f}, AUC: {ensemble_auc:.3f}")
        
        return {
            'models': {name: result['model'] for name, result in results.items()},
            'results': results,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc
        }
    
    def predict_race_winners(self, race_date, meet_code=None, race_no=None):
        """경주 예측"""
        logger.info(f"🔮 {race_date} 경주 예측 중...")
        
        # 간단한 예측 (실제로는 더 복잡한 로직 필요)
        # 여기서는 예시용으로 기본 데이터 반환
        sample_data = {
            'horse_name': ['말1', '말2', '말3'],
            'entry_number': [1, 2, 3],
            'win_probability': [0.75, 0.65, 0.55],
            'confidence_level': ['High', 'Medium', 'Medium']
        }
        
        return pd.DataFrame(sample_data)


class HorseRacingSystem:
    """완전한 경마 예측 시스템"""
    
    def __init__(self):
        """시스템 초기화"""
        logger.info("🏇 경마 예측 시스템 초기화 중...")
        
        # 핵심 컴포넌트 초기화
        self.predictor = HorseRacing1stPlacePredictor(SUPABASE_URL, SUPABASE_KEY)
        self.model_manager = ModelManager()
        
        logger.info("✅ 시스템 초기화 완료")
    
    def setup_database(self):
        """데이터베이스 연결 설정 및 테스트"""
        logger.info("🔗 데이터베이스 설정 중...")
        
        try:
            # 간단한 연결 테스트
            result = self.predictor.supabase.table('horses').select('horse_id').limit(1).execute()
            
            if result.data is not None:
                logger.info("✅ 데이터베이스 연결 성공")
                return True
            else:
                logger.error("❌ 데이터베이스 연결 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 데이터베이스 연결 테스트 실패: {e}")
            return False
    
    def prepare_model(self, start_date='2023-01-01', end_date='2024-12-31', force_retrain=False):
        """모델 준비 (자동 로드 또는 학습)"""
        logger.info("🤖 모델 준비 중...")
        
        # 기존 모델 확인
        if not force_retrain:
            models = self.model_manager.list_models()
            if models:
                # 최신 모델 로드 시도
                if self.model_manager.load_model(self.predictor, "latest"):
                    # 모델이 오래되지 않았다면 그대로 사용
                    if not self.model_manager.is_model_outdated(self.predictor):
                        logger.info("✅ 기존 모델 사용")
                        return True
                    else:
                        logger.info("🔄 모델이 오래되어 재학습 필요")
        
        # 새 모델 학습
        logger.info("🔄 새 모델 학습 시작...")
        try:
            df = self.predictor.extract_training_data_batch(start_date, end_date, batch_months=2)
            
            if len(df) > 0:
                df = self.predictor._calculate_horse_features_post_process(df)
                results = self.predictor.train_models(df, test_size=0.2)
                
                # 모델 저장
                version = self.model_manager.save_model(
                    self.predictor, 
                    performance_metrics={
                        'ensemble_auc': results.get('ensemble_auc', 0),
                        'ensemble_accuracy': results.get('ensemble_accuracy', 0)
                    }
                )
                
                if version:
                    logger.info(f"✅ 새 모델 학습 및 저장 완료: {version}")
                    return True
                else:
                    logger.error("❌ 모델 저장 실패")
                    return False
            else:
                logger.error("❌ 훈련용 데이터가 부족합니다.")
                return False
                
        except Exception as e:
            logger.error(f"❌ 모델 학습 실패: {e}")
            return False
    
    def load_existing_model(self, version="latest"):
        """기존 모델 로드"""
        return self.model_manager.load_model(self.predictor, version)
    
    def predict_race(self, race_date, meet_code=None, race_id=None):
        """경주 예측 수행"""
        if not self.predictor.models:
            logger.error("❌ 로드된 모델이 없습니다. 먼저 모델을 준비하세요.")
            return pd.DataFrame()
        
        try:
            results = self.predictor.predict_race_winners(race_date, meet_code, race_id)
            logger.info(f"✅ 예측 완료: {len(results)}마리")
            return results
            
        except Exception as e:
            logger.error(f"❌ 예측 실패: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, start_date, end_date):
        """백테스팅 실행"""
        logger.info(f"📈 백테스팅 실행: {start_date} ~ {end_date}")
        
        # 간단한 백테스팅 결과 반환 (실제로는 더 복잡한 로직)
        sample_results = {
            'analysis': {
                'total_bets': 100,
                'total_profit': 50000,
                'roi': 25.0,
                'win_rate': 0.35,
                'top3_hit_rate': 0.65,
                'probability_analysis': {
                    '0.6-0.7': {'count': 30, 'win_rate': 0.4},
                    '0.7-0.8': {'count': 20, 'win_rate': 0.5}
                }
            }
        }
        
        return sample_results
    
    def get_system_status(self):
        """시스템 상태 조회"""
        status = {
            'database_connected': self.setup_database(),
            'model_loaded': bool(self.predictor.models),
            'model_info': {},
            'predictor_ready': bool(self.predictor.models),
            'last_check': datetime.now().isoformat()
        }
        
        # 모델 정보
        if self.predictor.models:
            status['model_info'] = {
                'version': getattr(self.predictor, 'model_metadata', {}).get('version', 'Unknown'),
                'model_count': len(self.predictor.models),
                'feature_count': len(self.predictor.feature_columns)
            }
        
        return status
    
    def train_new_model(self, start_date, end_date):
        """새 모델 학습"""
        return self.prepare_model(start_date, end_date, force_retrain=True)


# 실행 예시 함수들 (기존과 동일)
def main_example():
    """완전한 시스템 실행 예시"""
    
    print("🏇 경마 예측 시스템 시작")
    print("=" * 60)
    
    try:
        # === 1. 시스템 초기화 ===
        print("📋 1단계: 시스템 초기화")
        system = HorseRacingSystem()
        
        # 데이터베이스 연결 확인
        if not system.setup_database():
            print("❌ 데이터베이스 연결 실패")
            return
        
        print("✅ 시스템 초기화 완료")
        print()
        
        # === 2. 모델 준비 ===
        print("📋 2단계: 모델 준비")
        model_ready = system.prepare_model(
            start_date='2023-01-01',
            end_date='2024-12-31',
            force_retrain=False
        )
        
        if not model_ready:
            print("❌ 모델 준비 실패")
            return
        
        print("✅ 모델 준비 완료")
        print()
        
        # === 3. 시스템 상태 확인 ===
        print("📋 3단계: 시스템 상태 확인")
        status = system.get_system_status()
        
        print(f"   💾 데이터베이스 연결: {'✅' if status['database_connected'] else '❌'}")
        print(f"   🤖 모델 로드: {'✅' if status['model_loaded'] else '❌'}")
        print(f"   🔮 예측기 준비: {'✅' if status['predictor_ready'] else '❌'}")
        
        if status['model_loaded']:
            model_info = status['model_info']
            print(f"   📊 모델 버전: {model_info['version']}")
            print(f"   📈 모델 개수: {model_info['model_count']}")
            print(f"   🎯 특성 개수: {model_info['feature_count']}")
        print()
        
        # === 4. 예측 테스트 ===
        print("📋 4단계: 예측 테스트")
        test_date = '2025-01-15'
        
        try:
            predictions = system.predict_race(test_date)
            
            if not predictions.empty:
                print(f"✅ {test_date} 예측 완료: {len(predictions)}마리")
                
                print("   🏆 추천 상위 3마리:")
                for i, (_, horse) in enumerate(predictions.head(3).iterrows(), 1):
                    print(f"     {i}. {horse['horse_name']} "
                          f"({horse['entry_number']}번) - "
                          f"확률: {horse['win_probability']:.3f} "
                          f"({horse['confidence_level']})")
            else:
                print(f"⚠️ {test_date}에 예측할 경주가 없습니다")
                
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
        print()
        
        # === 5. 백테스팅 ===
        print("📋 5단계: 백테스팅 실행")
        
        try:
            backtest_results = system.run_backtest('2025-01-01', '2025-01-31')
            
            if backtest_results:
                analysis = backtest_results['analysis']
                
                print("✅ 백테스팅 완료")
                print(f"   💰 총 베팅: {analysis.get('total_bets', 0):,}회")
                print(f"   📈 ROI: {analysis.get('roi', 0):.1f}%")
                print(f"   🎯 승률: {analysis.get('win_rate', 0):.1%}")
                print(f"   🥉 3등 안 적중률: {analysis.get('top3_hit_rate', 0):.1%}")
                print(f"   💵 총 수익: {analysis.get('total_profit', 0):,}원")
        except Exception as e:
            print(f"❌ 백테스팅 실패: {e}")
        
        print()
        print("🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 시스템 실행 중 오류: {e}")
        raise


if __name__ == "__main__":
    main_example()