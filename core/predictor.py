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
from dateutil.relativedelta import relativedelta
#import xgboost as xgb  # Added import for xgb
# algorithm1.py

from model_manager import ModelManager

warnings.filterwarnings('ignore')

class HorseRacing1stPlacePredictor:
    def __init__(self, supabase_url, supabase_key):
        """
        경마 1등 예측 모델 초기화
        
        Args:
            supabase_url: Supabase 프로젝트 URL
            supabase_key: Supabase API 키
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_manager = ModelManager()
        

    def extract_training_data_batch(self, start_date='2023-01-01', end_date='2025-03-30', batch_months=2):
        """
        배치 처리로 훈련용 데이터 추출 (월 단위로 나누어 처리)
        """
        from datetime import datetime, timedelta
        import pandas as pd
        
        print("📊 배치 단위로 데이터 추출 중...")
        
        # 날짜 범위를 배치로 나누기
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        all_data = []
        batch_count = 0
        
        while current_date < end_date_obj:
            # 배치 끝 날짜 계산 (batch_months 개월씩)
            if batch_months == 1:
                # 1개월씩
                if current_date.month == 12:
                    batch_end = current_date + relativedelta(months=1)
                else:
                    batch_end = current_date + relativedelta(months=batch_months)
            else:
                # 지정된 개월 수만큼
                batch_end = current_date + timedelta(days=batch_months * 30)
            
            batch_end = min(batch_end, end_date_obj)
            
            batch_start_str = current_date.strftime('%Y-%m-%d')
            batch_end_str = batch_end.strftime('%Y-%m-%d')
            
            batch_count += 1
            print(f"\n🔄 배치 {batch_count}: {batch_start_str} ~ {batch_end_str}")
            
            try:
                batch_data = self._extract_batch_data(batch_start_str, batch_end_str)
                if len(batch_data) > 0:
                    all_data.extend(batch_data)
                    print(f"✅ {len(batch_data)}개 추가 (총 {len(all_data)}개)")
                else:
                    print("⚠️ 이 배치에는 데이터가 없습니다.")
                    
            except Exception as e:
                print(f"❌ 배치 {batch_count} 처리 중 오류: {e}")
                # 더 작은 배치로 재시도
                if batch_months > 1:
                    print("🔄 더 작은 단위로 재시도...")
                    smaller_batch = self.extract_training_data_batch(
                        batch_start_str, batch_end_str, batch_months=1
                    )
                    if len(smaller_batch) > 0:
                        all_data.extend(smaller_batch)
            
            current_date = batch_end
            
            # 메모리 관리를 위해 중간 저장점 제공
            if len(all_data) > 50000:
                print(f"🗂️ 중간 점검: {len(all_data)}개 레코드 처리됨")
        
        if not all_data:
            print("❌ 데이터를 가져올 수 없습니다. 대안 방법을 시도합니다...")
            return self._extract_data_alternative(start_date, end_date)
        
        df = pd.DataFrame(all_data)
        print(f"✅ 전체 {len(df)}개 레코드 추출 완료")
        
        return self._preprocess_data(df, is_training=True)

    def _extract_batch_data(self, start_date, end_date):
        """
        단일 배치 데이터 추출 (단순화된 쿼리)
        """
        all_data = []
        page_size = 50  # 페이지 크기 줄임
        offset = 0
        
        while True:
            # 단순화된 쿼리 - Window function 최소화
            query = f"""
            SELECT row_to_json(r) as result
            FROM (
                SELECT *
                FROM race_analysis_complete
                WHERE final_rank IS NOT NULL 
                AND race_date BETWEEN $1::date AND $2::date
                AND prev_total_races >= 3  -- 최소 3경주 이상 출전한 말만
                ORDER BY race_date, race_id, meet_code, entry_number
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
                
                # 마지막 페이지 확인
                if len(page_data) < page_size:
                    break
                    
                offset += page_size
                
            except Exception as e:
                print(f"⚠️ 페이지 {offset//page_size + 1} 추출 실패: {e}")
                break
        
        return all_data

    
    def _extract_data_alternative(self, start_date, end_date):
        """
        RPC 함수가 없을 때 대안적 데이터 추출 방법
        """
        print("🔄 대안적 방법으로 데이터 추출...")
        
        # 기본 데이터 추출
        race_entries = self.supabase.table('race_entries')\
            .select('*, horses(*), races(*), jockeys(*), trainers(*), betting_odds(*)')\
            .gte('race_date', start_date)\
            .lte('race_date', end_date)\
            .not_.is_('final_rank', 'null')\
            .execute()
        
        if not race_entries.data:
            print("❌ 데이터가 없습니다.")
            return pd.DataFrame()
        
        df = pd.DataFrame(race_entries.data)
        df = self._flatten_supabase_data(df)
        df = self._calculate_features_python(df)
        
        return self._preprocess_data(df)
    
    def _flatten_supabase_data(self, df):
        """중첩된 Supabase 데이터 평면화"""
        # horses 데이터 평면화
        if 'horses' in df.columns:
            horses_df = pd.json_normalize(df['horses'])
            horses_df.columns = ['horse_' + col for col in horses_df.columns]
            df = pd.concat([df.drop('horses', axis=1), horses_df], axis=1)
        
        # races 데이터 평면화
        if 'races' in df.columns:
            races_df = pd.json_normalize(df['races'])
            races_df.columns = ['race_' + col for col in races_df.columns]
            df = pd.concat([df.drop('races', axis=1), races_df], axis=1)
        
        # 기타 테이블들도 동일하게 처리
        for table in ['jockeys', 'trainers', 'betting_odds']:
            if table in df.columns:
                table_df = pd.json_normalize(df[table])
                table_df.columns = [table[:-1] + '_' + col for col in table_df.columns]
                df = pd.concat([df.drop(table, axis=1), table_df], axis=1)
        
        return df
    
    def _calculate_features_python(self, df):
        """Python으로 특성 계산"""
        df = df.sort_values(['horse_id', 'race_date'])
        
        # 말별 과거 성적 계산
        df['prev_total_races'] = df.groupby('horse_id').cumcount()
        df['prev_5_avg_rank'] = df.groupby('horse_id')['final_rank'].rolling(5, min_periods=1).mean().shift(1).values
        df['prev_total_avg_rank'] = df.groupby('horse_id')['final_rank'].expanding().mean().shift(1).values
        df['prev_wins'] = df.groupby('horse_id')['final_rank'].apply(lambda x: (x == 1).cumsum().shift(1)).values
        df['prev_top3'] = df.groupby('horse_id')['final_rank'].apply(lambda x: (x <= 3).cumsum().shift(1)).values
        
        # 1등 여부
        df['is_winner'] = (df['final_rank'] == 1).astype(int)
        
        # 최소 3경주 이상 출전한 말만 필터링
        df = df[df['prev_total_races'] >= 3]
        
        return df
    
    def safe_convert_to_numeric(self, df):
        """
        모든 object 컬럼을 안전하게 숫자로 변환
        """
        print("🔧 모든 문자열 컬럼을 숫자로 변환 중...")
        
        # 특별 매핑이 필요한 컬럼들
        special_mappings = {
            'budam': {
                '핸디캡': 0, 
                '마령': 1, 
                '별정a': 2, 
                '별정b': 3, 
                '별정c': 4, 
                '별정d': 5,
                'nan': 0, None: 0, '': 0, 'unknown': 0
            },
            'weight_type': {
                # weight_type은 항상 2라고 하셨으니 그대로 2로 설정
                2: 2, '2': 2, 
                'nan': 2, None: 2, '': 2, 'unknown': 2
            }
        }        
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"  🔄 {col} 처리 중...")
                
                # 이미 LabelEncoder로 처리된 컬럼들은 건너뛰기
                if col in ['horse_class', 'race_grade', 'track_condition', 'weather']:
                    continue
                
                # 특별 매핑이 있는 컬럼
                if col in special_mappings:
                    df[col] = df[col].map(special_mappings[col]).fillna(0)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)                    
              
        
        print("✅ 모든 컬럼 숫자 변환 완료")
        return df
    
    def _preprocess_data(self, df, is_training=False):
        """
        데이터 전처리
        Args:
            df: 처리할 데이터프레임
            is_training: 학습용 데이터인지 여부
        """
        print("🔧 데이터 전처리 중...")
        
        # final_rank가 16을 초과하는 값들 16으로 변경
        if 'final_rank' in df.columns:
            over_16 = df['final_rank'] > 16
            if over_16.any():
                df.loc[over_16, 'final_rank'] = 16

        if not is_training:
            columns_to_drop = ['final_rank', 'is_winner']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    print(f"  🚫 예측 시 {col} 컬럼 제거")

        # 결측치 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 카테고리 변수 처리
        categorical_cols = ['horse_class', 'race_grade', 'track_condition', 'weather']
        
        if is_training:
            # 학습 시: unknown 데이터 제외
            print("📚 학습 데이터 처리 중...")
            
            # 1. 먼저 결측값이 있는 행 제거
            for col in categorical_cols:
                if col in df.columns:
                    before_len = len(df)
                    df = df.dropna(subset=[col])  # 해당 컬럼이 결측인 행 제거
                    after_len = len(df)
                    if before_len != after_len:
                        print(f"   {col} 결측값 {before_len - after_len}개 행 제거")
            
            # 2. LabelEncoder 학습 (결측값 없는 깨끗한 데이터로)
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                    print(f"   {col} 인코딩 완료: {len(self.label_encoders[col].classes_)}개 클래스")
            # 이상치 제거
            df = df[df['final_rank'] <= 20]
            df = df[df['total_horses'] >= 5]       
        
        else:
            # 예측 시: unknown으로 처리
            print("🔮 예측 데이터 처리 중...")
            
            for col in categorical_cols:
                if col in df.columns:
                    # 1. 결측값을 'unknown'으로 처리
                    df[col] = df[col].fillna('unknown').astype(str)
                    
                    # 2. 학습된 LabelEncoder로 변환
                    if col in self.label_encoders:
                        df[col] = self._safe_transform_with_unknown(df[col], col)
                    else:
                        print(f"⚠️ {col}에 대한 LabelEncoder가 없습니다!")
                        df[col] = 0  # 기본값으로 처리
        
        # 새로운 특성 생성
        df['jockey_win_rate'] = df['jockey_total_wins'] / (df['jockey_total_races'] + 1)
        df['trainer_win_rate'] = df['trainer_total_wins'] / (df['trainer_total_races'] + 1)
        df['horse_win_rate'] = df['prev_wins'] / (df['prev_total_races'] + 1)
        df['horse_top3_rate'] = df['prev_top3'] / (df['prev_total_races'] + 1)
        df['experience_score'] = np.log1p(df['prev_total_races'])
        df['recent_form'] = 6 - df['prev_5_avg_rank'].fillna(6)
        

        print(f"✅ 전처리 완료: {len(df)}개 레코드")
        return df


    def _safe_transform_with_unknown(self, series, column_name):
        """
        예측 시 안전한 변환 (새로운 값은 unknown으로 처리)
        """
        encoder = self.label_encoders[column_name]
        known_classes = set(encoder.classes_)
        
        # 새로운 값들 찾기
        current_values = set(series.unique())
        unseen_values = current_values - known_classes
        
        if unseen_values:
            print(f"   ⚠️ {column_name}에서 새로운 값 발견: {unseen_values}")
            
            # 새로운 값들을 unknown으로 대체
            series_copy = series.copy()
            for unseen_val in unseen_values:
                series_copy = series_copy.replace(unseen_val, 'unknown')
            
            # unknown도 학습된 클래스에 없다면 (이런 경우는 없어야 하지만)
            if 'unknown' not in known_classes:
                # 가장 빈번한 클래스로 대체
                most_common = encoder.classes_[0]
                series_copy = series_copy.replace('unknown', most_common)
                print(f"   unknown을 {most_common}으로 대체")
        else:
            series_copy = series
        
        return encoder.transform(series_copy)
    
    def train_models(self, df, test_size=0.2, model_name='horse_racing_model'):
        """
        1등 예측 모델 훈련
        """
        print("🤖 모델 훈련 중...")
        
        # 특성 선택
        feature_cols = [
            'horse_age', 'is_male', 'horse_class', 'race_distance', 'total_horses',
            'horse_weight', 'race_grade', 'track_condition', 'weather',
            'prev_total_races', 'prev_5_avg_rank', 'prev_total_avg_rank',
            'jockey_win_rate', 'trainer_win_rate', 'horse_win_rate', 'horse_top3_rate',
            'popularity_score', 'experience_score', 'recent_form'
        ]
        
        # 실제로 존재하는 컬럼만 선택
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['is_winner']
        
        print(f"📋 사용 특성: {len(feature_cols)}개")
        print(f"🎯 타겟 분포: 1등 {y.sum()}개 / 전체 {len(y)}개 ({y.mean()*100:.2f}%)")
        
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
                n_estimators=500,  # 300 → 500
                max_depth=15,      # 10 → 15
                min_samples_split=5,   # 20 → 5
                min_samples_leaf=2,    # 10 → 2
                class_weight={0: 1, 1: 20},  # 10 → 20
                max_features='log2',   # sqrt → log2
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, 
                max_depth=6, 
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔥 {name} 훈련 중...")
            
            # 훈련
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
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            print(f"  정확도: {accuracy:.3f}")
            print(f"  정밀도: {precision:.3f}")
            print(f"  재현율: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
            print(f"  AUC: {auc:.3f}")
        
        self.models = results
        
        # 앙상블 예측
        ensemble_prob = np.mean([results[name]['probabilities'] for name in results], axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_prob)
        
        print(f"\n🎭 앙상블 결과:")
        print(f"  정확도: {ensemble_accuracy:.3f}")
        print(f"  AUC: {ensemble_auc:.3f}")
        
        # 특성 중요도 (RandomForest 기준)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': results['RandomForest']['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 특성 중요도 TOP 10:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        #모델 저장
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_success = self.model_manager.save_model(model_name, model_data)
        if save_success:
            print(f"✅ 모델이 저장되었습니다: {model_name}")
        else:
            print(f"❌ 모델 저장에 실패했습니다.")


        
        print(f"\n🎭 추가: 3등 안 예측 성능 평가")
        top3_evaluation = self.evaluate_top3_prediction(df, test_size)
        
        return {
            'test_data': df_sorted.iloc[split_idx:],
            'results': results,
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc,
            'feature_importance': feature_importance
        }    

    def get_loaded_model(self, model_name: str = 'horse_racing_model'):
        """
        저장된 모델 불러오기 (개선된 버전)
        """
        print("=" * 50)
        print("🔄 모델 로드 시도")
        print("=" * 50)
        
        model_data = self.model_manager.load_model_safe(model_name)
        
        if not model_data:
            print(f"❌ 모델 '{model_name}'을(를) 찾을 수 없습니다.")
            print("\n💡 해결 방법:")
            print("1. 사용 가능한 모델 목록 확인:")
            print("   predictor.model_manager.list_saved_models()")
            print("2. 새로운 모델 훈련:")
            print("   predictor.precision_boost_training(df)")
            return False
        
        # 모델 데이터 적용
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.best_threshold = model_data.get('best_threshold', 0.5)
        
        print(f"✅ 모델 '{model_name}' 로드 성공!")
        return True
    
    def predict_race_winners(self, race_date, meet_code=None, race_no=None, show=True):
        """
        특정 경주의 1등 예측 (경마 특화 특성 생성 포함)
        """
        print(f"🔮 {race_date} 경주 예측 중...")       

        # WHERE 조건 구성
        where_conditions = [f"race_date = '{race_date}'"]
        if meet_code:
            where_conditions.append(f"meet_code = '{meet_code}'")
        if race_no:
            where_conditions.append(f"race_id = '{race_no}'")  # 문자열로 처리
        
        where_clause = " AND ".join(where_conditions)
        
        # 수정된 쿼리 - 중복 제거 및 올바른 JOIN
        query = f"""
                SELECT row_to_json(r) as result
                FROM (
                    SELECT DISTINCT
                        race_id,
                        horse_id,
                        race_date,
                        meet_code,
                        entry_number,
                        horse_weight,
                        final_rank,
                        finish_time,
                        horse_race_days,
                        horse_weight_diff,
                        budam,
                        budam_weight,
                        horse_rating, 
                        is_winner,
                        
                        -- 말 정보
                        horse_age,
                        is_male,
                        horse_class,
                        horse_name,

                        -- 경주 정보
                        race_distance,
                        total_horses,
                        planned_horses,
                        race_grade,
                        track_condition,
                        weather,
                        weight_type,  

                        prev_total_races,
                        prev_5_avg_rank,
                        prev_total_avg_rank,

                        prev_3_avg_speed_mps, 
                        prev_5_avg_speed_mps,
                        speed_improvement_trend,
                        prev_3_avg_time_same_distance, 

                        prev_top3,
                        prev_top5,
                        year_top3,
                        year_top5,
                        total_races,
                        total_win_rate,
                        total_place_rate,
                        year_races,
                        year_win_rate,
                        year_place_rate, 

                        -- 기수 정보 (NULL 처리)
                        COALESCE(jockey_total_races, 0) as jockey_total_races,
                        COALESCE(jockey_total_wins, 0) as jockey_total_wins,
                        COALESCE(jockey_year_races, 0) as jockey_year_races,
                        COALESCE(jockey_year_wins, 0) as jockey_year_wins,
                        
                        -- 조교사 정보 (NULL 처리)
                        COALESCE(trainer_total_races, 0) as trainer_total_races,
                        COALESCE(trainer_total_wins, 0) as trainer_total_wins,
                        COALESCE(trainer_year_races, 0) as trainer_year_races,
                        COALESCE(trainer_year_wins, 0) as trainer_year_wins,
                        
                        avg_rank_at_distance,
                        races_at_distance
                                            
                    FROM race_analysis_complete
                    WHERE {where_clause}
                    ORDER BY race_id, entry_number
                ) r
                """
                
        try:
            result = self.supabase.rpc('execute_sql', {
                'sql_query': query,
                'params': []
            }).execute()

            if not result.data:
                print("❌ 해당 경주에 대한 데이터가 없습니다.")
                return None
                
            df = pd.DataFrame([row["result"] for row in result.data])
            df = self.safe_convert_to_numeric(df)
            
            # 중복 제거 (혹시 모를 중복 데이터)
            df = df.drop_duplicates(subset=['race_id', 'race_date', 'meet_code', 'horse_id', 'entry_number'])
            
            print(f"📊 조회된 데이터: {len(df)}개 레코드")
            print(f"📊 고유 경주 수: {df[['race_date', 'meet_code', 'race_id']].drop_duplicates().shape[0]}개")
            print(f"📊 고유 말 수: {df['horse_id'].nunique()}개")
            
            # 🔧 각 말의 과거 데이터 계산
            df = self._calculate_prediction_features(df, race_date)
            
            # 🎯 핵심 수정: 경마 특화 특성 생성 추가!
            print("🏇 예측용 경마 특화 특성 생성 중...")
            try:
                df = self.create_racing_specific_features(df)
                print("✅ 경마 특화 특성 생성 완료")
            except Exception as e:
                print(f"⚠️ 경마 특화 특성 생성 실패: {e}")
                print("기본 특성만으로 예측을 진행합니다.")
            
            # 데이터 전처리
            copy_df = df.copy()
            df = self._preprocess_data(df, is_training=False)
            
            # 모델이 학습되지 않았다면 에러
            if not self.models:
                print("❌ 모델이 학습되지 않았습니다. 먼저 모델을 학습하거나 불러와주세요.")
                return None
            
            # 필요한 특성이 있는지 확인
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                print(f"⚠️ 누락된 특성: {missing_features}")
                print("🔧 누락된 특성을 0으로 채워서 진행합니다...")
                
                # 누락된 특성을 0으로 채우기
                for feature in missing_features:
                    df[feature] = 0
                
                print("✅ 누락 특성 처리 완료")
            
            # 예측 수행
            predictions = []
            
            for name, result in self.models.items():
                model = result['model']
                
                try:
                    if name == 'LogisticRegression' or 'LR' in name:
                        X_scaled = self.scaler.transform(df[self.feature_columns])
                        prob = model.predict_proba(X_scaled)[:, 1]
                        
                    elif 'XGB' in name:
                        X_numpy = df[self.feature_columns].values
                        prob = model.predict_proba(X_numpy)[:, 1]
                    else:
                        prob = model.predict_proba(df[self.feature_columns])[:, 1]
                    
                    predictions.append(prob)
                except Exception as e:
                    print(f"⚠️ {name} 모델 예측 실패: {e}")
                    continue
            
            if not predictions:
                print("❌ 모든 모델 예측이 실패했습니다.")
                return None
            
            # 앙상블 예측
            ensemble_prob = np.mean(predictions, axis=0)
            
            # 결과 정리
            result_df = copy_df[['race_id', 'race_date', 'meet_code','horse_id', 'horse_name', 'entry_number', 
                        'horse_age', 'horse_class', 'is_male', 'final_rank','prev_total_races']].copy()
            result_df['win_probability'] = ensemble_prob
            
            # 경주별로 예측 등수 계산
            def calculate_race_rank(group):
                group = group.copy()
                group['prediction_rank'] = group['win_probability'].rank(ascending=False, method='min').astype(int)
                return group
            
            result_df = result_df.groupby(['race_id', 'meet_code']).apply(calculate_race_rank).reset_index(drop=True)
            result_df = result_df.sort_values(['meet_code','race_id','race_date', 'prediction_rank'])

            # 🎯 정밀도 중심 추천 (임계값 적용)
            threshold = getattr(self, 'best_threshold', 0.5)
            result_df['high_confidence'] = (result_df['win_probability'] > threshold).astype(int)
            result_df['recommendation'] = result_df['high_confidence'].map({
                1: '🎯 강력 추천',
                0: '⚠️ 보류'
            })

            # 🆕 경험 부족 표시 추가
            result_df['is_inexperienced'] = (result_df['prev_total_races'] <= 5).astype(int)
            result_df['experience_flag'] = result_df['is_inexperienced'].map({
                1: '🔰 신참',  # 5경주 이하
                0: ''         # 경험 충분
            })

            if show:
                # 경주별 결과 출력
                print("\n" + "="*60)
                print("🏆 예측 결과")
                print("="*60)

                unique_races = result_df[['meet_code', 'race_id']].drop_duplicates().sort_values(['meet_code','race_id'])
                
                for _, row in unique_races.iterrows():
                    race_id = row['race_id']
                    meet_code = row['meet_code']
                    race_data = result_df[(result_df['race_id'] == race_id) & (result_df['meet_code'] == meet_code)].head(3)
                    
                    print(f"\n🏁 {meet_code} 경주 {race_id}번 - TOP 5 예측")
                    print("-" * 50)
                    
                    for idx, row in race_data.iterrows():
                        gender = '수컷' if row['is_male'] == 1 else '암컷'
                        actual_rank = f" (실제: {int(row['final_rank'])}등)" if pd.notna(row['final_rank']) else ""
                        confidence_icon = "🎯" if row['high_confidence'] == 1 else "⚠️"
                        experience_info = f" {row['experience_flag']}" if row['experience_flag'] else ""
                        experience_races = f"({int(row['prev_total_races'])}경주)" if pd.notna(row['prev_total_races']) else "(경험불명)"
                
                        print(f"  {confidence_icon} {int(row['prediction_rank'])}등 | "
                            f"#{int(row['entry_number'])}번 | "
                            f"{row['horse_name']}{experience_info} | "
                            f"{int(row['horse_age'])}세 {gender} | "
                            f"등급:{row['horse_class']} | "
                            f"확률:{row['win_probability']:.3f} | "
                            f"{row['recommendation']}"
                            f"{actual_rank}")

                # 강력 추천 요약
                high_conf = result_df[result_df['high_confidence'] == 1]
                print(f"\n🎯 정밀도 중심 추천 요약 (임계값: {threshold:.3f}):")
                if len(high_conf) > 0:
                    print(f"강력 추천: {len(high_conf)}마리")
                    for _, horse in high_conf.iterrows():
                        print(f"  🏆 {horse['horse_name']} (#{horse['entry_number']}번, 확률: {horse['win_probability']:.3f})")
                else:
                    print("⚠️ 이번 경주는 확신할 만한 말이 없습니다.")

                # 🆕 신참 말 별도 경고
                inexperienced_in_top3 = result_df[
                    (result_df['prediction_rank'] <= 3) & 
                    (result_df['is_inexperienced'] == 1)
                ]
                
                if len(inexperienced_in_top3) > 0:
                    print(f"\n🔰 신참 말 주의사항:")
                    print("-" * 30)
                    for _, horse in inexperienced_in_top3.iterrows():
                        print(f"  ⚠️ {horse['horse_name']} (#{horse['entry_number']}번): "
                            f"과거 {int(horse['prev_total_races'])}경주만 출전 - 변수 가능성 높음")
                    return result_df
                    
        except Exception as e:
            print(f"❌ 예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 더 엄격한 학습 데이터 필터링
    def _filter_training_data(self, df):
        """
        학습 데이터에서 품질이 낮은 데이터 제거
        """
        initial_len = len(df)
        
        # 1. 필수 컬럼들이 모두 있는 행만 유지
        required_cols = ['horse_class', 'race_grade', 'track_condition', 'weather', 
                        'prev_total_races', 'jockey_total_races', 'trainer_total_races']
        
        for col in required_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # 2. 이상한 값들 제거
        df = df[df['prev_total_races'] >= 3]  # 최소 3경주 이상
        df = df[df['horse_age'] >= 2]         # 2세 이상
        df = df[df['horse_age'] <= 10]        # 10세 이하
        
        print(f"📊 데이터 필터링: {initial_len} → {len(df)} ({len(df)/initial_len*100:.1f}%)")
        
        return df
    
    def _calculate_prediction_features(self, df, current_date):
        """예측용 특성 계산"""
        print("📊 각 말의 과거 성적 계산 중...")
        
        # 기본값으로 초기화
        df['prev_total_races'] = 0
        df['prev_5_avg_rank'] = 6.0
        df['prev_total_avg_rank'] = 6.0
        df['prev_wins'] = 0
        df['prev_top3'] = 0
        df['avg_rank_at_distance'] = 6.0
        df['races_at_distance'] = 0
        
        processed_count = 0
        
        for horse_id in df['horse_id'].unique():
            try:
                # 과거 성적 조회 쿼리 (현재 날짜 이전)
                past_races = self.supabase.table('race_entries')\
                    .select('final_rank, race_date')\
                    .eq('horse_id', horse_id)\
                    .lt('race_date', current_date)\
                    .not_.is_('final_rank', 'null')\
                    .order('race_date', desc=True)\
                    .execute()
                
                if past_races.data and len(past_races.data) > 0:
                    ranks = [r['final_rank'] for r in past_races.data if r['final_rank'] is not None]
                    
                    if ranks:  # 유효한 순위 데이터가 있을 때만
                        # 특성 계산
                        mask = df['horse_id'] == horse_id
                        df.loc[mask, 'prev_total_races'] = len(ranks)
                        df.loc[mask, 'prev_5_avg_rank'] = np.mean(ranks[:5]) if len(ranks) >= 1 else 6.0
                        df.loc[mask, 'prev_total_avg_rank'] = np.mean(ranks) if ranks else 6.0
                        df.loc[mask, 'prev_wins'] = sum(1 for r in ranks if r == 1)
                        df.loc[mask, 'prev_top3'] = sum(1 for r in ranks if r <= 3)
                        
                        processed_count += 1
                
                # 거리별 성적 계산 (안전한 방식 - JOIN 없이)
                race_distance = df[df['horse_id'] == horse_id]['race_distance'].iloc[0] if len(df[df['horse_id'] == horse_id]) > 0 else None
                
                if race_distance:
                    try:
                        # 1단계: 해당 말의 모든 과거 경주 ID 가져오기
                        past_race_entries = self.supabase.table('race_entries')\
                            .select('race_id, race_date, meet_code, final_rank')\
                            .eq('horse_id', horse_id)\
                            .lt('race_date', current_date)\
                            .not_.is_('final_rank', 'null')\
                            .execute()
                        
                        if past_race_entries.data:
                            same_distance_ranks = []
                            
                            # 2단계: 각 경주의 거리 정보 개별 조회
                            for entry in past_race_entries.data[:10]:  # 최근 10경주만 확인 (성능 향상)
                                try:
                                    race_info = self.supabase.table('races')\
                                        .select('race_distance')\
                                        .eq('race_id', entry['race_id'])\
                                        .eq('race_date', entry['race_date'])\
                                        .eq('meet_code', entry['meet_code'])\
                                        .execute()
                                    
                                    if (race_info.data and 
                                        race_info.data[0]['race_distance'] == race_distance):
                                        same_distance_ranks.append(entry['final_rank'])
                                        
                                except Exception as e:
                                    continue  # 개별 쿼리 실패는 건너뛰기
                            
                            # 결과 적용
                            if same_distance_ranks:
                                mask = df['horse_id'] == horse_id
                                df.loc[mask, 'avg_rank_at_distance'] = np.mean(same_distance_ranks)
                                df.loc[mask, 'races_at_distance'] = len(same_distance_ranks)
                                
                    except Exception as distance_error:
                        # 거리별 성적 계산 실패 시 기본값 유지
                        print(f"⚠️ 거리별 성적 계산 실패 (말 {horse_id}): {distance_error}")
                        pass
                            
            except Exception as e:
                print(f"⚠️ 말 {horse_id} 과거 성적 계산 실패: {e}")
                continue
        
        print(f"✅ {processed_count}/{df['horse_id'].nunique()}마리 과거 성적 계산 완료")
        
        # 추가 기본 특성 생성
        df['jockey_win_rate'] = df['jockey_total_wins'] / (df['jockey_total_races'] + 1)
        df['trainer_win_rate'] = df['trainer_total_wins'] / (df['trainer_total_races'] + 1)
        df['horse_win_rate'] = df['prev_wins'] / (df['prev_total_races'] + 1)
        df['horse_top3_rate'] = df['prev_top3'] / (df['prev_total_races'] + 1)
        df['experience_score'] = np.log1p(df['prev_total_races'])
        df['recent_form'] = 6 - df['prev_5_avg_rank']
        
        # 인기도 점수 (임시로 entry_number로 대체)
        if 'popularity_score' not in df.columns:
            df['popularity_score'] = 1.0 / (df['entry_number'] + 1)
        
        return df
    
    def backtest_strategy(self, start_date, end_date, confidence_threshold=0.3):
        """
        백테스팅 전략 - 1등 예측 모델을 사용하여 3등안에 드는지 테스트
        """
        print(f"📈 백테스팅 수행(3등 안 예측): {start_date} ~ {end_date}")
        
        # 기간별 모든 경주 조회
        races = self.supabase.table('races')\
            .select('race_date, meet_code, race_id')\
            .gte('race_date', start_date)\
            .lte('race_date', end_date)\
            .execute()
        
        total_bets = 0
        total_profit = 0
        top3_hits = 0    # 3등 안 적중
        wins = 0        # 1등 적중
        budget = 100000 # 초기 에산 10만원

        detailed_results = [] # 상세결과 저장
        
        for race in races.data[:50]:  # 테스트용으로 50경주만
            try:
                predictions = self.predict_race_winners(
                    race['race_date'], 
                    race['meet_code'], 
                    race['race_id'], show=False
                )
                
                if isinstance(predictions, str) or predictions is None:
                    continue
                
                # 가장 확신하는 말에 베팅
                best_horse = predictions.iloc[3]
                
                # 상위 3마리 말 선택 (확률 높은 순)
                top_picks = predictions.head(1)  # 상위 3마리
            
                for idx, horse in top_picks.iterrows():
                    if horse['win_probability'] > confidence_threshold:
                        total_bets += 1
                        if horse['win_probability'] > 0.9:
                            bet_price = 10000
                        elif horse['win_probability'] > 0.8:
                            bet_price = 5000  
                        elif horse['win_probability'] > 0.7:
                            bet_price = 2000  
                        else:
                            bet_price = 1000

                        budget -= bet_price  # 베팅금 차감

                        
                        # 실제 결과 확인
                        actual_result = self.supabase.table('race_entries')\
                            .select('final_rank, horse_id')\
                            .eq('race_date', race['race_date'])\
                            .eq('meet_code', race['meet_code'])\
                            .eq('race_id', race['race_id'])\
                            .eq('entry_number', horse['entry_number'])\
                            .execute()
                        
                        if actual_result.data:
                            actual_rank = actual_result.data[0]['final_rank']
                            horse_id = actual_result.data[0]['horse_id']
                            
                            # 결과 기록
                            result_record = {
                                'race_date': race['race_date'],
                                'meet_code': race['meet_code'],
                                'race_id': race['race_id'],
                                'horse_id': horse_id,
                                'entry_number': horse['entry_number'],
                                'predicted_prob': horse['win_probability'],
                                'actual_rank': actual_rank,
                                'is_top3': actual_rank <= 3,
                                'is_winner': actual_rank == 1
                            }
                            detailed_results.append(result_record)
                            
                            # 3등 안에 들었는지 확인 (수정된 부분)
                            if actual_rank <= 3:
                                top3_hits += 1
                                # 3등 기준 수익 계산 (예: 1등=3배, 2등=2배, 3등=1.5배)
                                if actual_rank in [1, 2, 3]:
                                    budget += bet_price*1.6 
                                    if actual_rank == 1:
                                        wins += 1
                            
                            # 디버깅용 출력
                            status = "✅ TOP3" if actual_rank <= 3 else "❌ 실패"
                            print(f" {race['race_date']} {race['meet_code']} R{race['race_id']} {horse_id}({horse['entry_number']}번): {horse['win_probability']:.3f} → {actual_rank}등 {status}")
                            
                        
            except Exception as e:
                print(f"오류: {e}")
                continue
        
        # 결과 분석
        if total_bets > 0:
            top3_hit_rate = top3_hits / total_bets
            win_rate = wins / total_bets
            roi = (total_profit / (total_bets * 1000)) * 100
            
            print(f"\n📊 백테스팅 결과 (1등 모델로 3등 예측):")
            print(f"  총 베팅: {total_bets}회")
            print(f"  1등 적중: {wins}회 ({win_rate:.1%})")
            print(f"  3등 안 적중: {top3_hits}회 ({top3_hit_rate:.1%})")
            print(f"  남은 예산: {budget}원")
            print(f"  ROI: {roi:.1f}%")
            
            # 확률별 성과 분석
            print(f"\n📈 확률대별 성과:")
            prob_ranges = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            
            for min_prob, max_prob in prob_ranges:
                range_results = [r for r in detailed_results 
                            if min_prob <= r['predicted_prob'] < max_prob]
                if range_results:
                    range_top3 = sum(1 for r in range_results if r['is_top3'])
                    range_total = len(range_results)
                    range_hit_rate = range_top3 / range_total
                    print(f"  확률 {min_prob:.1f}~{max_prob:.1f}: {range_top3}/{range_total} ({range_hit_rate:.1%})")
            
            return {
                'total_bets': total_bets,
                'wins': wins,
                'top3_hits': top3_hits,
                'win_rate': win_rate,
                'top3_hit_rate': top3_hit_rate,
                'total_profit': total_profit,
                'roi': roi,
                'detailed_results': detailed_results
            }
        else:
            print("베팅할 경주가 없었습니다.")
            return None

    def evaluate_top3_prediction(self, df, test_size=0.2):
        """
        1등 학습 모델의 3등 예측 성능 평가
        """
        print("🎯 3등 안 예측 성능 평가 중...")
        
        # 시간 순서를 고려한 분할
        df_sorted = df.sort_values('race_date')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        test_df = df_sorted.iloc[split_idx:].copy()
        
        # 3등 안 타겟 생성
        test_df['is_top3'] = (test_df['final_rank'] <= 3).astype(int)
        
        X_test = test_df[self.feature_columns]
        y_top3 = test_df['is_top3']
        y_winner = test_df['is_winner']
        
        # 각 모델로 예측
        print("\n📊 모델별 3등 예측 성능:")
        
        for name, result in self.models.items():
            model = result['model']
            
            if name == 'LogisticRegression':
                X_test_scaled = self.scaler.transform(X_test)
                prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                prob = model.predict_proba(X_test)[:, 1]
            
            # 다양한 임계값으로 3등 예측 성능 평가
            thresholds =  [0.6, 0.7, 0.8, 0.9]
            
            print(f"\n🔥 {name} 모델:")
            for threshold in thresholds:
                pred_top3 = (prob > threshold).astype(int)
                
                # 3등 안 예측 성능
                from sklearn.metrics import classification_report
                top3_precision = precision_score(y_top3, pred_top3)
                top3_recall = recall_score(y_top3, pred_top3)
                top3_f1 = f1_score(y_top3, pred_top3)
                
                print(f"  임계값 {threshold}: 정밀도={top3_precision:.3f}, 재현율={top3_recall:.3f}, F1={top3_f1:.3f}")
        
        return test_df
    

    # 기존 코드에 추가할 통합 솔루션
    def precision_boost_training(self, df, test_size=0.2, model_name='precision_boosted_model'):
        """
        정밀도 극대화를 위한 통합 솔루션 (NaN 값 처리 개선)
        """
        print("🚀 정밀도 극대화 모델 훈련 시작!")
        df = self.safe_convert_to_numeric(df)
        
        # 1. 경마 특화 특성 생성
        print("\n1️⃣ 경마 특화 특성 생성...")
        df = self.create_racing_specific_features(df)
        
        # 2. 업데이트된 특성 목록
        feature_cols = [
            # 기본 특성
            'horse_weight', 'horse_age', 'is_male', 'horse_class', 'race_distance', 
            'finish_time', 'horse_race_days', 'horse_weight_diff', 'budam', 'budam_weight', 'horse_rating', 
            'race_distance', 'total_horses', 'planned_horses', 'race_grade', 'track_condition', 'weather', 'weight_type',
            'prev_total_races', 'prev_5_avg_rank', 'prev_total_avg_rank',
            'prev_3_avg_speed_mps', 'prev_5_avg_speed_mps', 'speed_improvement_trend', 'prev_3_avg_time_same_distance', 

            'prev_top3', 'prev_top5',
            'year_top3', 'year_top5',
            'total_races', 'total_win_rate', 'total_place_rate',        
            'year_races', 'year_win_rate', 'year_place_rate',     

            'jockey_total_races', 'jockey_total_wins', 'jockey_year_races', 'jockey_year_wins',
            'trainer_total_races', 'trainer_total_wins', 'trainer_year_races', 'trainer_year_wins',
            'avg_rank_at_distance', 'races_at_distance',

            'horse_top3_rate','experience_score', 'recent_form',
            
            # 🎯 새로운 핵심 특성들
            'championship_probability',  # 가장 중요!
            'dominance_score',
            'consistency_score', 
            'distance_fitness',
            'jockey_horse_synergy',
            'momentum',
            'championship_rank',
            'is_clear_favorite',
            'distance_performance_score', 'time_advantage_score', 
            'speed_competitiveness', 'practical_racing_score'
            'relative_win_rate'
        ]
        
        # 실제 존재하는 컬럼만 선택
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        print(f"📋 총 특성 수: {len(feature_cols)}개")
        
        # 3. 데이터 분할
        df_sorted = df.sort_values('race_date')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        X_train = df_sorted.iloc[:split_idx][feature_cols]
        X_test = df_sorted.iloc[split_idx:][feature_cols]
        y_train = df_sorted.iloc[:split_idx]['is_winner']
        y_test = df_sorted.iloc[split_idx:]['is_winner']
        
        print(f"\n📊 분할 전 데이터 상태:")
        print(f"  훈련 세트: 1등 {y_train.sum()}개 / 전체 {len(y_train)}개 ({y_train.mean()*100:.2f}%)")
        print(f"  테스트 세트: 1등 {y_test.sum()}개 / 전체 {len(y_test)}개 ({y_test.mean()*100:.2f}%)")
        
        # 🔧 4. NaN 값 완전 제거 (SMOTE 적용 전 필수!)
        print("\n2️⃣ NaN 값 완전 제거 중...")
        
        print(f"  제거 전: X_train shape = {X_train.shape}")
        print(f"  NaN 값 개수: {X_train.isnull().sum().sum()}개")
        
        # 방법 1: NaN이 있는 행 완전 제거
        nan_mask = X_train.isnull().any(axis=1)
        clean_indices = ~nan_mask
        
        X_train_clean = X_train[clean_indices]
        y_train_clean = y_train[clean_indices]
        
        print(f"  제거 후: X_train shape = {X_train_clean.shape}")
        print(f"  제거된 행: {nan_mask.sum()}개")
        print(f"  남은 NaN 개수: {X_train_clean.isnull().sum().sum()}개")
        
        # 만약 여전히 NaN이 있다면 0으로 대체
        if X_train_clean.isnull().sum().sum() > 0:
            print("  ⚠️ 여전히 NaN이 있어서 0으로 대체합니다.")
            X_train_clean = X_train_clean.fillna(0)
        
        # 테스트 데이터도 동일하게 처리
        X_test_clean = X_test.fillna(0)
        
        print(f"  최종 훈련 데이터: {X_train_clean.shape}, NaN: {X_train_clean.isnull().sum().sum()}개")
        print(f"  최종 테스트 데이터: {X_test_clean.shape}, NaN: {X_test_clean.isnull().sum().sum()}개")
        
        # 5. SMOTE 적용 (이제 안전함)
        print("\n3️⃣ SMOTE로 데이터 균형 조정...")
        from imblearn.over_sampling import SMOTE
        
        # SMOTE 적용 전 마지막 검증
        assert X_train_clean.isnull().sum().sum() == 0, "여전히 NaN 값이 있습니다!"
        assert not X_train_clean.isin([np.inf, -np.inf]).any().any(), "무한대 값이 있습니다!"
        
        smote = SMOTE(
            sampling_strategy=0.15,  # 1등을 15%까지
            random_state=42,
            k_neighbors=min(3, y_train_clean.sum() - 1)  # 1등 샘플 수보다 작게
        )
        
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_clean, y_train_clean)
            print(f"  ✅ SMOTE 성공!")
            print(f"  SMOTE 후: 1등 {y_train_balanced.sum()}개 / 전체 {len(y_train_balanced)}개 ({y_train_balanced.mean()*100:.2f}%)")
        except Exception as e:
            print(f"  ❌ SMOTE 실패: {e}")
            print("  원본 데이터를 그대로 사용합니다.")
            X_train_balanced = X_train_clean
            y_train_balanced = y_train_clean
        
        # 6. 스케일링
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test_clean)
        
        # 7. 정밀도 최적화 모델들
        print("\n4️⃣ 정밀도 최적화 모델 훈련...")
        
        # XGBoost import 추가
        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            print("  ⚠️ XGBoost가 설치되지 않아 제외됩니다.")
            xgb_available = False
        
        models = {
            # 🎯 정밀도 특화 RandomForest
            'PrecisionRF': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight={0: 1, 1: 10},
                max_features='sqrt',
                random_state=42
            ),
            
            # 🎯 보수적 로지스틱 회귀
            'PrecisionLR': LogisticRegression(
                class_weight={0: 1, 1: 15},
                C=0.05,
                max_iter=2000,
                random_state=42
            )
        }
        
        # XGBoost가 있을 때만 추가
        if xgb_available:
            models['PrecisionXGB'] = xgb.XGBClassifier(
                n_estimators=200,  # 줄임
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=8,
                random_state=42,
                eval_metric='logloss',  # 명시적으로 설정
                use_label_encoder=False  # 경고 방지
            )
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔥 {name} 훈련 중...")
            
            try:
                # 훈련
                if 'XGB' in name and xgb_available:
                    # DataFrame을 numpy array로 변환
                    X_train_xgb = X_train_balanced.values
                    X_test_xgb = X_test_clean.values
                    
                    try:
                        # eval_set 사용 시도
                        model.fit(
                            X_train_xgb, y_train_balanced,
                            eval_set=[(X_test_xgb, y_test)],  # numpy로 변환된 데이터 사용
                            verbose=False
                        )
                    except (TypeError, AttributeError):
                        # 기본 방식으로 재시도
                        model.fit(X_train_xgb, y_train_balanced)
                    
                    y_pred = model.predict(X_test_xgb)  # numpy 사용
                    y_prob = model.predict_proba(X_test_xgb)[:, 1]  # numpy 사용
                elif 'LR' in name:
                    model.fit(X_train_scaled, y_train_balanced)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    
                else:  # RandomForest
                    model.fit(X_train_balanced, y_train_balanced)
                    y_pred = model.predict(X_test_clean)
                    y_prob = model.predict_proba(X_test_clean)[:, 1]
                
                # 성능 평가
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_prob)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'probabilities': y_prob
                }
                
                print(f"  정확도: {accuracy:.3f}")
                print(f"  정밀도: {precision:.3f} ⭐⭐⭐")
                print(f"  재현율: {recall:.3f}")
                print(f"  F1: {f1:.3f}")
                
            except Exception as e:
                print(f"  ❌ {name} 모델 훈련 실패: {e}")
                continue
        
        if not results:
            print("❌ 모든 모델 훈련이 실패했습니다.")
            return None
        
        # 8. 앙상블 및 임계값 최적화
        print("\n5️⃣ 정밀도 기반 스마트 앙상블...")
        
        # 정밀도가 높은 모델에 더 높은 가중치
        precision_scores = [results[name]['precision'] for name in results]
        
        if max(precision_scores) > 0:
            max_precision = max(precision_scores)
            weights = []
            for p in precision_scores:
                weight = (p / max_precision) ** 2 if p > 0 else 0.1
                weights.append(weight)
            weights = np.array(weights) / np.sum(weights)
        else:
            weights = np.ones(len(precision_scores)) / len(precision_scores)
        
        # 앙상블 예측
        ensemble_prob = np.average(
            [results[name]['probabilities'] for name in results],
            axis=0,
            weights=weights
        )
        
        # 최적 임계값 찾기
        best_threshold = 0.5
        best_precision = 0
        
        for threshold in np.arange(0.3, 0.9, 0.02):
            pred = (ensemble_prob > threshold).astype(int)
            if pred.sum() > 0:
                prec = precision_score(y_test, pred, zero_division=0)
                if prec > best_precision:
                    best_precision = prec
                    best_threshold = threshold
        
        ensemble_pred = (ensemble_prob > best_threshold).astype(int)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_precision = precision_score(y_test, ensemble_pred, zero_division=0)
        ensemble_recall = recall_score(y_test, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        
        print(f"\n🎭 최적화된 앙상블 결과:")
        print(f"  최적 임계값: {best_threshold:.3f}")
        print(f"  정확도: {ensemble_accuracy:.3f}")
        print(f"  정밀도: {ensemble_precision:.3f} 🎯🎯🎯")
        print(f"  재현율: {ensemble_recall:.3f}")
        print(f"  F1: {ensemble_f1:.3f}")
        
        # 9. 모델 저장
        self.models = results
        self.scaler = scaler
        self.best_threshold = best_threshold
        
        # ModelManager를 통한 안전한 저장
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'best_threshold': self.best_threshold,
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        success = self.model_manager.save_model_safe(model_name, model_data)
        if success:
            print(f"💾 모델이 안전하게 저장되었습니다: {model_name}")
        else:
            print(f"⚠️ 모델 저장에 실패했지만 메모리에는 로드되어 있습니다.")
        
        print(f"\n✅ 정밀도 극대화 모델 훈련 완료!")
        print(f"🎯 최종 정밀도: {ensemble_precision:.1%}")
        
        return {
            'results': results,
            'ensemble_precision': ensemble_precision,
            'ensemble_accuracy': ensemble_accuracy,
            'best_threshold': best_threshold,
            'weights': dict(zip(results.keys(), weights.round(3)))
        }

    # 예측 함수도 업데이트
    def predict_with_precision_focus(self, race_date, meet_code=None, race_no=None):
        """
        정밀도 중심 예측 (보수적 접근)
        """
        # 기존 예측 로직 실행
        result_df = self.predict_race_winners(race_date, meet_code, race_no)
        
        if result_df is None:
            return None
        
        # 보수적 임계값 적용
        threshold = getattr(self, 'best_threshold', 0.6)
        result_df['high_confidence'] = (result_df['win_probability'] > threshold).astype(int)
        result_df['recommendation'] = result_df['high_confidence'].map({
            1: '🎯 강력 추천',
            0: '⚠️ 보류'
        })
        
        print(f"\n🎯 정밀도 중심 추천 (임계값: {threshold:.3f}):")
        high_conf = result_df[result_df['high_confidence'] == 1]
        
        if len(high_conf) > 0:
            print(f"강력 추천: {len(high_conf)}마리")
            for _, horse in high_conf.iterrows():
                print(f"  🏆 {horse['horse_name']} (확률: {horse['win_probability']:.3f})")
        else:
            print("⚠️ 이번 경주는 확신할 만한 말이 없습니다.")
        
        return result_df
    
    # 기존 HorseRacing1stPlacePredictor 클래스에 추가할 함수

    def create_racing_specific_features(self, df):
        """
        경마에 특화된 고급 특성 생성 (정밀도 향상에 핵심)
        """
        print("🏇 경마 특화 특성 생성 중...")
        
        # 1. 🎯 상대적 경쟁력 (경주 내에서의 상대적 위치)
        def calculate_relative_strength(group):
            # 경주 내에서 각 말의 상대적 실력
            group['relative_experience'] = (group['prev_total_races'] - group['prev_total_races'].mean()) / (group['prev_total_races'].std() + 1)
            group['relative_win_rate'] = (group['horse_win_rate'] - group['horse_win_rate'].mean()) / (group['horse_win_rate'].std() + 0.01)
            group['relative_recent_form'] = (group['recent_form'] - group['recent_form'].mean()) / (group['recent_form'].std() + 0.1)
            
            # 경주 내 랭킹 (1등 가능성이 높을수록 낮은 숫자)
            group['experience_rank_in_race'] = group['prev_total_races'].rank(ascending=False, method='min')
            group['win_rate_rank_in_race'] = group['horse_win_rate'].rank(ascending=False, method='min')
            group['recent_form_rank_in_race'] = group['recent_form'].rank(ascending=False, method='min')
            
            return group
        
        df = df.groupby(['race_date', 'meet_code', 'race_id']).apply(calculate_relative_strength).reset_index(drop=True)
        
        # 2. 🎯 종합 우위 지수 (가장 중요!)
        df['dominance_score'] = (
            (df['relative_win_rate'] * 0.4) +           # 승률이 가장 중요
            (df['relative_recent_form'] * 0.3) +        # 최근 폼
            (df['relative_experience'] * 0.2) +         # 경험
            (-df['entry_number'] / df['total_horses'] * 0.1)  # 출전 번호 (낮을수록 유리)
        )
        
        # 3. 🎯 일관성 지수 (안정적인 말일수록 1등 가능성 높음)
        df['consistency_score'] = np.where(
            df['prev_total_races'] >= 5,
            1 / (df['prev_total_avg_rank'].fillna(6) + 0.1),  # 평균 순위가 좋을수록 높은 점수
            0.1  # 경험 부족하면 낮은 점수
        )
        
        # 4. 🎯 거리 적합성 (간단 버전)
        df['distance_fitness'] = np.where(
            df['races_at_distance'].fillna(0) >= 2,
            1 / (df['avg_rank_at_distance'].fillna(6) + 0.1),
            0.5  # 해당 거리 경험 없으면 중간 점수
        )
        
        # 5. 🎯 기수-말 궁합 (기수 승률로 대체)
        df['jockey_horse_synergy'] = df['jockey_win_rate']
        
        # 6. 🎯 컨디션 지표 (최근 성적 기반)
        df['momentum'] = np.where(
            df['prev_5_avg_rank'].notna(),
            (6 - df['prev_5_avg_rank']) / 2,  # 최근 5경주 성적을 모멘텀으로 변환
            0
        )
        
        # 7. 🎯 최종 우승 확률 점수 (모든 요소 종합)
        df['championship_probability'] = (
            df['dominance_score'] * 0.25 +
            df['consistency_score'] * 0.20 +
            df['distance_fitness'] * 0.15 +
            df['jockey_horse_synergy'] * 0.15 +
            (df['momentum'] / 3 + 0.33) * 0.10 +  # 정규화
            df['horse_win_rate'] * 0.15
        )


        # 1. 거리별 성능 점수 (가장 중요!)
        df['distance_performance_score'] = np.where(
            df['races_at_distance'] >= 2,
            (6 - df['avg_rank_at_distance'].fillna(6)) / 5 * np.log1p(df['races_at_distance']),
            0.1
        )

        # 2. 동일거리 시간 상대적 우위
        def calc_time_advantage(group):
            if 'prev_3_avg_time_same_distance' in group.columns:
                # 시간이 낮을수록 좋으므로 역순 랭킹
                group['time_rank_in_race'] = group['prev_3_avg_time_same_distance'].rank(method='min')
                group['time_advantage_score'] = (len(group) + 1 - group['time_rank_in_race']) / len(group)
            else:
                group['time_advantage_score'] = 0.5
            return group

        df = df.groupby(['race_date', 'meet_code', 'race_id']).apply(calc_time_advantage).reset_index(drop=True)

        # 3. 속도 기반 경쟁력
        df['speed_competitiveness'] = np.where(
            df['prev_3_avg_speed_mps'] > 0,
            df['prev_3_avg_speed_mps'] * (1 + df['distance_performance_score']),
            0
        )

        # 4. 종합 실전 점수 (핵심!)
        df['practical_racing_score'] = (
            df['distance_performance_score'] * 0.4 +
            df['time_advantage_score'] * 0.3 +
            (df['speed_competitiveness'] / (df['speed_competitiveness'].max() + 0.1)) * 0.3
        )
        
        # 8. 🎯 경주별 상대 순위 (가장 중요한 특성!)
        def assign_race_rankings(group):
            group['championship_rank'] = group['championship_probability'].rank(ascending=False, method='min')
            group['is_top_candidate'] = (group['championship_rank'] <= 3).astype(int)
            group['is_clear_favorite'] = (group['championship_rank'] == 1).astype(int)
            return group
        
        df = df.groupby(['race_date', 'meet_code', 'race_id']).apply(assign_race_rankings).reset_index(drop=True)
        
        # 새로운 특성들 추가
        new_features = [
            'dominance_score', 'consistency_score', 'distance_fitness', 
            'jockey_horse_synergy', 'momentum', 'championship_probability',
            'championship_rank', 'is_top_candidate', 'is_clear_favorite',
            'relative_win_rate', 'relative_recent_form', 'win_rate_rank_in_race'
            'distance_performance_score', 'time_advantage_score', 
            'speed_competitiveness', 'practical_racing_score'
        ]
        
        print(f"✅ {len(new_features)}개 경마 특화 특성 생성 완료")
        print("🎯 핵심 특성: championship_probability, dominance_score, is_clear_favorite")
        
        return df
    def list_available_models(self):
        """
        사용 가능한 모델 목록 조회
        """
        return self.model_manager.list_saved_models()

    def check_current_model_performance(self):
        """
        현재 로드된 모델 성능 확인
        """
        return self.model_manager.check_model_performance()

    def cleanup_old_models(self, keep_latest=3):
        """
        오래된 모델 파일 정리
        """
        return self.model_manager.cleanup_old_models(keep_latest)

    def export_model_summary(self, output_file="model_summary.json"):
        """
        모델 요약 정보 내보내기
        """
        return self.model_manager.export_model_info(output_file)
    
    
    # 상세 NaN 분석 함수들

    def analyze_nan_details(self, df):
        """
        NaN 데이터 상세 분석
        """
        print("🔍 NaN 데이터 상세 분석")
        print("=" * 80)
        
        # 1. 전체 현황
        total_cells = len(df) * len(df.columns)
        nan_cells = df.isnull().sum().sum()
        print(f"📊 전체 현황:")
        print(f"  데이터 크기: {df.shape}")
        print(f"  전체 셀: {total_cells:,}개")
        print(f"  NaN 셀: {nan_cells:,}개 ({nan_cells/total_cells*100:.2f}%)")
        
        # 2. 컬럼별 NaN 상세 분석
        print(f"\n📋 컬럼별 NaN 상세 분석:")
        print("=" * 80)
        
        nan_summary = []
        for col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                nan_percentage = (nan_count / len(df)) * 100
                
                # 데이터 타입 확인
                dtype = str(df[col].dtype)
                
                # 유니크 값 개수 (NaN 제외)
                unique_count = df[col].nunique()
                
                # 샘플 값들 (NaN 아닌 것들)
                sample_values = df[col].dropna().head(3).tolist()
                
                nan_summary.append({
                    'column': col,
                    'nan_count': nan_count,
                    'nan_percentage': nan_percentage,
                    'dtype': dtype,
                    'unique_count': unique_count,
                    'sample_values': sample_values
                })
        
        # NaN 비율로 정렬
        nan_summary.sort(key=lambda x: x['nan_percentage'], reverse=True)
        
        for info in nan_summary:
            print(f"\n🔹 {info['column']}")
            print(f"   NaN: {info['nan_count']:,}개 ({info['nan_percentage']:.1f}%)")
            print(f"   타입: {info['dtype']}")
            print(f"   유니크값: {info['unique_count']:,}개")
            print(f"   샘플: {info['sample_values']}")
        
        # 3. 카테고리별 분석
        print(f"\n🏷️ 카테고리별 NaN 분석:")
        print("=" * 80)
        
        categories = {
            '🆕 API 신규 컬럼': [
                'recent_race_rating', 'recent_horse_weight', 'recent_burden_weight',
                'api_total_races', 'api_total_wins', 'api_total_places',
                'api_total_win_rate', 'api_total_place_rate',
                'api_year_races', 'api_year_wins', 'api_year_win_rate', 'api_year_place_rate'
            ],
            '🏇 경마 특화 특성': [
                'championship_probability', 'dominance_score', 'consistency_score',
                'distance_fitness', 'jockey_horse_synergy', 'momentum',
                'championship_rank', 'is_clear_favorite', 'relative_win_rate'
            ],
            '📊 과거 성적': [
                'prev_total_races', 'prev_5_avg_rank', 'prev_total_avg_rank',
                'prev_wins', 'prev_top3', 'avg_rank_at_distance', 'races_at_distance'
            ],
            '👤 기수/조교사': [
                'jockey_total_races', 'jockey_total_wins', 'jockey_year_races', 'jockey_year_wins',
                'trainer_total_races', 'trainer_total_wins', 'trainer_year_races', 'trainer_year_wins'
            ],
            '🏁 기본 정보': [
                'horse_weight', 'horse_age', 'is_male', 'horse_class', 'race_distance',

                'total_horses','race_grade', 'track_condition', 'weather'
            ]
        }
        
        for category, cols in categories.items():
            category_cols = [col for col in cols if col in df.columns]
            if category_cols:
                category_nan = sum(df[col].isnull().sum() for col in category_cols)
                category_total = len(df) * len(category_cols)
                
                print(f"\n{category}:")
                print(f"  전체 NaN: {category_nan:,}/{category_total:,} ({category_nan/category_total*100:.1f}%)")
                
                for col in category_cols:
                    if col in df.columns:
                        nan_count = df[col].isnull().sum()
                        if nan_count > 0:
                            print(f"    ❌ {col}: {nan_count:,}개 ({nan_count/len(df)*100:.1f}%)")
                        else:
                            print(f"    ✅ {col}: 0개")
        
        # 4. NaN 패턴 분석
        print(f"\n🔗 NaN 패턴 분석:")
        print("=" * 80)
        
        # 완전히 NaN인 행
        all_nan_mask = df.isnull().all(axis=1)
        all_nan_count = all_nan_mask.sum()
        print(f"  모든 컬럼이 NaN인 행: {all_nan_count:,}개")
        
        # 50% 이상 NaN인 행
        nan_per_row = df.isnull().sum(axis=1)
        mostly_nan_mask = nan_per_row > (len(df.columns) * 0.5)
        mostly_nan_count = mostly_nan_mask.sum()
        print(f"  50% 이상 NaN인 행: {mostly_nan_count:,}개")
        
        # NaN 개수별 행 분포
        print(f"\n  📈 행별 NaN 개수 분포:")
        nan_counts = nan_per_row.value_counts().sort_index()
        for nan_count, row_count in nan_counts.head(10).items():
            print(f"    NaN {nan_count:2d}개인 행: {row_count:,}개")
        
        # 5. 가장 문제가 되는 컬럼들 찾기
        print(f"\n🚨 가장 문제가 되는 컬럼들 (NaN 50% 이상):")
        print("=" * 80)
        
        problematic_cols = []
        for col in df.columns:
            nan_percentage = (df[col].isnull().sum() / len(df)) * 100
            if nan_percentage >= 50:
                problematic_cols.append((col, nan_percentage))
        
        problematic_cols.sort(key=lambda x: x[1], reverse=True)
        
        if problematic_cols:
            for col, percentage in problematic_cols:
                print(f"  ❌ {col}: {percentage:.1f}% NaN")
            
            print(f"\n💡 제안사항:")
            print(f"  1. 위 컬럼들을 특성에서 제외하거나")
            print(f"  2. 기본값으로 대체하거나") 
            print(f"  3. 데이터 수집 과정을 점검해보세요")
        else:
            print(f"  ✅ 50% 이상 NaN인 컬럼은 없습니다")
        
        # 6. 샘플 행 분석
        print(f"\n🔍 NaN 샘플 행 분석 (상위 5개 행):")
        print("=" * 80)
        
        # NaN이 많은 행들 찾기
        nan_per_row = df.isnull().sum(axis=1)
        top_nan_rows = nan_per_row.nlargest(5).index
        
        for idx in top_nan_rows:
            row_nan_count = nan_per_row[idx]
            print(f"\n  📍 행 {idx}: {row_nan_count}개 NaN")
            
            # 해당 행의 NaN인 컬럼들 보여주기
            nan_cols = df.loc[idx].isnull()
            nan_col_names = df.columns[nan_cols].tolist()
            
            if len(nan_col_names) <= 10:
                print(f"    NaN 컬럼들: {nan_col_names}")
            else:
                print(f"    NaN 컬럼들: {nan_col_names[:10]}... (총 {len(nan_col_names)}개)")
            
            # 해당 행의 정상 데이터 몇 개 보여주기
            valid_data = df.loc[idx].dropna()
            if len(valid_data) > 0:
                print(f"    정상 데이터 샘플: {dict(valid_data.head(3))}")
        
        return nan_summary

    def show_nan_heatmap(self, df, max_cols=20):
        """
        NaN 히트맵 시각화 (텍스트 버전)
        """
        print(f"\n🔥 NaN 히트맵 (상위 {max_cols}개 컬럼):")
        print("=" * 80)
        
        # NaN이 많은 컬럼들 선택
        nan_counts = df.isnull().sum().sort_values(ascending=False)
        top_nan_cols = nan_counts.head(max_cols).index.tolist()
        
        if not top_nan_cols:
            print("✅ NaN이 있는 컬럼이 없습니다!")
            return
        
        # 샘플 행들 (100개씩)
        sample_rows = range(0, min(len(df), 1000), 10)  # 10개씩 건너뛰며 100개 행
        
        print("    " + "".join(f"{i%10}" for i in range(len(top_nan_cols))))
        print("    " + "-" * len(top_nan_cols))
        
        for row_idx in sample_rows:
            if row_idx >= len(df):
                break
                
            row_display = f"{row_idx:3d}|"
            for col in top_nan_cols:
                if df.loc[row_idx, col] is pd.isna(df.loc[row_idx, col]) and pd.isna(df.loc[row_idx, col]):
                    row_display += "X"  # NaN
                else:
                    row_display += "."  # 정상 데이터
            
            print(row_display)
            
            if len(sample_rows) > 20 and row_idx == sample_rows[19]:
                print("    ... (중간 생략) ...")
                break
        
        print("\n  범례: X = NaN, . = 정상 데이터")
        print("  컬럼 순서 (NaN 많은 순):")
        for i, col in enumerate(top_nan_cols):
            nan_count = df[col].isnull().sum()
            print(f"    {i%10}: {col} ({nan_count:,}개 NaN)")

    # 사용 예시 함수
    def full_nan_analysis(self, df):
        """
        완전한 NaN 분석 실행
        """
        print("🚀 완전한 NaN 분석 시작!")
        print("=" * 100)
        
        # 1. 상세 분석
        nan_summary = self.analyze_nan_details(df)
        
        # 2. 히트맵
        self.show_nan_heatmap(df)
        
        # 3. 요약 및 제안
        print(f"\n📝 분석 요약 및 제안:")
        print("=" * 80)
        
        total_nan = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        
        if total_nan == 0:
            print("✅ NaN 문제 없음!")
        elif total_nan / total_cells < 0.1:
            print("✅ NaN 비율 낮음 (10% 미만) - 간단한 대체로 해결 가능")
        elif total_nan / total_cells < 0.3:
            print("⚠️ NaN 비율 보통 (10-30%) - 신중한 대체 전략 필요")
        else:
            print("🚨 NaN 비율 높음 (30% 이상) - 데이터 수집 과정 점검 필요")
        
        # 가장 문제되는 컬럼들
        high_nan_cols = [info for info in nan_summary if info['nan_percentage'] > 50]
        if high_nan_cols:
            print(f"\n🚨 제거 고려 대상 컬럼들 (NaN 50% 이상):")
            for info in high_nan_cols:
                print(f"  - {info['column']}: {info['nan_percentage']:.1f}% NaN")
        
        return nan_summary

    # 기존 HorseRacing1stPlacePredictor 클래스에 추가할 새로운 메서드들

    def train_ranking_models(self, df, test_size=0.2, model_name='ranking_model'):
        """
        순위 직접 예측 모델 훈련 (회귀 + 분류 하이브리드)
        """
        print("🎯 순위 예측 모델 훈련 시작!")
        df = self.safe_convert_to_numeric(df)
        
        # 1. 경마 특화 특성 생성
        print("\n1️⃣ 경마 특화 특성 생성...")
        df = self.create_racing_specific_features(df)
        
        # 2. 특성 준비
        feature_cols = [
            'horse_weight', 'horse_age', 'is_male', 'horse_class', 'race_distance', 
            'total_horses', 'race_grade', 'track_condition', 'weather',
            'prev_total_races', 'prev_5_avg_rank', 'prev_total_avg_rank',
            'prev_wins', 'prev_top3', 'prev_top5',
            'jockey_total_races', 'jockey_total_wins', 
            'trainer_total_races', 'trainer_total_wins',
            'avg_rank_at_distance', 'races_at_distance',
            'championship_probability', 'dominance_score', 'consistency_score', 
            'distance_fitness', 'momentum', 'relative_win_rate'
        ]
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        # 3. 데이터 분할
        df_sorted = df.sort_values('race_date')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        X_train = df_sorted.iloc[:split_idx][feature_cols].fillna(0)
        X_test = df_sorted.iloc[split_idx:][feature_cols].fillna(0)
        y_rank_train = df_sorted.iloc[:split_idx]['final_rank']
        y_rank_test = df_sorted.iloc[split_idx:]['final_rank']
        y_win_train = (df_sorted.iloc[:split_idx]['final_rank'] == 1).astype(int)
        y_win_test = (df_sorted.iloc[split_idx:]['final_rank'] == 1).astype(int)
        
        print(f"📊 훈련 데이터: {len(X_train)}개")
        print(f"📊 순위 분포: {y_rank_train.value_counts().head()}")
        
        # 4. 스케일링
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n2️⃣ 다양한 접근법으로 모델 훈련...")
        
        results = {}
        
        # 방법 1: 순위 직접 회귀 예측
        print("\n🔥 방법 1: 순위 회귀 모델")
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        rank_models = {
            'RankRF': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'RankGB': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        for name, model in rank_models.items():
            model.fit(X_train, y_rank_train)
            rank_pred = model.predict(X_test)
            
            # 순위를 1등 확률로 변환 (순위가 낮을수록 높은 확률)
            # 공식: P(1등) = 1 / (predicted_rank + offset)
            win_prob = 1 / (rank_pred + 0.5)
            win_prob = win_prob / win_prob.max()  # 정규화
            
            # 성능 평가
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            mse = mean_squared_error(y_rank_test, rank_pred)
            mae = mean_absolute_error(y_rank_test, rank_pred)
            
            # 1등 예측 성능
            auc = roc_auc_score(y_win_test, win_prob)
            
            results[name] = {
                'model': model,
                'type': 'regression',
                'mse': mse,
                'mae': mae,
                'auc': auc,
                'probabilities': win_prob
            }
            
            print(f"  {name}: MSE={mse:.3f}, MAE={mae:.3f}, AUC={auc:.3f}")
        
        # 방법 2: 다중 클래스 분류 (순위별 클래스)
        print("\n🔥 방법 2: 다중 클래스 분류")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # 순위를 클래스로 변환 (상위 5등까지는 개별 클래스, 나머지는 '기타')
        def rank_to_class(rank):
            if rank <= 5:
                return int(rank)
            else:
                return 6  # '기타' 클래스
        
        y_class_train = y_rank_train.apply(rank_to_class)
        y_class_test = y_rank_test.apply(rank_to_class)
        
        multiclass_models = {
            'MultiRF': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight='balanced',
                random_state=42
            ),
            'MultiLR': LogisticRegression(
                multi_class='ovr',
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            )
        }
        
        for name, model in multiclass_models.items():
            if 'LR' in name:
                model.fit(X_train_scaled, y_class_train)
                class_probs = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_class_train)
                class_probs = model.predict_proba(X_test)
            
            # 1등 확률 추출
            win_prob = class_probs[:, 0] if class_probs.shape[1] > 0 else np.zeros(len(X_test))
            
            # 예측된 순위 계산 (확률이 가장 높은 클래스)
            predicted_classes = np.argmax(class_probs, axis=1) + 1
            
            # 성능 평가
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(y_class_test, predicted_classes)
            auc = roc_auc_score(y_win_test, win_prob) if len(np.unique(y_win_test)) > 1 else 0
            
            results[name] = {
                'model': model,
                'type': 'multiclass',
                'accuracy': accuracy,
                'auc': auc,
                'probabilities': win_prob
            }
            
            print(f"  {name}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
        
        # 방법 3: 순위 기반 앙상블 (가장 정교한 방법)
        print("\n🔥 방법 3: 순위 기반 스마트 앙상블")
        
        # 경주별로 상대적 순위 예측
        def predict_race_rankings(X_test_race, models_dict):
            """경주 내에서 말들의 상대적 순위 예측"""
            predictions = []
            
            for name, result in models_dict.items():
                model = result['model']
                model_type = result['type']
                
                if model_type == 'regression':
                    if 'LR' in name:
                        rank_pred = model.predict(scaler.transform(X_test_race))
                    else:
                        rank_pred = model.predict(X_test_race)
                    predictions.append(rank_pred)
                elif model_type == 'multiclass':
                    if 'LR' in name:
                        rank_pred = model.predict(scaler.transform(X_test_race))
                    else:
                        rank_pred = model.predict(X_test_race)
                    predictions.append(rank_pred)
            
            # 앙상블 예측
            if predictions:
                ensemble_ranks = np.mean(predictions, axis=0)
                # 순위를 1등 확률로 변환
                win_probs = 1 / (ensemble_ranks + 0.1)
                win_probs = win_probs / win_probs.sum()  # 확률 총합이 1이 되도록
                return win_probs
            else:
                return np.ones(len(X_test_race)) / len(X_test_race)
        
        # 테스트 데이터로 경주별 예측 수행
        test_df = df_sorted.iloc[split_idx:].copy()
        test_df['predicted_win_prob'] = 0.0
        
        for race_group in ['race_date', 'meet_code', 'race_id']:
            if race_group in test_df.columns:
                break
        
        if 'race_date' in test_df.columns:
            unique_races = test_df.groupby(['race_date', 'race_id'] if 'race_id' in test_df.columns else ['race_date'])
            
            for race_key, race_group in unique_races:
                race_indices = race_group.index
                X_race = X_test.loc[race_indices]
                
                # 경주별 예측
                race_win_probs = predict_race_rankings(X_race, results)
                test_df.loc[race_indices, 'predicted_win_prob'] = race_win_probs
        
        # 최종 앙상블 성능 평가
        ensemble_auc = roc_auc_score(y_win_test, test_df['predicted_win_prob']) if len(np.unique(y_win_test)) > 1 else 0
        
        results['Ensemble'] = {
            'type': 'ensemble',
            'auc': ensemble_auc,
            'probabilities': test_df['predicted_win_prob'].values
        }
        
        print(f"  앙상블: AUC={ensemble_auc:.3f}")
        
        # 5. 모델 저장
        self.ranking_models = results
        self.scaler = scaler
        
        model_data = {
            'models': self.ranking_models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'best_threshold': 0.6,  # 기본 임계값
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        success = self.model_manager.save_model_safe(model_name, model_data)
        if success:
            print(f"💾 순위 예측 모델이 저장되었습니다: {model_name}")
        
        print(f"\n✅ 순위 예측 모델 훈련 완료!")
        print(f"🎯 최고 AUC: {max([r['auc'] for r in results.values() if 'auc' in r]):.3f}")
        
        return results

    def predict_with_ranking(self, race_date, meet_code=None, race_no=None, show=True):
        """
        순위 예측 모델을 사용한 경주 예측
        """
        print(f"🎯 순위 기반 예측: {race_date}")
        
        # WHERE 조건 구성
        where_conditions = [f"race_date = '{race_date}'"]
        if meet_code:
            where_conditions.append(f"meet_code = '{meet_code}'")
        if race_no:
            where_conditions.append(f"race_id = '{race_no}'")  # 문자열로 처리
        
        where_clause = " AND ".join(where_conditions)
        
        # 수정된 쿼리 - 중복 제거 및 올바른 JOIN
        query = f"""
                SELECT row_to_json(r) as result
                FROM (
                    SELECT DISTINCT
                        race_id,
                        horse_id,
                        race_date,
                        meet_code,
                        entry_number,
                        horse_weight,
                        final_rank,
                        finish_time,
                        horse_race_days,
                        horse_weight_diff,
                        budam,
                        budam_weight,
                        horse_rating, 
                        is_winner,
                        
                        -- 말 정보
                        horse_age,
                        is_male,
                        horse_class,
                        horse_name,

                        -- 경주 정보
                        race_distance,
                        total_horses,
                        planned_horses,
                        race_grade,
                        track_condition,
                        weather,
                        weight_type,  

                        prev_total_races,
                        prev_5_avg_rank,
                        prev_total_avg_rank,

                        prev_3_avg_speed_mps, 
                        prev_5_avg_speed_mps,
                        speed_improvement_trend,
                        prev_3_avg_time_same_distance, 

                        prev_wins,
                        prev_top3,
                        prev_top5,
                        year_wins,
                        year_top3,
                        year_top5,
                        total_races,
                        total_win_rate,
                        total_place_rate,
                        year_races,
                        year_win_rate,
                        year_place_rate, 

                        -- 기수 정보 (NULL 처리)
                        COALESCE(jockey_total_races, 0) as jockey_total_races,
                        COALESCE(jockey_total_wins, 0) as jockey_total_wins,
                        COALESCE(jockey_year_races, 0) as jockey_year_races,
                        COALESCE(jockey_year_wins, 0) as jockey_year_wins,
                        
                        -- 조교사 정보 (NULL 처리)
                        COALESCE(trainer_total_races, 0) as trainer_total_races,
                        COALESCE(trainer_total_wins, 0) as trainer_total_wins,
                        COALESCE(trainer_year_races, 0) as trainer_year_races,
                        COALESCE(trainer_year_wins, 0) as trainer_year_wins,
                        
                        avg_rank_at_distance,
                        races_at_distance
                                            
                    FROM race_analysis_complete
                    WHERE {where_clause}
                    ORDER BY race_id, entry_number
                ) r
                """
                
        
        result = self.supabase.rpc('execute_sql', {
            'sql_query': query,
            'params': []
        }).execute()

        if not result.data:
            print("❌ 해당 경주에 대한 데이터가 없습니다.")
            return None
            
        df = pd.DataFrame([row["result"] for row in result.data])
        df = self.safe_convert_to_numeric(df)
        
        # 중복 제거 (혹시 모를 중복 데이터)
        df = df.drop_duplicates(subset=['race_id', 'race_date', 'meet_code', 'horse_id', 'entry_number'])
        
        print(f"📊 조회된 데이터: {len(df)}개 레코드")
        print(f"📊 고유 경주 수: {df[['race_date', 'meet_code', 'race_id']].drop_duplicates().shape[0]}개")
        print(f"📊 고유 말 수: {df['horse_id'].nunique()}개")
        
        # 🔧 각 말의 과거 데이터 계산
        df = self._calculate_prediction_features(df, race_date)
        
        # 🎯 핵심 수정: 경마 특화 특성 생성 추가!
        print("🏇 예측용 경마 특화 특성 생성 중...")
        try:
            df = self.create_racing_specific_features(df)
            print("✅ 경마 특화 특성 생성 완료")
        except Exception as e:
            print(f"⚠️ 경마 특화 특성 생성 실패: {e}")
            print("기본 특성만으로 예측을 진행합니다.")
        
        # 데이터 전처리
        copy_df = df.copy()
        df = self._preprocess_data(df, is_training=False)
    
        # 순위 예측 모델들로 예측
        if not self.models:
            print("❌ 순위 예측 모델이 없습니다. 먼저 train_ranking_models()를 실행하세요.")
            return None
        
        # 각 모델로 예측 수행
        predictions = []
        for name, result in self.models.items():
            if 'rank' in name.lower():
                # 회귀 모델: 순위 직접 예측
                rank_pred = result['model'].predict(df[self.feature_columns])
                win_prob = 1 / (rank_pred + 0.1)
            elif  'multi' in name.lower():
                # 다중 클래스: 1등 확률 직접 추출
                probs = result['model'].predict_proba(df[self.feature_columns])
                win_prob = probs[:, 0]  # 1등(클래스 1) 확률
            
            predictions.append(win_prob)
        
        # 앙상블 예측
        if predictions:
            ensemble_prob = np.mean(predictions, axis=0)
            
            # 경주 내에서 확률 정규화 (총합 = 1)
            ensemble_prob = ensemble_prob / ensemble_prob.sum()
        else:
            ensemble_prob = np.ones(len(df)) / len(df)
        
        # 결과 정리
        result_df = copy_df[['race_id', 'horse_name', 'entry_number', 'final_rank']].copy()
        result_df['predicted_rank'] = ensemble_prob.argsort().argsort() + 1  # 확률을 순위로 변환
        result_df['win_probability'] = ensemble_prob
        result_df = result_df.sort_values('predicted_rank')
        
        # 결과 출력
        print(f"\n🏆 순위 기반 예측 결과:")
        print("-" * 60)
        for idx, row in result_df.head(5).iterrows():
            actual = f"(실제: {int(row['final_rank'])}등)" if pd.notna(row['final_rank']) else ""
            print(f"  {int(row['predicted_rank'])}등 예측 | "
                f"#{int(row['entry_number'])}번 {row['horse_name']} | "
                f"확률: {row['win_probability']:.3f} {actual}")
            


        # 결과 정리
        result_df = copy_df[['race_id', 'race_date', 'meet_code','horse_id', 'horse_name', 'entry_number', 
                    'horse_age', 'horse_class', 'is_male', 'final_rank','prev_total_races']].copy()
        result_df['prediction_rank'] = ensemble_prob.argsort().argsort() + 1  # 확률을 순위로 변환
        result_df['win_probability'] = ensemble_prob         
        result_df = result_df.sort_values(['meet_code','race_id','race_date', 'prediction_rank'])

        # 🎯 정밀도 중심 추천 (임계값 적용)
        threshold = getattr(self, 'best_threshold', 0.5)
        result_df['high_confidence'] = (result_df['win_probability'] > threshold).astype(int)
        result_df['recommendation'] = result_df['high_confidence'].map({
            1: '🎯 강력 추천',
            0: '⚠️ 보류'
        })

        # 🆕 경험 부족 표시 추가
        result_df['is_inexperienced'] = (result_df['prev_total_races'] <= 5).astype(int)
        result_df['experience_flag'] = result_df['is_inexperienced'].map({
            1: '🔰 신참',  # 5경주 이하
            0: ''         # 경험 충분
        })

        if show:
            # 경주별 결과 출력
            print("\n" + "="*60)
            print("🏆 예측 결과")
            print("="*60)

            unique_races = result_df[['meet_code', 'race_id']].drop_duplicates().sort_values(['meet_code','race_id'])
            
            for _, row in unique_races.iterrows():
                race_id = row['race_id']
                meet_code = row['meet_code']
                race_data = result_df[(result_df['race_id'] == race_id) & (result_df['meet_code'] == meet_code)].head(3)
                
                print(f"\n🏁 {meet_code} 경주 {race_id}번 - TOP 5 예측")
                print("-" * 50)
                
                for idx, row in race_data.iterrows():
                    gender = '수컷' if row['is_male'] == 1 else '암컷'
                    actual_rank = f" (실제: {int(row['final_rank'])}등)" if pd.notna(row['final_rank']) else ""
                    confidence_icon = "🎯" if row['high_confidence'] == 1 else "⚠️"
                    experience_info = f" {row['experience_flag']}" if row['experience_flag'] else ""
                    experience_races = f"({int(row['prev_total_races'])}경주)" if pd.notna(row['prev_total_races']) else "(경험불명)"
            
                    print(f"  {confidence_icon} {int(row['prediction_rank'])}등 | "
                        f"#{int(row['entry_number'])}번 | "
                        f"{row['horse_name']}{experience_info} | "
                        f"{int(row['horse_age'])}세 {gender} | "
                        f"등급:{row['horse_class']} | "
                        f"확률:{row['win_probability']:.3f} | "
                        f"{row['recommendation']}"
                        f"{actual_rank}")

            # 강력 추천 요약
            high_conf = result_df[result_df['high_confidence'] == 1]
            print(f"\n🎯 정밀도 중심 추천 요약 (임계값: {threshold:.3f}):")
            if len(high_conf) > 0:
                print(f"강력 추천: {len(high_conf)}마리")
                for _, horse in high_conf.iterrows():
                    print(f"  🏆 {horse['horse_name']} (#{horse['entry_number']}번, 확률: {horse['win_probability']:.3f})")
            else:
                print("⚠️ 이번 경주는 확신할 만한 말이 없습니다.")

            # 🆕 신참 말 별도 경고
            inexperienced_in_top3 = result_df[
                (result_df['prediction_rank'] <= 3) & 
                (result_df['is_inexperienced'] == 1)
            ]
            
            if len(inexperienced_in_top3) > 0:
                print(f"\n🔰 신참 말 주의사항:")
                print("-" * 30)
                for _, horse in inexperienced_in_top3.iterrows():
                    print(f"  ⚠️ {horse['horse_name']} (#{horse['entry_number']}번): "
                        f"과거 {int(horse['prev_total_races'])}경주만 출전 - 변수 가능성 높음")
                return result_df
        
        return result_df

    def compare_prediction_methods(self, df, test_size=0.2):
        """
        기존 이진 분류 vs 순위 예측 방법 비교
        """
        print("🔬 예측 방법 비교 분석")
        print("=" * 80)
        
        # 1. 기존 이진 분류 모델 훈련
        print("\n1️⃣ 기존 이진 분류 모델 (1등/비1등)")
        binary_results = self.precision_boost_training(df, test_size, 'binary_comparison')
        
        # 2. 순위 예측 모델 훈련  
        print("\n2️⃣ 순위 예측 모델")
        ranking_results = self.train_ranking_models(df, test_size, 'ranking_comparison')
        
        # 3. 성능 비교
        print("\n3️⃣ 성능 비교 결과")
        print("=" * 80)
        
        # 이진 분류 최고 성능
        binary_best_auc = max([r['auc'] for r in binary_results['results'].values() if 'auc' in r])
        binary_best_precision = max([r['precision'] for r in binary_results['results'].values() if 'precision' in r])
        
        # 순위 예측 최고 성능
        ranking_best_auc = max([r['auc'] for r in ranking_results.values() if 'auc' in r])
        
        print(f"📊 이진 분류 모델:")
        print(f"   최고 AUC: {binary_best_auc:.3f}")
        print(f"   최고 정밀도: {binary_best_precision:.3f}")
        print(f"   앙상블 정밀도: {binary_results['ensemble_precision']:.3f}")
        
        print(f"\n📊 순위 예측 모델:")
        print(f"   최고 AUC: {ranking_best_auc:.3f}")
        print(f"   순위 정보 활용: ✅")
        print(f"   경주별 상대적 예측: ✅")
        
        # 개선도 계산
        auc_improvement = ((ranking_best_auc - binary_best_auc) / binary_best_auc) * 100
        
        print(f"\n🎯 개선 효과:")
        if auc_improvement > 0:
            print(f"   AUC 개선: +{auc_improvement:.1f}% 🎉")
            print(f"   순위 정보 활용으로 예측 성능 향상!")
        else:
            print(f"   AUC 변화: {auc_improvement:.1f}%")
            print(f"   추가 튜닝이 필요할 수 있습니다.")
        
        return {
            'binary_results': binary_results,
            'ranking_results': ranking_results,
            'auc_improvement': auc_improvement
        }