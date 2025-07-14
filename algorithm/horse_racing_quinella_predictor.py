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
import sys
import os
import json
from itertools import combinations, permutations

# 상위 디렉토리의 const.py 가져오기
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from const import API_KEY, SUPABASE_URL, SUPABASE_KEY

warnings.filterwarnings('ignore')

class HorseRacingQuinellaPredictor:
    """경마 연승 예측 모델 (1-2등 조합 예측)"""
    
    def __init__(self, supabase_url, supabase_key):
        """
        경마 연승 예측 모델 초기화
        
        Args:
            supabase_url: Supabase 프로젝트 URL
            supabase_key: Supabase API 키
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def extract_training_data(self, start_date='2023-01-01', end_date='2024-12-31'):
        """
        연승 예측용 훈련 데이터 추출
        """
        print("📊 연승 예측용 데이터 추출 중...")
        
        # 기본 경주 데이터 추출 (단승과 동일하지만 타겟이 다름)
        query = """
        WITH horse_stats AS (
            -- 말별 과거 성적 통계
            SELECT 
                horse_id,
                race_date,
                COUNT(*) OVER (
                    PARTITION BY horse_id 
                    ORDER BY race_date 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prev_total_races,
                AVG(final_rank) OVER (
                    PARTITION BY horse_id 
                    ORDER BY race_date 
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) as prev_5_avg_rank,
                AVG(final_rank) OVER (
                    PARTITION BY horse_id 
                    ORDER BY race_date 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prev_total_avg_rank,
                SUM(CASE WHEN final_rank = 1 THEN 1 ELSE 0 END) OVER (
                    PARTITION BY horse_id 
                    ORDER BY race_date 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prev_wins,
                SUM(CASE WHEN final_rank <= 3 THEN 1 ELSE 0 END) OVER (
                    PARTITION BY horse_id 
                    ORDER BY race_date 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as prev_top3
            FROM race_entries
            WHERE race_date BETWEEN $1::date AND $2::date
        ),
        distance_stats AS (
            -- 거리별 성적
            SELECT 
                re.horse_id,
                r.race_distance,
                AVG(re.final_rank) as avg_rank_at_distance,
                COUNT(*) as races_at_distance
            FROM race_entries re
            JOIN races r ON re.race_id = r.race_id
            WHERE re.race_date < $3::date
            GROUP BY re.horse_id, r.race_distance
        )
        select row_to_json(r) as result
        FROM (
            SELECT 
                re.race_id,
                re.horse_id,
                re.race_date,
                re.meet_code,
                re.entry_number,
                re.horse_weight,
                re.final_rank,
                CASE WHEN re.final_rank = 1 THEN 1 ELSE 0 END as is_winner,
                CASE WHEN re.final_rank <= 2 THEN 1 ELSE 0 END as is_top2,
                CASE WHEN re.final_rank <= 3 THEN 1 ELSE 0 END as is_top3,
                
                -- 말 정보
                h.age as horse_age,
                CASE WHEN h.gender = '수컷' THEN 1 ELSE 0 END as is_male,
                h.rank as horse_class,
                h.name as horse_name,
                
                -- 경주 정보
                r.race_distance,
                r.total_horses,
                r.planned_horses,
                r.race_grade,
                r.track_condition,
                r.weather,
                r.weight_type,
                
                -- 말 과거 성적
                hs.prev_total_races,
                hs.prev_5_avg_rank,
                hs.prev_total_avg_rank,
                hs.prev_wins,
                hs.prev_top3,
                
                -- 기수 정보
                j.total_races as jockey_total_races,
                j.total_wins as jockey_total_wins,
                j.year_races as jockey_year_races,
                j.year_wins as jockey_year_wins,
                
                -- 조교사 정보
                t.rc_cnt_t as trainer_total_races,
                t.ord1_cnt_t as trainer_total_wins,
                t.rc_cnt_y as trainer_year_races,
                t.ord1_cnt_y as trainer_year_wins,
                
                -- 거리별 성적
                ds.avg_rank_at_distance,
                ds.races_at_distance
                        
            FROM race_entries re
            JOIN horses h ON re.horse_id = h.horse_id
            JOIN races r ON re.race_id = r.race_id
            LEFT JOIN horse_stats hs ON re.horse_id = hs.horse_id AND re.race_date = hs.race_date
            LEFT JOIN jockeys j ON re.jk_no = j.jk_no
            LEFT JOIN trainers t ON re.trainer_id = t.trainer_id
            LEFT JOIN distance_stats ds ON re.horse_id = ds.horse_id AND r.race_distance = ds.race_distance
            WHERE re.race_date BETWEEN $4::date AND $5::date
            AND re.final_rank IS NOT NULL
            AND hs.prev_total_races >= 3  -- 최소 3경주 이상 출전한 말만
            ORDER BY re.race_date, r.race_id, re.entry_number
        ) r
        """

        # Supabase에서 직접 SQL 실행
        try:
            result = self.supabase.rpc('execute_sql', {
                'sql_query': query, 
                'params': [start_date, end_date, end_date, start_date, end_date]
            }).execute()
            
            if not result.data:
                print("❌ 데이터를 가져올 수 없습니다.")
                return pd.DataFrame()
            
            df = pd.DataFrame([row["result"] for row in result.data])
            print(f"✅ {len(df)}개 레코드 추출 완료")
            
            return self._preprocess_data_for_quinella(df, is_training=True)
            
        except Exception as e:
            print(f"❌ 데이터 추출 오류: {e}")
            return pd.DataFrame()
    
    def _preprocess_data_for_quinella(self, df, is_training=False):
        """
        연승 예측을 위한 데이터 전처리
        """
        print("🔧 연승용 데이터 전처리 중...")
        
        # 기본 전처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 카테고리 변수 처리
        categorical_cols = ['horse_class', 'race_grade', 'track_condition', 'weather']
        
        if is_training:
            print("📚 연승 학습 데이터 처리 중...")
            
            # 결측값이 있는 행 제거
            for col in categorical_cols:
                if col in df.columns:
                    before_len = len(df)
                    df = df.dropna(subset=[col])
                    after_len = len(df)
                    if before_len != after_len:
                        print(f"   {col} 결측값 {before_len - after_len}개 행 제거")
            
            # LabelEncoder 학습
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                    print(f"   {col} 인코딩 완료: {len(self.label_encoders[col].classes_)}개 클래스")
        
        else:
            print("🔮 연승 예측 데이터 처리 중...")
            
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('unknown').astype(str)
                    if col in self.label_encoders:
                        df[col] = self._safe_transform_with_unknown(df[col], col)
                    else:
                        print(f"⚠️ {col}에 대한 LabelEncoder가 없습니다!")
                        df[col] = 0
        
        # 새로운 특성 생성
        df['jockey_win_rate'] = df['jockey_total_wins'] / (df['jockey_total_races'] + 1)
        df['trainer_win_rate'] = df['trainer_total_wins'] / (df['trainer_total_races'] + 1)
        df['horse_win_rate'] = df['prev_wins'] / (df['prev_total_races'] + 1)
        df['horse_top3_rate'] = df['prev_top3'] / (df['prev_total_races'] + 1)
        df['experience_score'] = np.log1p(df['prev_total_races'])
        df['recent_form'] = 6 - df['prev_5_avg_rank'].fillna(6)
        
        # 연승 특화 특성
        df['horse_top2_rate'] = df['prev_top3'] / (df['prev_total_races'] + 1)  # 상위권 진입 확률
        df['consistency_score'] = 1 / (df['prev_5_avg_rank'].fillna(6) + 1)    # 일관성 점수
        
        # 이상치 제거
        df = df[df['final_rank'] <= 20]
        df = df[df['total_horses'] >= 5]
        
        print(f"✅ 연승용 전처리 완료: {len(df)}개 레코드")
        return df

    def _safe_transform_with_unknown(self, series, column_name):
        """
        예측 시 안전한 변환
        """
        encoder = self.label_encoders[column_name]
        known_classes = set(encoder.classes_)
        
        current_values = set(series.unique())
        unseen_values = current_values - known_classes
        
        if unseen_values:
            print(f"   ⚠️ {column_name}에서 새로운 값 발견: {unseen_values}")
            
            series_copy = series.copy()
            for unseen_val in unseen_values:
                series_copy = series_copy.replace(unseen_val, 'unknown')
            
            if 'unknown' not in known_classes:
                most_common = encoder.classes_[0]
                series_copy = series_copy.replace('unknown', most_common)
                print(f"   unknown을 {most_common}으로 대체")
        else:
            series_copy = series
        
        return encoder.transform(series_copy)
    
    def train_quinella_models(self, df, test_size=0.2):
        """
        연승 예측 모델 훈련 (1-2등 예측)
        """
        print("🤖 연승 예측 모델 훈련 중...")
        
        # 특성 선택
        feature_cols = [
            'horse_age', 'is_male', 'horse_class', 'race_distance', 'total_horses',
            'horse_weight', 'race_grade', 'track_condition', 'weather',
            'prev_total_races', 'prev_5_avg_rank', 'prev_total_avg_rank',
            'jockey_win_rate', 'trainer_win_rate', 'horse_win_rate', 'horse_top3_rate',
            'experience_score', 'recent_form', 'horse_top2_rate', 'consistency_score'
        ]
        
        # 실제로 존재하는 컬럼만 선택
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y_top2 = df['is_top2']  # 1-2등 예측용
        y_top3 = df['is_top3']  # 1-3등 예측용 (삼연승)
        
        print(f"📋 사용 특성: {len(feature_cols)}개")
        print(f"🎯 1-2등 분포: {y_top2.sum()}개 / 전체 {len(y_top2)}개 ({y_top2.mean()*100:.2f}%)")
        print(f"🎯 1-3등 분포: {y_top3.sum()}개 / 전체 {len(y_top3)}개 ({y_top3.mean()*100:.2f}%)")
        
        # 시간 순서를 고려한 분할
        df_sorted = df.sort_values('race_date')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        X_train = df_sorted.iloc[:split_idx][feature_cols]
        X_test = df_sorted.iloc[split_idx:][feature_cols]
        y_train_top2 = df_sorted.iloc[:split_idx]['is_top2']
        y_test_top2 = df_sorted.iloc[split_idx:]['is_top2']
        y_train_top3 = df_sorted.iloc[:split_idx]['is_top3']
        y_test_top3 = df_sorted.iloc[split_idx:]['is_top3']
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1-2등 예측 모델 훈련
        top2_models = {
            'RandomForest_Top2': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
            ),
            'GradientBoosting_Top2': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, random_state=42
            ),
            'LogisticRegression_Top2': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            )
        }
        
        # 1-3등 예측 모델 훈련
        top3_models = {
            'RandomForest_Top3': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
            ),
            'GradientBoosting_Top3': GradientBoostingClassifier(
                n_estimators=200, max_depth=6, random_state=42
            )
        }
        
        results = {}
        
        # 1-2등 모델 훈련 및 평가
        print(f"\n🔥 1-2등 예측 모델 훈련...")
        for name, model in top2_models.items():
            if 'LogisticRegression' in name:
                model.fit(X_train_scaled, y_train_top2)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train_top2)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test_top2, y_pred)
            precision = precision_score(y_test_top2, y_pred, zero_division=0)
            recall = recall_score(y_test_top2, y_pred, zero_division=0)
            f1 = f1_score(y_test_top2, y_pred, zero_division=0)
            auc = roc_auc_score(y_test_top2, y_prob)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'target': 'top2'
            }
            
            print(f"  {name}: 정확도={accuracy:.3f}, 정밀도={precision:.3f}, 재현율={recall:.3f}, AUC={auc:.3f}")
        
        # 1-3등 모델 훈련 및 평가
        print(f"\n🔥 1-3등 예측 모델 훈련...")
        for name, model in top3_models.items():
            model.fit(X_train, y_train_top3)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test_top3, y_pred)
            precision = precision_score(y_test_top3, y_pred, zero_division=0)
            recall = recall_score(y_test_top3, y_pred, zero_division=0)
            f1 = f1_score(y_test_top3, y_pred, zero_division=0)
            auc = roc_auc_score(y_test_top3, y_prob)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'target': 'top3'
            }
            
            print(f"  {name}: 정확도={accuracy:.3f}, 정밀도={precision:.3f}, 재현율={recall:.3f}, AUC={auc:.3f}")
        
        self.models = results
        return results

    def predict_quinella_combinations(self, race_date, meet_code=None, race_no=None, bet_type='quinella'):
        """
        연승 조합 예측
        
        Args:
            bet_type: 'quinella' (연승복식), 'exacta' (연승단식), 'trifecta' (삼연승)
        """
        print(f"🔮 {race_date} {bet_type.upper()} 예측 중...")
        
        # 데이터 조회
        where_conditions = [f"re.race_date = '{race_date}'"]
        if meet_code:
            where_conditions.append(f"re.meet_code = '{meet_code}'")
        if race_no:
            where_conditions.append(f"r.race_id = {race_no}")
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT row_to_json(r) as result
        from (
            SELECT 
                re.race_id, re.horse_id, re.race_date, re.meet_code, re.entry_number,
                re.horse_weight, re.final_rank,
                h.age as horse_age,
                CASE WHEN h.gender = '수컷' THEN 1 ELSE 0 END as is_male,
                h.rank as horse_class, h.name as horse_name,
                r.race_distance, r.total_horses, r.planned_horses,
                r.race_grade, r.track_condition, r.weather, r.weight_type,
                j.total_races as jockey_total_races, j.total_wins as jockey_total_wins,
                j.year_races as jockey_year_races, j.year_wins as jockey_year_wins,
                t.rc_cnt_t as trainer_total_races, t.ord1_cnt_t as trainer_total_wins,
                t.rc_cnt_y as trainer_year_races, t.ord1_cnt_y as trainer_year_wins
            FROM race_entries re
            JOIN horses h ON re.horse_id = h.horse_id
            JOIN races r ON re.race_id = r.race_id
            LEFT JOIN jockeys j ON re.jk_no = j.jk_no
            LEFT JOIN trainers t ON re.trainer_id = t.trainer_id
            WHERE {where_clause}
            ORDER BY re.entry_number
        )r
        """
        
        try:
            result = self.supabase.rpc('execute_sql', {'sql_query': query, 'params': []}).execute()
            
            if not result.data:
                return "❌ 해당 경주에 대한 데이터가 없습니다."
            
            df = pd.DataFrame([row["result"] for row in result.data])
            df = self._calculate_prediction_features(df, race_date)
            df = self._preprocess_data_for_quinella(df, is_training=False)
            
            # 각 말의 1-2등 확률 예측
            top2_predictions = []
            for name, result in self.models.items():
                if 'Top2' in name:
                    model = result['model']
                    if 'LogisticRegression' in name:
                        X_scaled = self.scaler.transform(df[self.feature_columns])
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        prob = model.predict_proba(df[self.feature_columns])[:, 1]
                    top2_predictions.append(prob)
            
            # 앙상블 확률
            if top2_predictions:
                ensemble_top2_prob = np.mean(top2_predictions, axis=0)
            else:
                print("⚠️ 사용 가능한 모델이 없습니다.")
                return "모델이 훈련되지 않았습니다."
            
            # 결과 데이터프레임 생성
            result_df = df[['horse_name', 'entry_number']].copy()
            result_df['top2_probability'] = ensemble_top2_prob
            result_df = result_df.sort_values('top2_probability', ascending=False)
            
            # 연승 조합 생성
            combinations_list = []
            
            if bet_type == 'quinella':  # 연승복식 (순서 상관없음)
                for combo in combinations(range(len(result_df)), 2):
                    i, j = combo
                    horse1 = result_df.iloc[i]
                    horse2 = result_df.iloc[j]
                    
                    # 두 말의 확률 조합 (곱셈 + 가중평균)
                    combined_prob = (horse1['top2_probability'] + horse2['top2_probability']) * 0.5
                    
                    combinations_list.append({
                        'bet_type': 'quinella',
                        'horse1_name': horse1['horse_name'],
                        'horse1_number': horse1['entry_number'],
                        'horse2_name': horse2['horse_name'],
                        'horse2_number': horse2['entry_number'],
                        'combination': f"{horse1['entry_number']}-{horse2['entry_number']}",
                        'probability': combined_prob,
                        'confidence': 'High' if combined_prob > 0.5 else 'Medium' if combined_prob > 0.35 else 'Low'
                    })
            
            elif bet_type == 'exacta':  # 연승단식 (순서 중요)
                for perm in permutations(range(len(result_df)), 2):
                    i, j = perm
                    horse1 = result_df.iloc[i]  # 1등 후보
                    horse2 = result_df.iloc[j]  # 2등 후보
                    
                    # 순서를 고려한 확률 (1등 확률 높게 가중)
                    combined_prob = horse1['top2_probability'] * 0.7 + horse2['top2_probability'] * 0.3
                    
                    combinations_list.append({
                        'bet_type': 'exacta',
                        'horse1_name': horse1['horse_name'],
                        'horse1_number': horse1['entry_number'],
                        'horse2_name': horse2['horse_name'],
                        'horse2_number': horse2['entry_number'],
                        'combination': f"{horse1['entry_number']}→{horse2['entry_number']}",
                        'probability': combined_prob,
                        'confidence': 'High' if combined_prob > 0.4 else 'Medium' if combined_prob > 0.25 else 'Low'
                    })
            
            # 확률 순으로 정렬
            combinations_df = pd.DataFrame(combinations_list)
            combinations_df = combinations_df.sort_values('probability', ascending=False)
            
            return {
                'individual_predictions': result_df,
                'combinations': combinations_df.head(10),  # 상위 10개 조합만
                'summary': {
                    'bet_type': bet_type,
                    'total_horses': len(result_df),
                    'total_combinations': len(combinations_df),
                    'recommended_combinations': len(combinations_df[combinations_df['confidence'] == 'High'])
                }
            }
            
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return f"예측 중 오류가 발생했습니다: {e}"

    def _calculate_prediction_features(self, df, current_date):
        """예측용 특성 계산"""
        for horse_id in df['horse_id'].unique():
            try:
                past_races = self.supabase.table('race_entries')\
                    .select('final_rank')\
                    .eq('horse_id', horse_id)\
                    .lt('race_date', current_date)\
                    .order('race_date', desc=True)\
                    .execute()
                
                if past_races.data:
                    ranks = [r['final_rank'] for r in past_races.data if r['final_rank'] is not None]
                    
                    mask = df['horse_id'] == horse_id
                    df.loc[mask, 'prev_total_races'] = len(ranks)
                    df.loc[mask, 'prev_5_avg_rank'] = np.mean(ranks[:5]) if ranks else 6
                    df.loc[mask, 'prev_total_avg_rank'] = np.mean(ranks) if ranks else 6
                    df.loc[mask, 'prev_wins'] = sum(1 for r in ranks if r == 1)
                    df.loc[mask, 'prev_top3'] = sum(1 for r in ranks if r <= 3)
                else:
                    mask = df['horse_id'] == horse_id
                    df.loc[mask, 'prev_total_races'] = 0
                    df.loc[mask, 'prev_5_avg_rank'] = 6
                    df.loc[mask, 'prev_total_avg_rank'] = 6
                    df.loc[mask, 'prev_wins'] = 0
                    df.loc[mask, 'prev_top3'] = 0
                    
            except Exception as e:
                print(f"⚠️ {horse_id} 과거 데이터 계산 오류: {e}")
                continue
        
        return df

    def backtest_quinella_strategy(self, start_date, end_date, bet_type='quinella', confidence_threshold=0.4):
        """
        연승 백테스팅 전략
        """
        print(f"📈 {bet_type.upper()} 백테스팅 수행: {start_date} ~ {end_date}")
        
        try:
            races = self.supabase.table('races')\
                .select('race_date, meet_code, race_id')\
                .gte('race_date', start_date)\
                .lte('race_date', end_date)\
                .execute()
            
            total_bets = 0
            total_profit = 0
            wins = 0
            bet_amount = 1000  # 1건당 1천원
            
            for race in races.data[:30]:  # 테스트용으로 30경주만
                try:
                    predictions = self.predict_quinella_combinations(
                        race['race_date'], 
                        race['meet_code'], 
                        race['race_id'],
                        bet_type
                    )
                    
                    if isinstance(predictions, str):
                        continue
                    
                    # 높은 확신도의 조합에만 베팅
                    high_conf_combinations = predictions['combinations'][
                        predictions['combinations']['probability'] > confidence_threshold
                    ]
                    
                    if len(high_conf_combinations) == 0:
                        continue
                    
                    # 가장 확신하는 조합에 베팅
                    best_combination = high_conf_combinations.iloc[0]
                    total_bets += 1
                    
                    # 실제 결과 확인
                    actual_results = self.supabase.table('race_entries')\
                        .select('entry_number, final_rank')\
                        .eq('race_date', race['race_date'])\
                        .eq('meet_code', race['meet_code'])\
                        .order('final_rank')\
                        .limit(3)\
                        .execute()
                    
                    if actual_results.data and len(actual_results.data) >= 2:
                        actual_1st = actual_results.data[0]['entry_number']
                        actual_2nd = actual_results.data[1]['entry_number']
                        
                        bet_numbers = [best_combination['horse1_number'], best_combination['horse2_number']]
                        
                        hit = False
                        if bet_type == 'quinella':
                            # 연승복식: 순서 상관없이 1-2등
                            hit = (actual_1st in bet_numbers and actual_2nd in bet_numbers)
                        elif bet_type == 'exacta':
                            # 연승단식: 순서대로 1-2등
                            hit = (actual_1st == best_combination['horse1_number'] and 
                                  actual_2nd == best_combination['horse2_number'])
                        
                        if hit:
                            wins += 1
                            # 연승 배당률 시뮬레이션
                            if bet_type == 'quinella':
                                payout_ratio = np.random.uniform(5, 15)
                            else:  # exacta
                                payout_ratio = np.random.uniform(8, 25)
                            
                            profit = bet_amount * payout_ratio - bet_amount
                            total_profit += profit
                            print(f"✅ 적중! {race['race_date']} {best_combination['combination']} -> +{profit:,.0f}원")
                        else:
                            total_profit -= bet_amount
                            print(f"❌ 실패: {race['race_date']} {best_combination['combination']}")
                    
                except Exception as e:
                    print(f"경주 처리 오류: {e}")
                    continue
            
            if total_bets > 0:
                win_rate = wins / total_bets
                roi = (total_profit / (total_bets * bet_amount)) * 100
                
                print(f"\n📊 {bet_type.upper()} 백테스팅 결과:")
                print(f"  총 베팅: {total_bets}회")
                print(f"  적중: {wins}회")
                print(f"  적중률: {win_rate:.1%}")
                print(f"  총 투자: {total_bets * bet_amount:,}원")
                print(f"  총 수익: {total_profit:,}원")
                print(f"  ROI: {roi:.1f}%")
                
                return {
                    'bet_type': bet_type,
                    'total_bets': total_bets,
                    'wins': wins,
                    'win_rate': win_rate,
                    'total_investment': total_bets * bet_amount,
                    'total_profit': total_profit,
                    'roi': roi
                }
            else:
                print("베팅할 경주가 없었습니다.")
                return None
                
        except Exception as e:
            print(f"❌ 백테스팅 오류: {e}")
            return None

# 사용 예시
def main():
    # 모델 초기화
    predictor = HorseRacingQuinellaPredictor(SUPABASE_URL, SUPABASE_KEY)
    
    # 1. 데이터 추출 및 연승 모델 훈련
    print("=" * 50)
    print("🏇 경마 연승 예측 모델 훈련")
    print("=" * 50)
    
    df = predictor.extract_training_data('2023-01-01', '2024-11-30')
    
    if len(df) > 0:
        results = predictor.train_quinella_models(df, test_size=0.2)
        
        # 2. 연승 조합 예측 테스트
        print("\n" + "=" * 50)
        print("🔮 연승 예측 테스트")
        print("=" * 50)
        
        quinella_prediction = predictor.predict_quinella_combinations('2024-07-28', '서울', 13, 'quinella')
        print("\n🎯 연승복식 예측 결과:")
        if isinstance(quinella_prediction, dict):
            print(quinella_prediction['combinations'].head())
        
        exacta_prediction = predictor.predict_quinella_combinations('2024-07-28', '서울', 13, 'exacta')
        print("\n🎯 연승단식 예측 결과:")
        if isinstance(exacta_prediction, dict):
            print(exacta_prediction['combinations'].head())
        
        # 3. 연승 백테스팅
        print("\n" + "=" * 50)
        print("📈 연승 백테스팅 테스트")
        print("=" * 50)
        
        quinella_backtest = predictor.backtest_quinella_strategy('2024-12-01', '2024-12-31', 'quinella')
        exacta_backtest = predictor.backtest_quinella_strategy('2024-12-01', '2024-12-31', 'exacta')
        
    else:
        print("❌ 훈련용 데이터가 부족합니다.")

if __name__ == "__main__":
    main()