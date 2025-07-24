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
        
    # def extract_training_data(self, start_date='2023-01-01', end_date='2025-03-30'):
    #     """
    #     훈련용 데이터 추출 및 특성 생성
    #     """
    #     print("📊 데이터 추출 중...")
        
    #     all_data = []
    #     page_size = 1000
    #     offset = 0
        
    #     while True:
    #         print(f"📥 페이지 {offset//page_size + 1} 추출 중... (오프셋: {offset})")

    #         # 기본 경주 데이터 추출
    #         query = f"""
    #         WITH horse_stats AS (
    #             -- 말별 과거 성적 통계
    #             SELECT 
    #                 horse_id,
    #                 race_date,
    #                 COUNT(*) OVER (
    #                     PARTITION BY horse_id 
    #                     ORDER BY race_date 
    #                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    #                 ) as prev_total_races,
    #                 AVG(final_rank) OVER (
    #                     PARTITION BY horse_id 
    #                     ORDER BY race_date 
    #                     ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    #                 ) as prev_5_avg_rank,
    #                 AVG(final_rank) OVER (
    #                     PARTITION BY horse_id 
    #                     ORDER BY race_date 
    #                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    #                 ) as prev_total_avg_rank,
    #                 SUM(CASE WHEN final_rank = 1 THEN 1 ELSE 0 END) OVER (
    #                     PARTITION BY horse_id 
    #                     ORDER BY race_date 
    #                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    #                 ) as prev_wins,
    #                 SUM(CASE WHEN final_rank <= 3 THEN 1 ELSE 0 END) OVER (
    #                     PARTITION BY horse_id 
    #                     ORDER BY race_date 
    #                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    #                 ) as prev_top3
    #             FROM race_entries
    #             WHERE race_date BETWEEN $1::date AND $2::date
    #         ),
    #         distance_stats AS (
    #             -- 거리별 성적
    #             SELECT 
    #                 re.horse_id,
    #                 r.race_distance,
    #                 AVG(re.final_rank) as avg_rank_at_distance,
    #                 COUNT(*) as races_at_distance
    #             FROM race_entries re
    #             JOIN races r ON re.race_id = r.race_id
    #             WHERE re.race_date < $3::date
    #             GROUP BY re.horse_id, r.race_distance
    #         )
    #         select row_to_json(r) as result
    #         FROM (
    #             SELECT 
    #                 re.race_id,
    #                 re.horse_id,
    #                 re.race_date,
    #                 re.meet_code,
    #                 re.entry_number,
    #                 re.horse_weight,
    #                 re.final_rank,
    #                 CASE WHEN re.final_rank = 1 THEN 1 ELSE 0 END as is_winner,
                    
    #                 -- 말 정보
    #                 h.age as horse_age,
    #                 CASE WHEN h.gender = '수컷' THEN 1 ELSE 0 END as is_male,
    #                 h.rank as horse_class,
    #                 h.name as horse_name,
                    
    #                 -- 경주 정보
    #                 r.race_distance,
    #                 r.total_horses,
    #                 r.planned_horses,
    #                 r.race_grade,
    #                 r.track_condition,
    #                 r.weather,
    #                 r.weight_type,
                    
    #                 -- 말 과거 성적
    #                 hs.prev_total_races,
    #                 hs.prev_5_avg_rank,
    #                 hs.prev_total_avg_rank,
    #                 hs.prev_wins,
    #                 hs.prev_top3,
                    
    #                 -- 기수 정보
    #                 j.total_races as jockey_total_races,
    #                 j.total_wins as jockey_total_wins,
    #                 j.year_races as jockey_year_races,
    #                 j.year_wins as jockey_year_wins,
                    
    #                 -- 조교사 정보
    #                 t.rc_cnt_t as trainer_total_races,
    #                 t.ord1_cnt_t as trainer_total_wins,
    #                 t.rc_cnt_y as trainer_year_races,
    #                 t.ord1_cnt_y as trainer_year_wins,
                    
    #                 -- 거리별 성적
    #                 ds.avg_rank_at_distance,
    #                 ds.races_at_distance
                            
    #             FROM race_entries re
    #             JOIN horses h ON re.horse_id = h.horse_id
    #             JOIN races r ON re.race_id = r.race_id
    #             LEFT JOIN horse_stats hs ON re.horse_id = hs.horse_id AND re.race_date = hs.race_date
    #             LEFT JOIN jockeys j ON re.jk_no = j.jk_no
    #             LEFT JOIN trainers t ON re.trainer_id = t.trainer_id
    #             LEFT JOIN distance_stats ds ON re.horse_id = ds.horse_id AND r.race_distance = ds.race_distance
    #             WHERE re.race_date BETWEEN $4::date AND $5::date
    #             AND re.final_rank IS NOT NULL
    #             AND hs.prev_total_races >= 3  -- 최소 3경주 이상 출전한 말만
    #             ORDER BY re.race_date, r.race_id, re.entry_number
    #             LIMIT {page_size} OFFSET {offset}
    #         ) r
    #         """
    #         result = self.supabase.rpc('execute_sql', {
    #             'sql_query': query, 
    #             'params': [start_date, end_date, end_date, start_date, end_date]
    #         }).execute()
    #         offset += page_size
                
    #         if not result.data:
    #             break

    #         page_data = [row["result"] for row in result.data]
    #         all_data.extend(page_data)

    #         print(f"✅ {len(page_data)}개 추가 (총 {len(all_data)}개)")

    #         # 마지막 페이지인지 확인
    #         if len(page_data) < page_size:
    #             break

        
    #     if not all_data:
    #         print("❌ 데이터를 가져올 수 없습니다. RPC 함수가 설정되지 않았을 수 있습니다.")
    #         return self._extract_data_alternative(start_date, end_date)
        
    #     df = pd.DataFrame(all_data)
    #     print(f"✅ {len(df)}개 레코드 추출 완료")
        
    #     return self._preprocess_data(df, is_training=True)
    

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
                    batch_end = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    batch_end = current_date.replace(month=current_date.month + 1)
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
        page_size = 500  # 페이지 크기 줄임
        offset = 0
        
        while True:
            # 단순화된 쿼리 - Window function 최소화
            query = f"""
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
                WHERE race_date <= $1::date
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
                WHERE re.race_date <= $2::date
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
                JOIN races r ON re.race_id = r.race_id  and re.race_date = r.race_Date
                LEFT JOIN horse_stats hs ON re.horse_id = hs.horse_id AND re.race_date = hs.race_date
                LEFT JOIN jockeys j ON re.jk_no = j.jk_no
                LEFT JOIN trainers t ON re.trainer_id = t.trainer_id
                LEFT JOIN distance_stats ds ON re.horse_id = ds.horse_id AND r.race_distance = ds.race_distance
                WHERE re.race_date BETWEEN $3::date AND $4::date
                AND re.final_rank IS NOT NULL
                AND hs.prev_total_races >= 3  -- 최소 3경주 이상 출전한 말만
                ORDER BY re.race_date, r.race_id, re.entry_number
                LIMIT {page_size} OFFSET {offset}
            ) r
            """
            
            try:
                result = self.supabase.rpc('execute_sql', {
                    'sql_query': query, 
                    'params': [end_date, end_date, start_date, end_date]
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

    def _calculate_horse_features_post_process(self, df):
        """
        추출 후 Python으로 말별 특성 계산 (Window function 대신)
        """
        print("🔧 말별 특성 계산 중...")
        
        # 날짜순 정렬
        df = df.sort_values(['horse_id', 'race_date'])
        
        # 말별로 과거 성적 계산
        def calculate_horse_stats(group):
            group = group.copy()
            
            # 누적 통계 계산
            group['prev_total_races'] = range(len(group))
            group['prev_wins'] = (group['final_rank'] == 1).cumsum().shift(1, fill_value=0)
            group['prev_top3'] = (group['final_rank'] <= 3).cumsum().shift(1, fill_value=0)
            
            # 최근 5경주 평균 순위
            group['prev_5_avg_rank'] = group['final_rank'].shift(1).rolling(
                window=5, min_periods=1
            ).mean().fillna(6)
            
            # 전체 평균 순위  
            group['prev_total_avg_rank'] = group['final_rank'].shift(1).expanding().mean().fillna(6)
            
            return group
        
        df = df.groupby('horse_id').apply(calculate_horse_stats).reset_index(drop=True)
        
        # 최소 경험 필터링
        df = df[df['prev_total_races'] >= 3]
        
        print(f"✅ 특성 계산 완료: {len(df)}개 레코드")
        return df
    
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
    
    def _preprocess_data(self, df, is_training=False):
        """
        데이터 전처리
        Args:
            df: 처리할 데이터프레임
            is_training: 학습용 데이터인지 여부
        """
        print("🔧 데이터 전처리 중...")
        
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
        
        # 이상치 제거
        df = df[df['final_rank'] <= 20]
        df = df[df['total_horses'] >= 5]
        
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
                n_estimators=200, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
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

    def get_loaded_model(self, model_name:str = 'horse_racing_model'):
        """
        저장된 모델 불러오기
        """
        model_data = self.model_manager.load_model(model_name)
        
        if not model_data:
            print(f"❌ 모델 '{model_name}'을(를) 찾을 수 없습니다.")
            return False
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        
        print(f"✅ 모델 '{model_name}'이(가) 성공적으로 불러와졌습니다.")
        return True
    
    def predict_race_winners(self, race_date, meet_code=None, race_no=None):
        """
        특정 경주의 1등 예측 (수정된 버전)
        """
        print(f"🔮 {race_date} 경주 예측 중...")       

        # WHERE 조건 구성
        where_conditions = [f"re.race_date = '{race_date}'"]
        if meet_code:
            where_conditions.append(f"re.meet_code = '{meet_code}'")
        if race_no:
            where_conditions.append(f"r.race_id = '{race_no}'")  # 문자열로 처리
        
        where_clause = " AND ".join(where_conditions)
        
        # 수정된 쿼리 - 중복 제거 및 올바른 JOIN
        query = f"""
                SELECT row_to_json(r) as result
                FROM (
                    SELECT DISTINCT
                        re.race_id,
                        re.horse_id,
                        re.race_date,
                        re.meet_code,
                        re.entry_number,
                        re.horse_weight,
                        re.final_rank,
                        CASE WHEN re.final_rank = 1 THEN 1 ELSE 0 END as is_winner,
                        
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

                        -- 기수 정보 (NULL 처리)
                        COALESCE(j.total_races, 0) as jockey_total_races,
                        COALESCE(j.total_wins, 0) as jockey_total_wins,
                        COALESCE(j.year_races, 0) as jockey_year_races,
                        COALESCE(j.year_wins, 0) as jockey_year_wins,
                        
                        -- 조교사 정보 (NULL 처리)
                        COALESCE(t.rc_cnt_t, 0) as trainer_total_races,
                        COALESCE(t.ord1_cnt_t, 0) as trainer_total_wins,
                        COALESCE(t.rc_cnt_y, 0) as trainer_year_races,
                        COALESCE(t.ord1_cnt_y, 0) as trainer_year_wins
                                            
                    FROM race_entries re
                    JOIN horses h ON re.horse_id = h.horse_id
                    JOIN races r ON re.race_id = r.race_id AND re.race_date = r.race_date
                    LEFT JOIN jockeys j ON re.jk_no = j.jk_no
                    LEFT JOIN trainers t ON re.trainer_id = t.trainer_id
                    WHERE {where_clause}
                    ORDER BY re.race_id, re.entry_number
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
            
            # 중복 제거 (혹시 모를 중복 데이터)
            df = df.drop_duplicates(subset=['race_id', 'horse_id', 'entry_number'])
            
            print(f"📊 조회된 데이터: {len(df)}개 레코드")
            print(f"📊 고유 경주 수: {df['race_id'].nunique()}개")
            print(f"📊 고유 말 수: {df['horse_id'].nunique()}개")
            
            # 각 말의 과거 데이터 계산
            df = self._calculate_prediction_features(df, race_date)
            df = self._preprocess_data(df, is_training=False)
            
            # 모델이 학습되지 않았다면 에러
            if not self.models:
                print("❌ 모델이 학습되지 않았습니다. 먼저 모델을 학습하거나 불러와주세요.")
                return None
            
            # 필요한 특성이 있는지 확인
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                print(f"❌ 누락된 특성: {missing_features}")
                return None
            
            # 예측 수행
            predictions = []
            
            for name, result in self.models.items():
                model = result['model']
                
                try:
                    if name == 'LogisticRegression':
                        X_scaled = self.scaler.transform(df[self.feature_columns])
                        prob = model.predict_proba(X_scaled)[:, 1]
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
            result_df = df[['race_id', 'horse_id', 'horse_name', 'entry_number', 
                        'horse_age', 'horse_class', 'is_male', 'final_rank']].copy()
            result_df['win_probability'] = ensemble_prob
            
            # 경주별로 예측 등수 계산 (수정된 부분)
            def calculate_race_rank(group):
                group = group.copy()
                group['prediction_rank'] = group['win_probability'].rank(ascending=False, method='min').astype(int)
                return group
            
            result_df = result_df.groupby('race_id').apply(calculate_race_rank).reset_index(drop=True)
            result_df = result_df.sort_values(['race_id', 'prediction_rank'])

            # 경주별 결과 출력 (수정된 부분)
            print("\n" + "="*60)
            print("🏆 예측 결과")
            print("="*60)
            
            for race_id in sorted(result_df['race_id'].unique()):
                race_data = result_df[result_df['race_id'] == race_id].head(3)
                
                print(f"\n🏁 경주 {race_id}번 - TOP 3 예측")
                print("-" * 50)
                
                for idx, row in race_data.iterrows():
                    gender = '수컷' if row['is_male'] == 1 else '암컷'
                    actual_rank = f" (실제: {int(row['final_rank'])}등)" if pd.notna(row['final_rank']) else ""
                    
                    print(f"  {int(row['prediction_rank'])}등 | "
                        f"#{int(row['entry_number'])}번 | "
                        f"{row['horse_name']} | "
                        f"ID:{row['horse_id']} | "
                        f"{int(row['horse_age'])}세 {gender} | "
                        f"등급:{row['horse_class']} | "
                        f"확률:{row['win_probability']:.3f}"
                        f"{actual_rank}")

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
        # 각 말의 과거 성적을 current_date 이전 데이터로 계산
        # 실제 구현에서는 별도 쿼리로 과거 데이터를 가져와야 함
        
        for horse_id in df['horse_id'].unique():
            # 과거 성적 조회 쿼리
            past_races = self.supabase.table('race_entries')\
                .select('final_rank')\
                .eq('horse_id', horse_id)\
                .lt('race_date', current_date)\
                .order('race_date', desc=True)\
                .execute()
            
            if past_races.data:
                ranks = [r['final_rank'] for r in past_races.data]
                
                # 특성 계산
                mask = df['horse_id'] == horse_id
                df.loc[mask, 'prev_total_races'] = len(ranks)
                df.loc[mask, 'prev_5_avg_rank'] = np.mean(ranks[:5]) if ranks else 6
                df.loc[mask, 'prev_total_avg_rank'] = np.mean(ranks) if ranks else 6
                df.loc[mask, 'prev_wins'] = sum(1 for r in ranks if r == 1)
                df.loc[mask, 'prev_top3'] = sum(1 for r in ranks if r <= 3)
        
        return df
    
    def backtest_strategy(self, start_date, end_date, confidence_threshold=0.7):
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

        detailed_results = [] # 상세결과 저장
        
        for race in races.data[:50]:  # 테스트용으로 50경주만
            try:
                predictions = self.predict_race_winners(
                    race['race_date'], 
                    race['meet_code'], 
                    race['race_id']
                )
                
                if isinstance(predictions, str):
                    continue
                
                # 가장 확신하는 말에 베팅
                best_horse = predictions.iloc[3]
                
                # 상위 3마리 말 선택 (확률 높은 순)
                top_picks = predictions.head(3)  # 상위 3마리
            
                for idx, horse in top_picks.iterrows():
                    if horse['win_probability'] > confidence_threshold:
                        total_bets += 1
                        
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
                                    profit = 1000 
                                    if actual_rank == 1:
                                        wins += 1
                                total_profit += profit
                            else:
                                total_profit -= 1000  # 실패 시 베팅금 손실
                            
                            # 디버깅용 출력
                            status = "✅ TOP3" if actual_rank <= 3 else "❌ 실패"
                            print(f"  {race['race_date']} R{race['race_id']} {horse_id}({horse['entry_number']}번): {horse['win_probability']:.3f} → {actual_rank}등 {status}")
                            
                        
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
            print(f"  총 수익: {total_profit:,}원")
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
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            
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