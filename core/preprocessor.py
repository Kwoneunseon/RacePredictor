"""
데이터 전처리 모듈 - 특성 생성, 인코딩, 스케일링
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import FEATURE_CONFIG, DATA_EXTRACTION

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        """전처리기 초기화"""
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.config = FEATURE_CONFIG
        self.is_fitted = False
        
    def process_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        학습용 데이터 전처리
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            전처리된 데이터프레임
        """
        logger.info(f"🔧 학습 데이터 전처리 시작: {len(df)}개 레코드")
        
        # 1. 말별 특성 계산
        df = self._calculate_horse_features(df)
        
        # 2. 기본 전처리
        df = self._basic_preprocessing(df)
        
        # 3. 학습용 카테고리 처리 (결측값 제거 + 인코딩 학습)
        df = self._process_categorical_training(df)
        
        # 4. 새로운 특성 생성
        df = self._create_derived_features(df)
        
        # 5. 이상치 제거
        df = self._remove_outliers(df)
        
        # 6. 특성 컬럼 설정
        self._set_feature_columns(df)
        
        # 7. 스케일링 (학습)
        df = self._fit_scaling(df)
        
        self.is_fitted = True
        logger.info(f"✅ 학습 데이터 전처리 완료: {len(df)}개 레코드")
        
        return df
    
    def process_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        예측용 데이터 전처리
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            전처리된 데이터프레임
        """
        if not self.is_fitted:
            raise ValueError("전처리기가 학습되지 않았습니다. 먼저 process_training_data()를 호출하세요.")
        
        logger.info(f"🔮 예측 데이터 전처리 시작: {len(df)}개 레코드")
        
        # 1. 말별 특성 계산 (예측용)
        df = self._calculate_horse_features(df, is_prediction=True)
        
        # 2. 기본 전처리
        df = self._basic_preprocessing(df)
        
        # 3. 예측용 카테고리 처리 (unknown 처리)
        df = self._process_categorical_prediction(df)
        
        # 4. 새로운 특성 생성
        df = self._create_derived_features(df)
        
        # 5. 스케일링 (변환만)
        df = self._transform_scaling(df)
        
        logger.info(f"✅ 예측 데이터 전처리 완료: {len(df)}개 레코드")
        
        return df
    
    def _calculate_horse_features(self, df: pd.DataFrame, is_prediction: bool = False) -> pd.DataFrame:
        """
        말별 과거 성적 특성 계산
        
        Args:
            df: 데이터프레임
            is_prediction: 예측용 데이터인지 여부
            
        Returns:
            특성이 추가된 데이터프레임
        """
        logger.info("🐎 말별 특성 계산 중...")
        
        if is_prediction:
            # 예측용: 각 말의 과거 성적을 별도로 조회해야 함
            return self._calculate_prediction_features(df)
        else:
            # 학습용: 시간 순서대로 정렬하여 과거 성적 계산
            return self._calculate_training_features(df)
    
    def _calculate_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습용 특성 계산 (시간 순서 기반)"""
        # 날짜순 정렬
        df = df.sort_values(['horse_id', 'race_date']).copy()
        
        def calculate_horse_stats(group):
            """말별 누적 통계 계산"""
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
        
        # 말별로 그룹화하여 특성 계산
        df = df.groupby('horse_id', group_keys=False).apply(calculate_horse_stats)
        
        # 최소 경험 필터링
        min_races = DATA_EXTRACTION.get('min_horse_races', 3)
        initial_len = len(df)
        df = df[df['prev_total_races'] >= min_races]
        
        logger.info(f"   최소 경주 필터링: {initial_len} → {len(df)} ({len(df)/initial_len*100:.1f}%)")
        
        return df.reset_index(drop=True)
    
    def _calculate_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """예측용 특성 계산 (과거 데이터 조회 필요)"""
        # 실제 구현에서는 database에서 각 말의 과거 성적을 조회해야 함
        # 여기서는 기본값으로 설정
        logger.warning("⚠️ 예측용 특성 계산은 별도 구현이 필요합니다.")
        
        # 기본값 설정
        df['prev_total_races'] = 5  # 기본값
        df['prev_wins'] = 1
        df['prev_top3'] = 2
        df['prev_5_avg_rank'] = 4.0
        df['prev_total_avg_rank'] = 4.5
        
        return df
    
    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리 (결측치 처리 등)"""
        logger.info("🔧 기본 전처리 중...")
        
        # 숫자형 컬럼 결측치 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 1등 여부 컬럼 생성
        if 'final_rank' in df.columns:
            df['is_winner'] = (df['final_rank'] == 1).astype(int)
        
        return df
    
    def _process_categorical_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습용 카테고리 데이터 처리"""
        logger.info("📚 학습용 카테고리 처리 중...")
        
        categorical_cols = self.config['categorical_features']
        
        # 1. 결측값이 있는 행 제거
        for col in categorical_cols:
            if col in df.columns:
                before_len = len(df)
                df = df.dropna(subset=[col])
                after_len = len(df)
                if before_len != after_len:
                    logger.info(f"   {col} 결측값 {before_len - after_len}개 행 제거")
        
        # 2. LabelEncoder 학습
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
                # 'unknown' 클래스 추가 (예측 시 새로운 값 처리용)
                unique_values = list(df[col].unique())
                if 'unknown' not in unique_values:
                    unique_values.append('unknown')
                
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(unique_values)
                
                # 실제 데이터 변환
                df[col] = self.label_encoders[col].transform(df[col])
                
                logger.info(f"   {col} 인코딩 완료: {len(self.label_encoders[col].classes_)}개 클래스")
        
        return df
    
    def _process_categorical_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """예측용 카테고리 데이터 처리"""
        logger.info("🔮 예측용 카테고리 처리 중...")
        
        categorical_cols = self.config['categorical_features']
        
        for col in categorical_cols:
            if col in df.columns:
                # 1. 결측값을 'unknown'으로 처리
                df[col] = df[col].fillna('unknown').astype(str)
                
                # 2. 학습된 LabelEncoder로 변환
                if col in self.label_encoders:
                    df[col] = self._safe_transform_with_unknown(df[col], col)
                else:
                    logger.warning(f"⚠️ {col}에 대한 LabelEncoder가 없습니다!")
                    df[col] = 0  # 기본값으로 처리
        
        return df
    
    def _safe_transform_with_unknown(self, series: pd.Series, column_name: str) -> pd.Series:
        """새로운 값을 unknown으로 안전하게 변환"""
        encoder = self.label_encoders[column_name]
        known_classes = set(encoder.classes_)
        
        # 새로운 값들 찾기
        current_values = set(series.unique())
        unseen_values = current_values - known_classes
        
        if unseen_values:
            logger.info(f"   ⚠️ {column_name}에서 새로운 값 발견: {unseen_values}")
            
            # 새로운 값들을 unknown으로 대체
            series_copy = series.copy()
            for unseen_val in unseen_values:
                series_copy = series_copy.replace(unseen_val, 'unknown')
            
            # unknown도 학습된 클래스에 없다면
            if 'unknown' not in known_classes:
                most_common = encoder.classes_[0]
                series_copy = series_copy.replace('unknown', most_common)
                logger.info(f"   unknown을 {most_common}으로 대체")
        else:
            series_copy = series
        
        return pd.Series(encoder.transform(series_copy), index=series.index)
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 특성 생성"""
        logger.info("⚡ 파생 특성 생성 중...")
        
        # 승률 계산
        df['jockey_win_rate'] = df['jockey_total_wins'] / (df['jockey_total_races'] + 1)
        df['trainer_win_rate'] = df['trainer_total_wins'] / (df['trainer_total_races'] + 1)
        df['horse_win_rate'] = df['prev_wins'] / (df['prev_total_races'] + 1)
        df['horse_top3_rate'] = df['prev_top3'] / (df['prev_total_races'] + 1)
        
        # 경험 점수
        df['experience_score'] = np.log1p(df['prev_total_races'])
        
        # 최근 폼
        df['recent_form'] = 6 - df['prev_5_avg_rank'].fillna(6)
        
        # 인기도 점수 (없으면 기본값)
        if 'popularity_score' not in df.columns:
            df['popularity_score'] = 5.0  # 중간값
        
        logger.info(f"   생성된 파생 특성: {len(self.config['derived_features'])}개")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """이상치 제거"""
        logger.info("🚫 이상치 제거 중...")
        
        initial_len = len(df)
        
        # 순위 이상치 제거
        if 'final_rank' in df.columns:
            df = df[df['final_rank'] <= DATA_EXTRACTION.get('max_horses_per_race', 20)]
        
        # 경주 참가 말 수 이상치 제거
        if 'total_horses' in df.columns:
            df = df[df['total_horses'] >= DATA_EXTRACTION.get('min_horses_per_race', 5)]
        
        # 말 나이 이상치 제거
        if 'horse_age' in df.columns:
            df = df[(df['horse_age'] >= 2) & (df['horse_age'] <= 10)]
        
        removed_count = initial_len - len(df)
        logger.info(f"   이상치 제거: {removed_count}개 ({removed_count/initial_len*100:.1f}%)")
        
        return df
    
    def _set_feature_columns(self, df: pd.DataFrame):
        """사용할 특성 컬럼 설정"""
        all_features = (
            self.config['numeric_features'] + 
            self.config['categorical_features'] + 
            self.config['derived_features']
        )
        
        # 실제 존재하는 컬럼만 선택
        self.feature_columns = [col for col in all_features if col in df.columns]
        
        logger.info(f"📋 선택된 특성: {len(self.feature_columns)}개")
        for feature in self.feature_columns:
            logger.debug(f"   - {feature}")
    
    def _fit_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """스케일링 학습 및 변환"""
        if not self.feature_columns:
            return df
        
        logger.info("📏 스케일링 학습 중...")
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        
        # 스케일된 값으로 대체
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = X_scaled
        
        return df_scaled
    
    def _transform_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """스케일링 변환 (학습된 scaler 사용)"""
        if not self.feature_columns or self.scaler is None:
            return df
        
        logger.info("📏 스케일링 변환 중...")
        
        # 누락된 특성 컬럼을 기본값으로 채우기
        for col in self.feature_columns:
            if col not in df.columns:
                logger.warning(f"⚠️ 특성 {col}이 없어서 기본값으로 설정")
                df[col] = 0
        
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # 스케일된 값으로 대체
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = X_scaled
        
        return df_scaled
    
    def get_feature_importance_names(self) -> List[str]:
        """특성 중요도 출력용 이름 반환"""
        return self.feature_columns.copy()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검사"""
        if df.empty:
            logger.error("❌ 빈 데이터프레임입니다.")
            return False
        
        required_cols = ['horse_id', 'race_date', 'final_rank']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ 필수 컬럼 누락: {missing_cols}")
            return False
        
        logger.info("✅ 데이터 유효성 검사 통과")
        return True
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """전처리 요약 정보 반환"""
        return {
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'categorical_encoders': list(self.label_encoders.keys()),
            'has_scaler': self.scaler is not None
        }