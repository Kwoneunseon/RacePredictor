"""
개선된 모델 관리 모듈 - sklearn 버전 호환성 문제 해결
"""
import os
import json
import joblib
import pickle
import shutil
import sklearn
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings

# sklearn 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import MODEL_DIR, MODEL_MANAGEMENT

logger = logging.getLogger(__name__)


class ModelManager:
    """모델 저장/로드/관리 클래스 (sklearn 호환성 개선)"""
    
    def __init__(self, model_save_path: str = None):
        """
        Args:
            model_save_path: 모델 저장 디렉토리 (None이면 config 사용)
        """
        self.model_save_path = model_save_path if model_save_path else MODEL_DIR
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
    
    def save_model_safe(self, model_name="horse_racing_model", model_data: Dict = None):
        """
        안전한 모델 저장 (sklearn 호환성 개선)
        """
        try:
            if model_data is None:
                print("❌ 저장할 모델 데이터가 없습니다.")
                return False
            
            # 현재 환경 정보
            env_info = {
                'sklearn_version': sklearn.__version__,
                'numpy_version': np.__version__,
                'save_timestamp': datetime.now().isoformat(),
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            }
            
            # 별도로 환경 정보 저장
            env_path = os.path.join(self.model_save_path, f"{model_name}_env.json")
            with open(env_path, 'w') as f:
                json.dump(env_info, f, indent=2)
            
            # 모델 데이터에 환경 정보 추가
            model_data_enhanced = model_data.copy()
            model_data_enhanced.update(env_info)
            
            success_count = 0
            
            # Method 1: joblib 저장 (sklearn_version 제한)
            try:
                joblib_path = os.path.join(self.model_save_path, f"{model_name}_joblib.pkl")
                # sklearn 객체 저장 시 호환성 모드 사용
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    joblib.dump(model_data_enhanced, joblib_path, compress=3)
                print(f"✅ joblib 저장 성공: {joblib_path}")
                success_count += 1
            except Exception as e:
                print(f"⚠️ joblib 저장 실패: {e}")
            
            # Method 2: 안전한 pickle 저장
            try:
                pickle_path = os.path.join(self.model_save_path, f"{model_name}_pickle.pkl")
                with open(pickle_path, 'wb') as f:
                    # 최고 호환성 프로토콜 사용
                    pickle.dump(model_data_enhanced, f, protocol=4)  # python 3.4+ 호환
                print(f"✅ pickle 저장 성공: {pickle_path}")
                success_count += 1
            except Exception as e:
                print(f"⚠️ pickle 저장 실패: {e}")
                
            # Method 3: 모델별 개별 저장 (호환성 최대화)
            try:
                individual_dir = os.path.join(self.model_save_path, f"{model_name}_individual")
                os.makedirs(individual_dir, exist_ok=True)
                
                # 각 구성 요소 개별 저장
                if 'models' in model_data:
                    for name, model_result in model_data['models'].items():
                        model_path = os.path.join(individual_dir, f"model_{name}.pkl")
                        joblib.dump(model_result['model'], model_path)
                
                # 전처리 객체들 개별 저장
                if 'scaler' in model_data:
                    scaler_path = os.path.join(individual_dir, "scaler.pkl")
                    joblib.dump(model_data['scaler'], scaler_path)
                
                if 'label_encoders' in model_data:
                    le_path = os.path.join(individual_dir, "label_encoders.pkl")
                    joblib.dump(model_data['label_encoders'], le_path)
                
                # 메타데이터 JSON 저장
                meta_data = {k: v for k, v in model_data.items() 
                           if k not in ['models', 'scaler', 'label_encoders']}
                meta_data.update(env_info)
                
                meta_path = os.path.join(individual_dir, "metadata.json")
                with open(meta_path, 'w') as f:
                    json.dump(meta_data, f, indent=2, default=str)
                
                print(f"✅ 개별 저장 성공: {individual_dir}")
                success_count += 1
                
            except Exception as e:
                print(f"⚠️ 개별 저장 실패: {e}")
            
            if success_count > 0:
                print(f"💾 모델 저장 완료: {model_name} ({success_count}개 방식)")
                print(f"   - sklearn 버전: {env_info['sklearn_version']}")
                return True
            else:
                print(f"❌ 모든 저장 방식 실패")
                return False
                
        except Exception as e:
            print(f"❌ 모델 저장 완전 실패: {e}")
            return False

    def load_model_safe(self, model_name="horse_racing_model"):
        """
        안전한 모델 로드 (sklearn 호환성 개선)
        """
        # 환경 정보 먼저 확인
        env_path = os.path.join(self.model_save_path, f"{model_name}_env.json")
        saved_env = None
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    saved_env = json.load(f)
                print(f"📋 저장된 환경 정보:")
                print(f"   - sklearn: {saved_env.get('sklearn_version', '알 수 없음')}")
                print(f"   - 현재 sklearn: {sklearn.__version__}")
            except:
                pass
        
        load_attempts = [
            # 안전한 순서로 시도
            ("individual", self._load_individual_model, "개별 구성요소"),
            (f"{model_name}_pickle.pkl", self._load_with_pickle_safe, "안전한 pickle"),
            (f"{model_name}_joblib.pkl", self._load_with_joblib_safe, "안전한 joblib"),
            (f"{model_name}.pkl", self._load_with_joblib_safe, "레거시")
        ]
        
        for identifier, load_func, method_name in load_attempts:
            print(f"🔄 {method_name} 방식으로 로드 시도...")
            
            try:
                model_data = load_func(model_name if identifier == "individual" else 
                                     os.path.join(self.model_save_path, identifier))
                
                if model_data and self._validate_model_data(model_data):
                    self._apply_model_data(model_data)
                    
                    print(f"✅ 모델 로드 성공: {method_name}")
                    print(f"   - 모델 개수: {len(model_data.get('models', {}))}")
                    print(f"   - 특성 개수: {len(model_data.get('feature_columns', []))}")
                    
                    return model_data
                    
            except Exception as e:
                print(f"⚠️ {method_name} 로드 실패: {str(e)[:100]}...")
                continue
        
        print(f"❌ 모든 로드 방식 실패: {model_name}")
        print("\n💡 해결 방법:")
        print("1. 모델을 다시 훈련시키세요 (가장 확실한 방법)")
        print("2. sklearn 버전을 맞춰보세요")
        if saved_env:
            print(f"   저장시 버전: {saved_env.get('sklearn_version')}")
            print(f"   현재 버전: {sklearn.__version__}")
        return None
    
    def _load_individual_model(self, model_name):
        """개별 구성요소로 저장된 모델 로드"""
        individual_dir = os.path.join(self.model_save_path, f"{model_name}_individual")
        
        if not os.path.exists(individual_dir):
            raise FileNotFoundError("개별 저장 디렉토리가 없습니다")
        
        # 메타데이터 로드
        meta_path = os.path.join(individual_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("메타데이터 파일이 없습니다")
        
        with open(meta_path, 'r') as f:
            model_data = json.load(f)
        
        # 전처리 객체들 로드
        scaler_path = os.path.join(individual_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            model_data['scaler'] = joblib.load(scaler_path)
        
        le_path = os.path.join(individual_dir, "label_encoders.pkl")
        if os.path.exists(le_path):
            model_data['label_encoders'] = joblib.load(le_path)
        
        # 모델들 로드
        models = {}
        for file in os.listdir(individual_dir):
            if file.startswith("model_") and file.endswith(".pkl"):
                model_name_part = file.replace("model_", "").replace(".pkl", "")
                model_path = os.path.join(individual_dir, file)
                
                try:
                    model = joblib.load(model_path)
                    models[model_name_part] = {'model': model}
                except Exception as e:
                    print(f"⚠️ 모델 {model_name_part} 로드 실패: {e}")
        
        if models:
            model_data['models'] = models
            return model_data
        else:
            raise ValueError("로드 가능한 모델이 없습니다")
    
    def _load_with_joblib_safe(self, model_path):
        """안전한 joblib 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"파일이 없습니다: {model_path}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(model_path)
    
    def _load_with_pickle_safe(self, model_path):
        """안전한 pickle 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"파일이 없습니다: {model_path}")
            
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _validate_model_data(self, model_data):
        """모델 데이터 유효성 검사"""
        required_keys = ['models', 'scaler', 'label_encoders', 'feature_columns']
        
        for key in required_keys:
            if key not in model_data:
                print(f"⚠️ 필수 키 누락: {key}")
                return False
        
        if not isinstance(model_data['models'], dict) or len(model_data['models']) == 0:
            print("⚠️ 유효한 모델이 없습니다.")
            return False
            
        return True
    
    def _apply_model_data(self, model_data):
        """모델 데이터를 현재 인스턴스에 적용"""
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.best_threshold = model_data.get('best_threshold', 0.5)

    def retrain_and_save(self, train_func, model_name="horse_racing_model", **kwargs):
        """
        모델을 다시 훈련시키고 저장 (호환성 문제 해결용)
        
        Args:
            train_func: 훈련 함수
            model_name: 모델 이름
            **kwargs: 훈련 함수에 전달할 인자들
        """
        print("🔄 호환성 문제로 인한 모델 재훈련을 시작합니다...")
        
        try:
            # 기존 모델 백업
            backup_dir = os.path.join(self.model_save_path, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for file in os.listdir(self.model_save_path):
                if model_name in file and file.endswith('.pkl'):
                    src = os.path.join(self.model_save_path, file)
                    dst = os.path.join(backup_dir, f"{timestamp}_{file}")
                    shutil.copy2(src, dst)
                    print(f"📦 백업: {file} -> {dst}")
            
            # 재훈련 실행
            print("🏃‍♂️ 모델 재훈련 중...")
            model_data = train_func(**kwargs)
            
            if model_data:
                # 새로운 환경으로 저장
                success = self.save_model_safe(model_name, model_data)
                if success:
                    print("✅ 재훈련 및 저장 완료!")
                    return True
                else:
                    print("❌ 재훈련 후 저장 실패")
                    return False
            else:
                print("❌ 재훈련 실패")
                return False
                
        except Exception as e:
            print(f"❌ 재훈련 과정 실패: {e}")
            return False

    # 기존 메서드들...
    def save_model(self, model_name="horse_racing_model", model_data: Dict = None):
        """기존 호환성을 위한 래퍼"""
        return self.save_model_safe(model_name, model_data)
    
    def load_model(self, model_name="horse_racing_model"):
        """기존 호환성을 위한 래퍼"""
        return self.load_model_safe(model_name)