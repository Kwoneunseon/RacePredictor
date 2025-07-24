"""
모델 관리 모듈 - 모델 저장/로드/버전 관리
"""
import os
import json
import joblib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import MODEL_DIR, MODEL_MANAGEMENT

logger = logging.getLogger(__name__)


class ModelManager:
    """모델 저장/로드/관리 클래스"""
    
    def __init__(self, model_save_path:str = None):
        """
        Args:
            model_dir: 모델 저장 디렉토리 (None이면 config 사용)
        """
        self.model_save_path = model_save_path if model_save_path else MODEL_DIR
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
    
    def save_model(self, model_name="horse_racing_model", model_data:Dict = None):
        """
        훈련된 모델과 전처리 객체들을 저장
        
        Args:
            model_name: 저장할 모델 이름
        """
        try:            
            model_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
            joblib.dump(model_data, model_path)
            
            print(f"💾 모델 저장 완료: {model_path}")
            print(f"   - 모델 개수: {len(model_data)}")
            print(f"   - 특성 개수: {len(model_data)}")
            print(f"   - 저장 시간: {model_data['save_date']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            return False

    def load_model(self, model_name="horse_racing_model"):
        """
        저장된 모델과 전처리 객체들을 로드
        
        Args:
            model_name: 로드할 모델 이름
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            model_path = os.path.join(self.model_save_path, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                print(f"⚠️ 저장된 모델이 없습니다: {model_path}")
                return False
            
            model_data = joblib.load(model_path)

            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']          

            
            print(f"📂 모델 로드 완료: {model_path}")
            print(f"   - 모델 개수: {len(self.models)}")
            print(f"   - 특성 개수: {len(self.feature_columns)}")
            print(f"   - 저장 시간: {model_data.get('save_date', '알 수 없음')}")
            
            # 모델 정보 출력
            print(f"   - 로드된 모델들:")
            for name, result in self.models.items():
                print(f"     * {name}: AUC={result.get('auc', 0):.3f}")
            
            return model_data
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return None

    def list_saved_models(self):
        """
        저장된 모델 목록 확인
        """
        try:
            model_files = [f for f in os.listdir(self.model_save_path) if f.endswith('.pkl')]
            
            if not model_files:
                print("📁 저장된 모델이 없습니다.")
                return []
            
            print(f"📁 저장된 모델 목록 ({self.model_save_path}):")
            model_info = []
            
            for model_file in model_files:
                model_path = os.path.join(self.model_save_path, model_file)
                try:
                    # 파일 정보
                    file_stat = os.stat(model_path)
                    file_size = file_stat.st_size / (1024 * 1024)  # MB
                    modified_time = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    model_info.append({
                        'file': model_file,
                        'size_mb': file_size,
                        'modified': modified_time
                    })
                    
                    print(f"  📄 {model_file}")
                    print(f"     크기: {file_size:.2f}MB")
                    print(f"     수정일: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    print(f"     ⚠️ 파일 정보 읽기 실패: {e}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ 모델 목록 조회 실패: {e}")
            return []

    def check_model_performance(self):
        """
        현재 로드된 모델의 성능 확인
        """
        if not self.models:
            print("⚠️ 로드된 모델이 없습니다.")
            return
        
        print("📊 현재 모델 성능:")
        print("-" * 50)
        
        for name, result in self.models.items():
            print(f"🔥 {name}:")
            print(f"  정확도: {result.get('accuracy', 0):.3f}")
            print(f"  정밀도: {result.get('precision', 0):.3f}")
            print(f"  재현율: {result.get('recall', 0):.3f}")
            print(f"  F1: {result.get('f1', 0):.3f}")
            print(f"  AUC: {result.get('auc', 0):.3f}")
            print()