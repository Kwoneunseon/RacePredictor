
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from _const import API_KEY, SUPABASE_URL, SUPABASE_KEY
#from predictor import HorseRacing1stPlacePredictor
from predictor_new import HorseRacing1stPlacePredictor
from datetime import datetime, timedelta

# 사용 예시
def main():
    predictor = HorseRacing1stPlacePredictor(SUPABASE_URL, SUPABASE_KEY)
    
    # 1. 기존 모델 로드 시도
    print("\n" + "=" * 50)
    print("🔄 기존 모델 로드 시도")
    print("=" * 50)
    
    model_loaded = predictor.get_loaded_model('precision_boosted_model_250824')

    if model_loaded:
        print("✅ 기존 모델을 성공적으로 로드했습n니다!")
         
        # 모델 재학습 여부 확인
        retrain = input("\n🤔 모델을 다시 학습하시겠습니까? (y/n): ").lower().strip()
        if retrain == 'y':
            model_loaded = False
    
    if not model_loaded:
        print("\n" + "=" * 50)
        print("🏇 경마 1등 예측 모델 훈련")
        print("=" * 50)

        yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')        
        df = predictor.extract_training_data_batch('2023-01-01', yesterday, batch_months=1)
                
        if len(df) > 0:
            predictor.precision_boost_training(df, test_size=0.2, model_name='precision_boosted_model_250824')            
        else:
            print("❌ 훈련 데이터가 충분하지 않습니다. 모델 훈련을 중단합니다.")
            return  
        

        
    # 2. 특정 경주 예측
    # print("\n" + "=" * 50)
    # print("🔮 경주 예측 테스트")
    # print("=" * 50)
        
    # prediction = predictor.predict_race_winners('2024-07-28', '서울', 13)
    # print(prediction)
    
    # #3. 백테스팅
    # print("\n" + "=" * 50)
    # print("📈 백테스팅 테스트")
    # print("=" * 50)
    
    # today = datetime.today().strftime('%Y-%m-%d')
    # backtest_result = predictor.backtest_strategy('2025-05-01', today, 0.6)       

    #4. 실제 예측 결과 
    print("\n" + "=" * 50)
    print("🏁 실제 경주 예측")
    print("=" * 50) 

    # 오늘 날짜를 파라미터로 전달
    today = datetime.today().strftime('%Y-%m-%d')
 
    predictor.predict_race_winners('2025-08-29')
    predictor.predict_race_winners('2025-08-30')
    predictor.predict_race_winners('2025-08-31')
 
 

if __name__ == "__main__":
    main()