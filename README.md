# 🏇 경마 1등 예측 모델 실사용 가이드

## 📋 개요

이 시스템은 Supabase를 기반으로 한 경마 1등 예측 AI 모델입니다. 과거 경주 데이터를 분석하여 말, 기수, 조교사의 성적과 경주 조건을 종합적으로 고려해 1등 확률을 예측합니다.

## 🎯 핵심 알고리즘

### 1. 특성 추출
- **말의 과거 성적**: 최근 5경주 평균 순위, 전체 평균 순위, 승수, 3위 이내 횟수
- **거리별 적성**: 해당 거리에서의 평균 순위와 출전 횟수
- **기수/조교사 실력**: 통산 승률, 최근 1년 승률
- **경주 조건**: 거리, 출주 두수, 날씨, 경주로 상태
- **인기도**: 배당률 기반 대중의 기대치

### 2. 모델 앙상블
- **Random Forest**: 비선형 패턴 학습, 특성 중요도 제공
- **Gradient Boosting**: 순차적 오류 보정, 높은 예측 정확도
- **Logistic Regression**: 해석 가능한 선형 관계 모델

### 3. 예측 점수 계산
```
최종 확률 = (RF 예측 + GB 예측 + LR 예측) / 3
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install supabase pandas scikit-learn numpy matplotlib seaborn python-dotenv

# 환경변수 설정 (.env 파일)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
```

### 2. 기본 사용법

```python
from horse_racing_prediction_v2 import HorseRacing1stPlacePredictor
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 모델 초기화
predictor = HorseRacing1stPlacePredictor(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

# 1. 모델 훈련 (최소 6개월 데이터 권장)
print("🤖 모델 훈련 중...")
df = predictor.extract_training_data('2023-01-01', '2024-11-30')
results = predictor.train_models(df, test_size=0.2)

# 2. 특정 경주 예측
print("🔮 경주 예측...")
prediction = predictor.predict_race_winners('2024-12-15', 'SEL', 1)
print(prediction)

# 3. 백테스팅
print("📈 백테스팅...")
backtest = predictor.backtest_strategy('2024-12-01', '2024-12-15')
```

### 3. 예측 결과 해석

```python
# 예측 결과 예시
   horse_name  entry_number  win_odds  win_probability  prediction_rank
0       번개            1       2.5           0.752               1.0
1       천둥            3       4.2           0.618               2.0
2       바람            2       6.8           0.487               3.0
```

- **win_probability**: AI가 예측한 1등 확률 (0~1)
- **prediction_rank**: 예측 순위 (1위가 가장 유력)
- **win_odds**: 배당률 (낮을수록 인기마)

## 📊 성능 지표

### 1. 모델 정확도
- **정확도**: 예측한 1등마가 실제로 1등할 확률
- **정밀도**: 1등으로 예측한 말 중 실제 1등 비율
- **재현율**: 실제 1등마를 올바르게 예측한 비율
- **F1-Score**: 정밀도와 재현율의 조화평균
- **AUC**: ROC 곡선 아래 면적 (0.5~1.0, 높을수록 좋음)

### 2. 베팅 성과
- **적중률**: 베팅한 말이 1등한 비율
- **ROI**: 투자 대비 수익률 (Return on Investment)
- **최대 연속 손실**: 리스크 관리용 지표
- **샤프 비율**: 위험 대비 수익률

## 💡 베팅 전략

### 1. 확신도 기반 전략
```python
# 높은 확신도 (70% 이상)만 베팅
if prediction['win_probability'].max() >= 0.7:
    bet_horse = prediction.iloc[0]
    print(f"베팅 추천: {bet_horse['horse_name']} (확률: {bet_horse['win_probability']:.1%})")
```

### 2. 가치 베팅 전략
```python
# 예측 확률 > 암시 확률인 말에 베팅
prediction['implied_prob'] = 1 / prediction['win_odds']
value_bets = prediction[prediction['win_probability'] > prediction['implied_prob']]

if len(value_bets) > 0:
    best_value = value_bets.iloc[0]
    print(f"가치 베팅: {best_value['horse_name']}")
```

### 3. 리스크 관리
- **베팅 금액 제한**: 총 자금의 2-5% 이내
- **연속 손실 제한**: 5회 연속 손실시 중단
- **일일 베팅 제한**: 하루 최대 3-5경주만

## 🔧 모델 개선 방법

### 1. 데이터 품질 향상
```sql
-- 데이터 정합성 체크
SELECT 
    COUNT(*) as total_races,
    COUNT(DISTINCT horse_id) as unique_horses,
    AVG(total_horses) as avg_horses_per_race
FROM race_entries 
WHERE race_date >= '2024-01-01';
```

### 2. 특성 엔지니어링
```python
# 새로운 특성 추가
df['pace_score'] = df['finish_time'] / df['race_distance']  # 속도 지표
df['weight_ratio'] = df['horse_weight'] / df['race_distance']  # 체중 대비 거리
df['form_trend'] = df['prev_5_avg_rank'] - df['prev_total_avg_rank']  # 최근 폼 변화
```

### 3. 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import GridSearchCV

# Random Forest 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## ⚠️ 주의사항

### 1. 데이터 품질
- **최소 데이터**: 6개월 이상의 경주 데이터
- **데이터 일관성**: 누락된 배당률이나 순위 정보 확인
- **최신성**: 정기적인 데이터 업데이트 필요

### 2. 과적합 방지
- **교차 검증**: 시간 순서를 고려한 분할
- **특성 선택**: 중요도가 낮은 특성 제거
- **정규화**: 모델 복잡도 제어

### 3. 현실적 기대치
- **적중률**: 일반적으로 15-25% (경마는 본질적으로 불확실성이 높음)
- **ROI**: 장기적으로 5-15% 달성시 매우 우수
- **변동성**: 단기적으로 큰 손실 가능, 장기 관점 필요

### 4. 법적/윤리적 고려사항
- **합법적 베팅**: 공인된 경마장에서만 베팅
- **도박 중독 방지**: 적절한 한도 설정
- **데이터 사용**: 공개된 데이터만 사용

## 📈 실전 운영 예시

### 일일 운영 루틴

```python
def daily_prediction_routine(race_date):
    """일일 예측 루틴"""
    
    # 1. 당일 경주 목록 조회
    races = predictor.supabase.table('races')\
        .select('*')\
        .eq('race_date', race_date)\
        .execute()
    
    recommendations = []
    
    for race in races.data:
        try:
            # 2. 각 경주 예측
            prediction = predictor.predict_race_winners(
                race['race_date'], 
                race['meet_code'], 
                race['race_no']
            )
            
            if isinstance(prediction, pd.DataFrame):
                best_horse = prediction.iloc[0]
                
                # 3. 베팅 기준 확인
                if (best_horse['win_probability'] >= 0.6 and 
                    best_horse['win_odds'] >= 2.0):
                    
                    recommendations.append({
                        'race_no': race['race_no'],
                        'horse_name': best_horse['horse_name'],
                        'probability': best_horse['win_probability'],
                        'odds': best_horse['win_odds'],
                        'confidence': 'HIGH' if best_horse['win_probability'] >= 0.7 else 'MEDIUM'
                    })
                    
        except Exception as e:
            print(f"R{race['race_no']} 예측 실패: {e}")
    
    return recommendations

# 사용 예시
today_picks = daily_prediction_routine('2024-12-15')
for pick in today_picks:
    print(f"R{pick['race_no']}: {pick['horse_name']} "
          f"(확률: {pick['probability']:.1%}, 배당: {pick['odds']:.1f}, "
          f"신뢰도: {pick['confidence']})")
```

### 주간 성과 분석

```python
def weekly_performance_review(start_date, end_date):
    """주간 성과 리뷰"""
    
    # 실제 베팅 기록 (별도 저장 필요)
    bets = load_betting_records(start_date, end_date)
    
    total_bets = len(bets)
    wins = sum(1 for bet in bets if bet['result'] == 'WIN')
    total_stake = sum(bet['amount'] for bet in bets)
    total_return = sum(bet['return'] for bet in bets)
    
    win_rate = wins / total_bets if total_bets > 0 else 0
    roi = ((total_return - total_stake) / total_stake * 100) if total_stake > 0 else 0
    
    print(f"📊 {start_date} ~ {end_date} 성과")
    print(f"총 베팅: {total_bets}회")
    print(f"적중: {wins}회 ({win_rate:.1%})")
    print(f"투자금: {total_stake:,}원")
    print(f"수익: {total_return - total_stake:,}원")
    print(f"ROI: {roi:+.1f}%")
    
    return {
        'period': f"{start_date}~{end_date}",
        'total_bets': total_bets,
        'wins': wins,
        'win_rate': win_rate,
        'roi': roi
    }
```

## 🛠️ 트러블슈팅

### 자주 발생하는 문제들

#### 1. 데이터 연결 오류
```python
# 연결 테스트
try:
    test = predictor.supabase.table('horses').select("horse_id").limit(1).execute()
    print("✅ Supabase 연결 정상")
except Exception as e:
    print(f"❌ 연결 오류: {e}")
    # API 키, URL 재확인 필요
```

#### 2. 모델 성능 저하
```python
# 정기적 모델 재훈련 (월 1회 권장)
def retrain_model():
    print("🔄 모델 재훈련 중...")
    
    # 최근 6개월 데이터로 재훈련
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    
    df = predictor.extract_training_data(start_date, end_date)
    results = predictor.train_models(df)
    
    # 성능 확인
    if results['ensemble_auc'] > 0.6:
        print("✅ 재훈련 완료 - 성능 양호")
        save_model(predictor)  # 모델 저장
    else:
        print("⚠️ 재훈련 완료 - 성능 점검 필요")
```

#### 3. 예측 오류
```python
def safe_prediction(race_date, meet_code, race_no):
    """안전한 예측 함수"""
    try:
        prediction = predictor.predict_race_winners(race_date, meet_code, race_no)
        
        if isinstance(prediction, str):
            return {"error": prediction}
        
        if len(prediction) == 0:
            return {"error": "예측 결과 없음"}
        
        return {"success": True, "prediction": prediction}
        
    except Exception as e:
        return {"error": f"예측 실패: {str(e)}"}

# 사용
result = safe_prediction('2024-12-15', 'SEL', 1)
if "error" in result:
    print(f"❌ {result['error']}")
else:
    print("✅ 예측 성공")
```

## 📝 베스트 프랙티스

### 1. 데이터 관리
- **정기 백업**: 주요 예측 결과와 성과 기록
- **버전 관리**: 모델 변경 이력 추적
- **데이터 검증**: 입력 데이터 품질 체크

### 2. 리스크 관리
- **포지션 사이징**: 켈리 공식 등 활용
- **다양화**: 여러 경주장, 거리, 등급 분산
- **손절매**: 명확한 중단 기준 설정

### 3. 지속적 개선
- **A/B 테스트**: 새로운 전략 검증
- **피드백 루프**: 결과 분석을 통한 모델 개선
- **도메인 지식**: 경마 전문가 의견 수렴

## 🎯 성공 사례

### 실제 운영 결과 (예시)
```
기간: 2024년 3분기 (3개월)
총 베팅: 127회
적중: 31회 (24.4%)
투자금: 127만원
수익: +18만원 (ROI: +14.2%)

주요 성공 요인:
1. 높은 확신도 경주만 선별
2. 가치 베팅 전략 병행
3. 엄격한 자금 관리
```

## 📞 지원 및 문의

### 기술적 문제
- GitHub Issues: [링크]
- 이메일: support@horseracing-ai.com
- 문서: [API 문서 링크]

### 커뮤니티
- 디스코드: [커뮤니티 링크]
- 블로그: [업데이트 소식]
- 유튜브: [사용법 영상]

---

## 📚 추가 자료

### 참고 논문
- "Machine Learning in Sports Betting" (2023)
- "Ensemble Methods for Horse Racing Prediction" (2022)
- "Feature Engineering in Sports Analytics" (2023)

### 관련 도구
- **데이터 시각화**: Plotly, Seaborn
- **백테스팅**: Backtrader, Zipline
- **모니터링**: MLflow, Weights & Biases

### 경마 도메인 지식
- **말의 특성**: 혈통, 조교 방법, 컨디션
- **경주 조건**: 거리별 특성, 날씨 영향, 경주로 상태
- **베팅 시장**: 배당률 형성 원리, 마권 종류별 특징

---

*"성공적인 경마 예측은 기술과 경험, 그리고 절제의 조화입니다."*

**⚠️ 면책조항**: 이 모델은 교육 및 연구 목적으로 제작되었습니다. 실제 베팅에 따른 손실에 대해서는 책임지지 않습니다. 항상 책임감 있는 베팅을 하시기 바랍니다.