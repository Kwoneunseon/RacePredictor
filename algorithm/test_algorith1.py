import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HorseRacingTestSuite:
    def __init__(self, predictor):
        """
        테스트 스위트 초기화
        
        Args:
            predictor: HorseRacing1stPlacePredictor 인스턴스
        """
        self.predictor = predictor
        self.test_results = {}
    
    def comprehensive_backtest(self, start_date, end_date, strategies=None):
        """
        포괄적인 백테스팅
        """
        if strategies is None:
            strategies = [
                {'name': '높은 확신도', 'threshold': 0.7, 'bet_amount': 1000},
                {'name': '중간 확신도', 'threshold': 0.5, 'bet_amount': 1000},
                {'name': '낮은 확신도', 'threshold': 0.3, 'bet_amount': 1000},
                {'name': '최고 인기마', 'threshold': 0.0, 'bet_amount': 1000, 'type': 'favorite'},
                {'name': '가치베팅', 'threshold': 0.4, 'bet_amount': 1000, 'type': 'value'}
            ]
        
        print(f"🧪 포괄적 백테스팅: {start_date} ~ {end_date}")
        
        # 테스트 기간의 모든 경주 조회
        races = self.predictor.supabase.table('races')\
            .select('race_date, meet_code, race_no, race_id')\
            .gte('race_date', start_date)\
            .lte('race_date', end_date)\
            .order('race_date')\
            .execute()
        
        strategy_results = {strategy['name']: {
            'bets': [], 'wins': 0, 'total_bet': 0, 'total_return': 0, 'roi': 0
        } for strategy in strategies}
        
        total_races = len(races.data)
        processed = 0
        
        for race in races.data:
            processed += 1
            if processed % 50 == 0:
                print(f"진행률: {processed}/{total_races} ({processed/total_races*100:.1f}%)")
            
            try:
                # 경주 예측
                predictions = self.predictor.predict_race_winners(
                    race['race_date'], 
                    race['meet_code'], 
                    race['race_no']
                )
                
                if isinstance(predictions, str):
                    continue
                
                # 실제 결과 조회
                actual_results = self.predictor.supabase.table('race_entries')\
                    .select('entry_number, final_rank, horse_id')\
                    .eq('race_id', race['race_id'])\
                    .execute()
                
                if not actual_results.data:
                    continue
                
                actual_winner = next((r for r in actual_results.data if r['final_rank'] == 1), None)
                
                # 각 전략별 테스트
                for strategy in strategies:
                    result = self._test_strategy(
                        strategy, predictions, actual_winner, race
                    )
                    
                    if result:
                        strategy_results[strategy['name']]['bets'].append(result)
                        strategy_results[strategy['name']]['total_bet'] += result['bet_amount']
                        strategy_results[strategy['name']]['total_return'] += result['return']
                        
                        if result['win']:
                            strategy_results[strategy['name']]['wins'] += 1
                            
            except Exception as e:
                print(f"오류 발생 - {race['race_date']} R{race['race_no']}: {e}")
                continue
        
        # 결과 계산
        for name, results in strategy_results.items():
            if results['total_bet'] > 0:
                results['total_bets'] = len(results['bets'])
                results['win_rate'] = results['wins'] / results['total_bets']
                results['profit'] = results['total_return'] - results['total_bet']
                results['roi'] = (results['profit'] / results['total_bet']) * 100
            else:
                results.update({
                    'total_bets': 0, 'win_rate': 0, 'profit': 0, 'roi': 0
                })
        
        self.test_results['comprehensive_backtest'] = strategy_results
        self._print_backtest_results(strategy_results)
        
        return strategy_results
    
    def _test_strategy(self, strategy, predictions, actual_winner, race_info):
        """개별 전략 테스트"""
        if strategy.get('type') == 'favorite':
            # 최고 인기마 (가장 낮은 배당률)
            best_horse = predictions.loc[predictions['win_odds'].idxmin()]
        elif strategy.get('type') == 'value':
            # 가치 베팅 (예측 확률 > 1/배당률)
            predictions['implied_prob'] = 1 / predictions['win_odds']
            predictions['value'] = predictions['win_probability'] - predictions['implied_prob']
            value_bets = predictions[predictions['value'] > 0]
            
            if len(value_bets) == 0:
                return None
            
            best_horse = value_bets.loc[value_bets['value'].idxmax()]
        else:
            # 확신도 기반
            if predictions['win_probability'].max() < strategy['threshold']:
                return None
            
            best_horse = predictions.iloc[0]  # 이미 확률순으로 정렬됨
        
        # 실제 결과 확인
        win = (actual_winner and 
               actual_winner['entry_number'] == best_horse['entry_number'])
        
        bet_amount = strategy['bet_amount']
        return_amount = bet_amount * best_horse['win_odds'] if win else 0
        
        return {
            'race_date': race_info['race_date'],
            'race_no': race_info['race_no'],
            'meet_code': race_info['meet_code'],
            'horse_entry': best_horse['entry_number'],
            'predicted_prob': best_horse['win_probability'],
            'odds': best_horse['win_odds'],
            'bet_amount': bet_amount,
            'return': return_amount,
            'win': win,
            'profit': return_amount - bet_amount
        }
    
    def _print_backtest_results(self, results):
        """백테스트 결과 출력"""
        print("\n" + "="*80)
        print("📊 백테스팅 결과 요약")
        print("="*80)
        
        result_df = []
        for strategy_name, result in results.items():
            if result['total_bets'] > 0:
                result_df.append({
                    '전략': strategy_name,
                    '총베팅': f"{result['total_bets']}회",
                    '적중': f"{result['wins']}회",
                    '적중률': f"{result['win_rate']:.1%}",
                    '총투자': f"{result['total_bet']:,}원",
                    '총수익': f"{result['profit']:,}원",
                    'ROI': f"{result['roi']:+.1f}%"
                })
        
        if result_df:
            df = pd.DataFrame(result_df)
            print(df.to_string(index=False))
        
        print("\n💡 해석:")
        print("- 적중률이 높다고 항상 수익성이 좋은 것은 아님")
        print("- ROI가 양수인 전략이 장기적으로 수익 가능")
        print("- 가치베팅 전략은 위험 대비 수익률이 중요")
    
    def model_performance_analysis(self, test_data, predictions):
        """모델 성능 상세 분석"""
        print("\n🔍 모델 성능 상세 분석")
        
        # 확률 구간별 정확도
        prob_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        test_data['prob_bin'] = pd.cut(predictions, bins=prob_bins, labels=bin_labels, include_lowest=True)
        
        print("\n📈 확률 구간별 실제 적중률:")
        calibration = test_data.groupby('prob_bin').agg({
            'is_winner': ['count', 'sum', 'mean']
        }).round(3)
        
        calibration.columns = ['예측수', '실제적중', '실제적중률']
        print(calibration)
        
        # 배당률 구간별 성능
        odds_bins = [0, 3, 5, 10, float('inf')]
        odds_labels = ['1-3배', '3-5배', '5-10배', '10배+']
        
        test_data['odds_bin'] = pd.cut(test_data['win_odds'], bins=odds_bins, labels=odds_labels, include_lowest=True)
        
        print("\n💰 배당률 구간별 예측 성능:")
        odds_performance = test_data.groupby('odds_bin').agg({
            'is_winner': ['count', 'mean'],
            'win_odds': 'mean'
        }).round(3)
        
        odds_performance.columns = ['예측수', '적중률', '평균배당']
        print(odds_performance)
    
    def feature_importance_analysis(self):
        """특성 중요도 분석"""
        if 'RandomForest' not in self.predictor.models:
            print("❌ RandomForest 모델이 훈련되지 않았습니다.")
            return
        
        rf_model = self.predictor.models['RandomForest']['model']
        feature_importance = pd.DataFrame({
            'feature': self.predictor.feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 특성 중요도 분석:")
        print(feature_importance.head(15))
        
        # 시각화
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def monthly_performance_analysis(self, backtest_results):
        """월별 성능 분석"""
        print("\n📅 월별 성능 분석")
        
        monthly_data = []
        
        for strategy_name, results in backtest_results.items():
            for bet in results['bets']:
                monthly_data.append({
                    'strategy': strategy_name,
                    'month': pd.to_datetime(bet['race_date']).strftime('%Y-%m'),
                    'profit': bet['profit'],
                    'win': bet['win']
                })
        
        if not monthly_data:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        monthly_df = pd.DataFrame(monthly_data)
        
        # 월별 수익률
        monthly_profit = monthly_df.groupby(['strategy', 'month']).agg({
            'profit': 'sum',
            'win': ['count', 'sum']
        }).round(2)
        
        monthly_profit.columns = ['월수익', '베팅수', '적중수']
        monthly_profit['적중률'] = (monthly_profit['적중수'] / monthly_profit['베팅수']).round(3)
        
        print("\n📊 전략별 월별 성과:")
        for strategy in monthly_df['strategy'].unique():
            print(f"\n[{strategy}]")
            strategy_data = monthly_profit.loc[strategy]
            print(strategy_data)
    
    def risk_analysis(self, backtest_results):
        """리스크 분석"""
        print("\n⚠️ 리스크 분석")
        
        for strategy_name, results in backtest_results.items():
            if results['total_bets'] == 0:
                continue
                
            profits = [bet['profit'] for bet in results['bets']]
            
            # 연속 손실 계산
            max_losing_streak = 0
            current_streak = 0
            
            for profit in profits:
                if profit < 0:
                    current_streak += 1
                    max_losing_streak = max(max_losing_streak, current_streak)
                else:
                    current_streak = 0
            
            # 최대 손실 (Drawdown)
            cumulative_profit = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative_profit)
            drawdown = running_max - cumulative_profit
            max_drawdown = np.max(drawdown)
            
            # 변동성 (표준편차)
            volatility = np.std(profits)
            
            # 샤프 비율 근사치 (무위험 수익률 0으로 가정)
            sharpe_ratio = np.mean(profits) / volatility if volatility > 0 else 0
            
            print(f"\n[{strategy_name}]")
            print(f"  최대 연속 손실: {max_losing_streak}회")
            print(f"  최대 손실폭: {max_drawdown:,.0f}원")
            print(f"  수익 변동성: {volatility:,.0f}원")
            print(f"  샤프 비율: {sharpe_ratio:.3f}")
    
    def generate_report(self, output_file='horse_racing_report.html'):
        """종합 리포트 생성"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>경마 예측 모델 성능 리포트</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏇 경마 예측 모델 성능 리포트</h1>
                <p>생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 백테스팅 요약</h2>
                <p>이 섹션에는 백테스팅 결과가 포함됩니다.</p>
                <!-- 백테스팅 결과 테이블 추가 -->
            </div>
            
            <div class="section">
                <h2>🎯 모델 성능</h2>
                <p>모델의 정확도, 정밀도, 재현율 등의 지표</p>
                <!-- 모델 성능 지표 추가 -->
            </div>
            
            <div class="section">
                <h2>💡 권장사항</h2>
                <ul>
                    <li>가장 수익성이 높은 전략을 식별</li>
                    <li>위험 관리 방안 수립</li>
                    <li>모델 개선 방향 제시</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 리포트가 {output_file}에 저장되었습니다.")

# 실사용 예시
def run_comprehensive_test():
    """포괄적 테스트 실행"""
    
    # 설정
    SUPABASE_URL = "https://your-project.supabase.co"
    SUPABASE_KEY = "your-anon-key"
    
    # 모델 로드 (이미 훈련된 모델 가정)
    from algorithm1 import HorseRacing1stPlacePredictor
    
    predictor = HorseRacing1stPlacePredictor(SUPABASE_URL, SUPABASE_KEY)
    
    # 데이터 로드 및 모델 훈련
    print("🤖 모델 훈련 중...")
    df = predictor.extract_training_data('2023-01-01', '2024-10-31')
    train_results = predictor.train_models(df, test_size=0.2)
    
    # 테스트 스위트 실행
    test_suite = HorseRacingTestSuite(predictor)
    
    print("\n" + "="*80)
    print("🧪 종합 테스트 시작")
    print("="*80)
    
    # 1. 포괄적 백테스팅
    backtest_results = test_suite.comprehensive_backtest(
        '2024-11-01', '2024-11-30'
    )
    
    # 2. 모델 성능 분석
    test_data = train_results['test_data']
    predictions = np.mean([
        train_results['results'][name]['probabilities'] 
        for name in train_results['results']
    ], axis=0)
    
    test_suite.model_performance_analysis(test_data, predictions)
    
    # 3. 특성 중요도 분석
    test_suite.feature_importance_analysis()
    
    # 4. 월별 성능 분석
    test_suite.monthly_performance_analysis(backtest_results)
    
    # 5. 리스크 분석
    test_suite.risk_analysis(backtest_results)
    
    # 6. 리포트 생성
    test_suite.generate_report()
    
    print("\n✅ 모든 테스트 완료!")
    
    return test_suite

if __name__ == "__main__":
    test_suite = run_comprehensive_test()