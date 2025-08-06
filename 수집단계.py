from datetime import datetime
from library import collecting_race_results

print(f"🏁 경마 데이터 수집 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#경마 경주 결과 수집
collecting_race_results(start_page=1, max_pages=50, start_date='20230101', end_date='20240101')  

print(f"✅ 수집 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 2025-07-20까지의 결과 수집 완료