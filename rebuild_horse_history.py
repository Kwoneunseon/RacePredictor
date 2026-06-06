"""
horse_race_history 테이블 재구성 스크립트
race_entries의 전체 기간을 기반으로 각 경주일 기준 누적 통계를 계산해서 upsert

실행: python rebuild_horse_history.py [시작일] [종료일]
예)   python rebuild_horse_history.py 2024-01-01 2025-05-31  (빈 구간만 채울 때)
      python rebuild_horse_history.py                        (전체 재구성)
"""
import sys
import pandas as pd
import numpy as np
from Utils.database import get_client, upsert
from Utils.logger import get_logger

logger = get_logger(__name__)
PAGE_SIZE = 1000
BATCH_SIZE = 500


def load_all_entries(client) -> pd.DataFrame:
    """race_entries 전체 로드 (final_rank > 0)"""
    rows = []
    offset = 0
    while True:
        res = client.table("race_entries")\
            .select("horse_id, race_date, final_rank, horse_weight")\
            .gt("final_rank", 0)\
            .order("race_date").order("horse_id")\
            .range(offset, offset + PAGE_SIZE - 1)\
            .execute()
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        if offset % 5000 == 0:
            logger.info(f"  로드 중... {offset}건")
    return pd.DataFrame(rows)


def compute_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 (horse_id, race_date) 에 대해 경주 직전 기준 누적 통계 계산.
    반환 컬럼: horse_id, as_of_date, prev_total_races, prev_wins,
               prev_top3, prev_5_avg_rank, prev_total_avg_rank
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["final_rank"] = pd.to_numeric(df["final_rank"], errors="coerce")
    df = df.dropna(subset=["horse_id", "race_date", "final_rank"])
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    results = []
    for horse_id, grp in df.groupby("horse_id", sort=False):
        grp = grp.sort_values("race_date").reset_index(drop=True)
        ranks = grp["final_rank"].values
        dates = grp["race_date"].values

        for i in range(len(grp)):
            prior = ranks[:i]  # 당일 경주 제외, 직전까지만
            if len(prior) == 0:
                prev_total = 0
                prev_wins = 0
                prev_top3 = 0
                prev_5_avg = float(ranks[i])   # 첫 경주면 자기 자신으로 폴백
                prev_total_avg = float(ranks[i])
            else:
                prev_total = len(prior)
                prev_wins = int((prior == 1).sum())
                prev_top3 = int((prior <= 3).sum())
                last5 = prior[-5:] if len(prior) >= 5 else prior
                prev_5_avg = round(float(last5.mean()), 2)
                prev_total_avg = round(float(prior.mean()), 2)

            results.append({
                "horse_id": horse_id,
                "as_of_date": str(dates[i])[:10],
                "prev_total_races": prev_total,
                "prev_wins": prev_wins,
                "prev_top3": prev_top3,
                "prev_5_avg_rank": prev_5_avg,
                "prev_total_avg_rank": prev_total_avg,
            })

    return pd.DataFrame(results)


def main():
    client = get_client()

    # 날짜 범위 인자
    filter_start = sys.argv[1] if len(sys.argv) > 1 else None
    filter_end   = sys.argv[2] if len(sys.argv) > 2 else None

    logger.info("race_entries 전체 로드 중...")
    df_all = load_all_entries(client)
    logger.info(f"총 {len(df_all)}건 로드 완료 ({df_all['race_date'].min()} ~ {df_all['race_date'].max()})")

    logger.info("누적 통계 계산 중... (시간이 걸릴 수 있습니다)")
    hist_df = compute_history(df_all)
    logger.info(f"계산 완료: {len(hist_df)}행")

    # 날짜 범위 필터 (특정 구간만 upsert할 때)
    if filter_start:
        hist_df = hist_df[hist_df["as_of_date"] >= filter_start]
    if filter_end:
        hist_df = hist_df[hist_df["as_of_date"] <= filter_end]
    logger.info(f"upsert 대상: {len(hist_df)}행 ({filter_start or '전체'} ~ {filter_end or '전체'})")

    # 배치 upsert
    rows = hist_df.to_dict("records")
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        ok = upsert("horse_race_history", batch, on_conflict="horse_id,as_of_date")
        if ok:
            total += len(batch)
            logger.info(f"  upsert {total}/{len(rows)}건 완료")
        else:
            logger.error(f"  배치 {i} upsert 실패")

    logger.info(f"horse_race_history 업데이트 완료: 총 {total}건")


if __name__ == "__main__":
    main()
