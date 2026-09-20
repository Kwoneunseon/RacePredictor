"""horse_race_history 테이블 재구성 모듈"""
import pandas as pd
import numpy as np
from config import NON_FINISH_RANK_MIN
from Utils.database import get_client, upsert
from Utils.ranks import valid_rank_mask
from Utils.logger import get_logger

logger = get_logger(__name__)
PAGE_SIZE = 1000
BATCH_SIZE = 500
_HISTORY_COLUMNS = [
    "horse_id", "as_of_date", "prev_total_races", "prev_wins",
    "prev_top3", "prev_5_avg_rank", "prev_total_avg_rank",
]


def load_all_entries(client) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        res = client.table("race_entries")\
            .select("horse_id, race_date, final_rank, horse_weight")\
            .gt("final_rank", 0).lt("final_rank", NON_FINISH_RANK_MIN)\
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
    각 (horse_id, race_date)에 대해 경주 직전 기준 누적 통계 계산.
    반환 컬럼: horse_id, as_of_date, prev_total_races, prev_wins,
               prev_top3, prev_5_avg_rank, prev_total_avg_rank
    """
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["final_rank"] = pd.to_numeric(df["final_rank"], errors="coerce")
    df = df.dropna(subset=["horse_id", "race_date", "final_rank"])
    # 0(미기입)·90번대(취소/제외/주행중지 등 비착순 상태 코드)는 착순 통계에서 제외
    df = df[valid_rank_mask(df["final_rank"])]
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    results = []
    for horse_id, grp in df.groupby("horse_id", sort=False):
        grp = grp.sort_values("race_date").reset_index(drop=True)
        ranks = grp["final_rank"].values
        dates = grp["race_date"].values

        for i in range(len(grp)):
            prior = ranks[:i]
            if len(prior) == 0:
                prev_total, prev_wins, prev_top3 = 0, 0, 0
                prev_5_avg = float(ranks[i])
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

    return pd.DataFrame(results, columns=_HISTORY_COLUMNS)


def rebuild(start_date: str = None, end_date: str = None) -> None:
    """
    horse_race_history 테이블 재구성.
    start_date, end_date: 특정 구간만 upsert할 때 지정 (None이면 전체).
    """
    client = get_client()

    logger.info("race_entries 전체 로드 중...")
    df_all = load_all_entries(client)
    logger.info(f"총 {len(df_all)}건 로드 완료")

    logger.info("누적 통계 계산 중...")
    hist_df = compute_history(df_all)
    logger.info(f"계산 완료: {len(hist_df)}행")

    if start_date:
        hist_df = hist_df[hist_df["as_of_date"] >= start_date]
    if end_date:
        hist_df = hist_df[hist_df["as_of_date"] <= end_date]
    logger.info(f"upsert 대상: {len(hist_df)}행 ({start_date or '전체'} ~ {end_date or '전체'})")

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
    import sys
    rebuild(
        start_date=sys.argv[1] if len(sys.argv) > 1 else None,
        end_date=sys.argv[2] if len(sys.argv) > 2 else None,
    )
