import pandas as pd
from Utils.database import upsert, get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


class Investor:
    def __init__(self, threshold: float = 0.6, bet_amount: int = 1000):
        self.threshold = threshold
        self.bet_amount = bet_amount

    def decide(self, predictions: pd.DataFrame) -> pd.DataFrame:
        bets = predictions[predictions["win_probability"] >= self.threshold].copy()
        bets["bet_amount"] = self.bet_amount
        bets["bet_type"] = "단승"
        logger.info(f"베팅 결정: {len(bets)}건 (임계값={self.threshold})")
        return bets

    def calculate_profit(self, odds: float, won: bool) -> int:
        if won:
            return int(self.bet_amount * odds) - self.bet_amount
        return -self.bet_amount

    def save_bets(self, bets: pd.DataFrame, race_id: str) -> bool:
        if bets.empty:
            return True
        rows = []
        for _, row in bets.iterrows():
            rows.append({
                "race_id": race_id,
                "horse_id": row["horse_id"],
                "bet_amount": self.bet_amount,
                "bet_type": "단승",
                "odds": row.get("win_odds", None),
            })
        return upsert("betting_records", rows, "id")

    def update_results(self, race_id: str) -> None:
        client = get_client()
        try:
            entries = client.table("race_entries")\
                .select("horse_id, final_rank")\
                .eq("race_id", race_id)\
                .execute().data or []
            rank_map = {e["horse_id"]: e["final_rank"] for e in entries}

            bets = client.table("betting_records")\
                .select("id, horse_id, bet_amount")\
                .eq("race_id", race_id)\
                .execute().data or []

            for bet in bets:
                final_rank = rank_map.get(bet["horse_id"])
                won = final_rank == 1
                odds_row = client.table("race_entries")\
                    .select("win_odds")\
                    .eq("race_id", race_id)\
                    .eq("horse_id", bet["horse_id"])\
                    .execute().data
                odds = odds_row[0]["win_odds"] if odds_row else 0
                profit = self.calculate_profit(odds or 0, won)
                client.table("betting_records")\
                    .update({"profit": profit})\
                    .eq("id", bet["id"])\
                    .execute()
            logger.info(f"{race_id} 결과 업데이트 완료")
        except Exception as e:
            logger.error(f"결과 업데이트 실패: {e}")
