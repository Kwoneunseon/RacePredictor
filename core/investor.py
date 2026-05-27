import pandas as pd
from Utils.database import upsert, get_client
from Utils.logger import get_logger
from core.ev_calculator import EVCalculator
from core.odds_collector import OddsCollector
from config import BANKROLL

logger = get_logger(__name__)


class Investor:
    def __init__(self, bankroll: float = BANKROLL):
        self.bankroll = bankroll
        self.ev_calc = EVCalculator()
        self.odds_collector = OddsCollector()

    def decide(
        self,
        predictions: pd.DataFrame,
        race_date: str,
        meet_code: int,
    ) -> list[dict]:
        if predictions.empty:
            return []

        odds_map = self.odds_collector.get_odds(race_date, meet_code)
        decisions = []

        KRA_PAYOUT_RATE = 0.75  # KRA 환급률 75%

        race_groups = predictions.groupby("race_id", sort=False)
        for race_id, group in race_groups:
            race_odds = odds_map.get(race_id, {})
            source = race_odds.get("_source", "fallback")

            top = group.sort_values("win_probability", ascending=False).iloc[0]
            horse_id = top.get("horse_id")
            model_prob = float(top["win_probability"])

            if source == "fallback":
                # 경주 내 상대 확률로 배당 추정: est_odds = (sum_probs / model_prob) * KRA_PAYOUT_RATE
                # 모든 말의 확률 합이 클수록(모델이 특정 말을 강하게 선택) EV가 높아짐
                sum_probs = float(group["win_probability"].fillna(0).sum())
                if sum_probs <= 0 or model_prob <= 0:
                    logger.debug(f"{race_id}: 확률 계산 불가, 스킵")
                    continue
                odds = (sum_probs / model_prob) * KRA_PAYOUT_RATE
            else:
                odds = float(race_odds.get(horse_id, 0))
                if odds <= 0:
                    logger.debug(f"{race_id}: 배당 없음, 스킵")
                    continue

            rec = self.ev_calc.recommend(model_prob, odds, self.bankroll, source)
            if rec is None:
                logger.debug(f"{race_id}: EV 미달, 스킵")
                continue

            decisions.append({
                "race_id": race_id,
                "horse_id": horse_id,
                "horse_name": top.get("horse_name", horse_id),
                "bet_type": rec.bet_type,
                "amount": rec.amount,
                "ev": rec.ev,
                "kelly_f": rec.kelly_f,
                "model_prob": model_prob,
                "odds": odds,
                "source": source,
            })

        logger.info(f"베팅 결정: {len(decisions)}건 / {len(list(race_groups))}경주")
        return decisions

    def save_bets(self, decisions: list[dict]) -> bool:
        if not decisions:
            return True
        rows = [
            {
                "race_id": d["race_id"],
                "horse_id": d["horse_id"],
                "bet_amount": d["amount"],
                "bet_type": d["bet_type"],
                "odds": d["odds"],
            }
            for d in decisions
        ]
        return upsert("betting_records", rows, "id")

    def calculate_profit(self, odds: float, won: bool, bet_amount: int) -> int:
        if won:
            return int(bet_amount * odds) - bet_amount
        return -bet_amount

    def update_results(self, race_id: str) -> None:
        client = get_client()
        try:
            entries = client.table("race_entries")\
                .select("horse_id, final_rank, win_odds")\
                .eq("race_id", race_id)\
                .execute().data or []
            info_map = {e["horse_id"]: e for e in entries}

            bets = client.table("betting_records")\
                .select("id, horse_id, bet_amount")\
                .eq("race_id", race_id)\
                .execute().data or []

            for bet in bets:
                info = info_map.get(bet["horse_id"], {})
                won = info.get("final_rank") == 1
                odds = float(info.get("win_odds") or 0)
                profit = self.calculate_profit(odds, won, bet["bet_amount"])
                client.table("betting_records")\
                    .update({"profit": profit})\
                    .eq("id", bet["id"])\
                    .execute()
            logger.info(f"{race_id} 결과 업데이트 완료")
        except Exception as e:
            logger.error(f"결과 업데이트 실패: {e}")
