from dataclasses import dataclass
from config import (
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    EV_THRESHOLD_SCRAPING,
    EV_THRESHOLD_FALLBACK,
)


@dataclass
class BetRecommendation:
    bet_type: str   # "단승" | "연승"
    amount: int     # 원, 100원 단위
    ev: float
    kelly_f: float


class EVCalculator:
    def calculate_ev(self, model_prob: float, odds: float) -> float:
        return model_prob * odds - 1

    def kelly_fraction(self, p: float, b: float) -> float:
        """b = odds - 1 (순이익 배율)"""
        q = 1 - p
        f = (p * b - q) / b
        return max(f, 0.0)

    def bet_size(self, bankroll: float, kelly_f: float) -> int:
        raw = bankroll * kelly_f * KELLY_FRACTION
        capped = min(raw, bankroll * MAX_BET_FRACTION)
        rounded = int(capped // 100) * 100
        return max(rounded, 100)

    def recommend(
        self,
        model_prob: float,
        odds: float,
        bankroll: float,
        source: str,
    ) -> "BetRecommendation | None":
        ev = self.calculate_ev(model_prob, odds)
        threshold = EV_THRESHOLD_SCRAPING if source == "scraping" else EV_THRESHOLD_FALLBACK
        if ev <= threshold:
            return None

        b = odds - 1
        kelly_f = self.kelly_fraction(model_prob, b)
        if kelly_f == 0:
            return None

        amount = self.bet_size(bankroll, kelly_f)
        bet_type = "단승" if model_prob >= 0.30 else "연승"

        return BetRecommendation(
            bet_type=bet_type,
            amount=amount,
            ev=ev,
            kelly_f=kelly_f,
        )
