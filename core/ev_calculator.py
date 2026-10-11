from dataclasses import dataclass
from config import (
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    EV_THRESHOLD_SCRAPING,
    EV_THRESHOLD_FALLBACK,
)

# race_payouts.bet_type(영문 snake_case: win/place/quinella/exacta/
# quinella_place/trio/trifecta)과 어휘를 통일한 정식 값. betting_records.bet_type
# 저장 및 이 모듈의 판단 로직은 항상 이 값을 쓰고, 사용자向 한글 표시가 필요한
# 화면(main.py 콘솔 출력, core/prediction_report.py HTML)에서만 이 매핑으로
# 변환한다. (2026-07-30, quant-model-expert: race_payouts와의 조인/백테스트를
# 위해 한글 표기를 폐기하고 영문으로 표준화)
BET_TYPE_LABELS_KO = {"win": "단승", "place": "연승"}


@dataclass
class BetRecommendation:
    bet_type: str   # "win" | "place" (race_payouts.bet_type과 동일 어휘)
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
        bet_type = "win" if model_prob >= 0.30 else "place"

        return BetRecommendation(
            bet_type=bet_type,
            amount=amount,
            ev=ev,
            kelly_f=kelly_f,
        )
