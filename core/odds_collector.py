from collections import defaultdict

import numpy as np

from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


class OddsCollector:
    def get_odds(self, race_date: str, meet_code: int) -> dict:
        """
        반환: {race_id: {horse_id: float, "_source": "scraping"|"fallback"}}
        Playwright 스크래핑 미구현 → 항상 fallback
        """
        return self._fallback_from_db(race_date, meet_code)

    def _fallback_from_db(self, race_date: str, meet_code: int) -> dict:
        client = get_client()

        races_res = client.table("races")\
            .select("race_id, race_distance")\
            .eq("race_date", race_date)\
            .eq("meet_code", meet_code)\
            .execute()
        if not races_res.data:
            return {}

        median_map = self._load_median_odds(meet_code)

        result = {}
        for race in races_res.data:
            race_id = race["race_id"]
            dist = race.get("race_distance")
            fallback_odds = median_map.get(dist, 15.0)

            entries_res = client.table("race_entries")\
                .select("horse_id")\
                .eq("race_id", race_id)\
                .execute()
            if not entries_res.data:
                continue

            result[race_id] = {e["horse_id"]: fallback_odds for e in entries_res.data}
            result[race_id]["_source"] = "fallback"

        logger.info(f"폴백 배당 로드: {len(result)}경주 (meet={meet_code})")
        return result

    def _load_median_odds(self, meet_code: int) -> dict[int, float]:
        client = get_client()
        res = client.table("race_entries")\
            .select("win_odds, races(race_distance, meet_code)")\
            .gt("win_odds", 0)\
            .execute()
        if not res.data:
            return {}

        grouped: dict[int, list] = defaultdict(list)
        for row in res.data:
            rc = row.get("races") or {}
            if rc.get("meet_code") != meet_code:
                continue
            dist = rc.get("race_distance")
            odds = row.get("win_odds")
            if dist and odds:
                grouped[dist].append(float(odds))

        return {dist: float(np.median(vals)) for dist, vals in grouped.items()}
