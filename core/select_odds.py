from collections import defaultdict

import numpy as np

from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


class OddsCollector:
    def get_odds(self, race_date: str, meet_code: int) -> dict:
        """
        반환: {race_id: {horse_id: float, "_source": "scraping"|"fallback"}}
        KRA 경마속보 스크래핑 시도 → 실패 시 DB 중앙값으로 fallback
        """
        scraped = self._scrape_odds(race_date, meet_code)
        if scraped:
            return scraped
        return self._fallback_from_db(race_date, meet_code)

    def _scrape_odds(self, race_date: str, meet_code: int) -> dict:
        """KRA 경마속보에서 경주별 말별 배당을 스크래핑."""
        try:
            from library.scraper_odds import fetch_race_win_odds

            client = get_client()
            races_res = client.table("races")\
                .select("race_id, race_no")\
                .eq("race_date", race_date)\
                .eq("meet_code", meet_code)\
                .execute()
            if not races_res.data:
                return {}

            entries_res = client.table("race_entries")\
                .select("race_id, horse_id, entry_number")\
                .eq("race_date", race_date)\
                .execute()
            if not entries_res.data:
                return {}

            entry_map: dict[str, dict[int, str]] = defaultdict(dict)
            for e in entries_res.data:
                if e.get("entry_number"):
                    entry_map[e["race_id"]][e["entry_number"]] = e["horse_id"]

            result = {}
            for race in races_res.data:
                race_id = race["race_id"]
                race_no = race.get("race_no")
                if not race_no:
                    continue
                odds_by_entry = fetch_race_win_odds(meet_code, race_no, race_date)
                if not odds_by_entry:
                    continue
                race_entry_map = entry_map.get(race_id, {})
                race_odds: dict = {}
                for entry_no, odds in odds_by_entry.items():
                    horse_id = race_entry_map.get(entry_no)
                    if horse_id:
                        race_odds[horse_id] = odds
                if race_odds:
                    race_odds["_source"] = "scraping"
                    result[race_id] = race_odds

            if result:
                logger.info(f"배당 스크래핑 성공: meet={meet_code} {race_date} → {len(result)}경주")
            return result

        except Exception as e:
            logger.debug(f"배당 스크래핑 실패: {e}")
            return {}

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
