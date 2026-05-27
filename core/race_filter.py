from config import FILTER_DISTANCES, FILTER_MIN_HORSES


class RaceFilter:
    def is_eligible(self, race: dict) -> bool:
        if race.get("race_distance") not in FILTER_DISTANCES:
            return False
        if race.get("race_grade", "") != "일반":
            return False
        if (race.get("total_horses") or 0) < FILTER_MIN_HORSES:
            return False
        return True

    def filter_races(self, races: list[dict]) -> list[dict]:
        return [r for r in races if self.is_eligible(r)]
