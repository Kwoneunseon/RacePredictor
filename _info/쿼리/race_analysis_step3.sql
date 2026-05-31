create materialized view public.race_analysis_step3 as
select
  race_id,
  horse_id,
  race_date,
  meet_code,
  entry_number,
  horse_weight,
  final_rank,
  finish_time,
  horse_race_days,
  horse_weight_diff,
  budam,
  budam_weight,
  horse_rating,
  jk_no,
  trainer_id,
  race_time,
  horse_age,
  gender,
  horse_class,
  horse_name,
  race_distance,
  total_horses,
  planned_horses,
  race_grade,
  track_condition,
  weather,
  weight_type,
  adjusted_horse_weight,
  budam_weight_diff,
  horse_jockey_total_races,
  horse_jockey_wins,
  horse_jockey_top3,
  COALESCE(
    avg(
      case
        when COALESCE(race_time, 0::numeric) > 0::numeric then race_distance::numeric / race_time
        else 15.7
      end
    ) over (
      partition by
        horse_id
      order by
        race_date rows between 5 PRECEDING
        and 1 PRECEDING
    ),
    15.7
  ) as prev_5_avg_speed_mps,
  COALESCE(
    avg(
      case
        when COALESCE(race_time, 0::numeric) > 0::numeric then race_distance::numeric / race_time
        else 15.7
      end
    ) over (
      partition by
        horse_id
      order by
        race_date rows between 3 PRECEDING
        and 1 PRECEDING
    ),
    15.7
  ) as prev_3_avg_speed_mps,
  COALESCE(
    avg(
      case
        when COALESCE(race_time, 0::numeric) > 0::numeric then race_distance::numeric / race_time
        else 0::numeric
      end
    ) over (
      partition by
        horse_id
      order by
        race_date rows between 3 PRECEDING
        and 1 PRECEDING
    ) - avg(
      case
        when COALESCE(race_time, 0::numeric) > 0::numeric then race_distance::numeric / race_time
        else 0::numeric
      end
    ) over (
      partition by
        horse_id
      order by
        race_date rows between 6 PRECEDING
        and 4 PRECEDING
    ),
    0::numeric
  ) as speed_improvement_trend
from
  race_analysis_step2;