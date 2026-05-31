create view public.horse_performance_stats as
select
  race_date,
  meet_code,
  race_id,
  horse_id,
  jk_no,
  trainer_id,
  entry_number,
  horse_weight,
  final_rank,
  finish_time,
  horse_race_days,
  horse_weight_diff,
  budam,
  budam_weight,
  horse_rating,
  prize_money,
  created_at,
  updated_at,
  race_time,
  race_distance,
  weather,
  track_condition,
  humidity_percent,
  total_horses,
  standardized_race_time,
  prev_race_time_std,
  prev_race_date,
  horse_total_races,
  horse_first_count,
  horse_second_count,
  horse_third_count,
  case
    when horse_total_races > 0 then (
      5 * horse_first_count + 4 * horse_second_count + 3 * horse_third_count + 2 * (
        horse_total_races - horse_first_count - horse_second_count - horse_third_count - GREATEST(
          0::bigint,
          horse_total_races - horse_first_count - horse_second_count - horse_third_count
        )
      )
    )::numeric / (15.0 * horse_total_races::numeric)
    else 0::numeric
  end as horse_performance
from
  horse_cumulative_stats;