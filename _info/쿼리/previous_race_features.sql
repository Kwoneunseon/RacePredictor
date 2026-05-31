create view public.previous_race_features as
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
  lag(standardized_race_time) over (
    partition by
      horse_id
    order by
      race_date
  ) as prev_race_time_std,
  lag(race_date) over (
    partition by
      horse_id
    order by
      race_date
  ) as prev_race_date
from
  standardized_results
order by
  horse_id,
  race_date;