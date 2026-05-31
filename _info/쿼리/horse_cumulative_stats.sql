create view public.horse_cumulative_stats as
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
  count(*) over (
    partition by
      horse_id
    order by
      race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_total_races,
  sum(
    case
      when final_rank = 1 then 1
      else 0
    end
  ) over (
    partition by
      horse_id
    order by
      race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_first_count,
  sum(
    case
      when final_rank = 2 then 1
      else 0
    end
  ) over (
    partition by
      horse_id
    order by
      race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_second_count,
  sum(
    case
      when final_rank = 3 then 1
      else 0
    end
  ) over (
    partition by
      horse_id
    order by
      race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_third_count
from
  previous_race_features;