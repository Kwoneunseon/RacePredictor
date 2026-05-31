create materialized view public.race_analysis_step4 as
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
  prev_5_avg_speed_mps,
  prev_3_avg_speed_mps,
  speed_improvement_trend,
  avg(race_time) over (
    partition by
      horse_id,
      race_distance
    order by
      race_date rows between 3 PRECEDING
      and 1 PRECEDING
  ) as prev_3_avg_time_same_distance_raw,
  COALESCE(
    lag(
      case
        when final_rank <= 3 then 1
        else 0
      end
    ) over (
      partition by
        horse_id,
        race_distance
      order by
        race_date
    ),
    0
  ) as prev_race_was_top3,
  lag(race_time) over (
    partition by
      horse_id,
      race_distance
    order by
      race_date
  ) as prev_race_time_raw
from
  race_analysis_step3;