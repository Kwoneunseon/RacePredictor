create materialized view public.race_analysis_step2 as
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
  case
    when horse_weight = 0 then COALESCE(
      avg(horse_weight) filter (
        where
          horse_weight > 0
      ) over (
        partition by
          horse_id
        order by
          race_date rows between 3 PRECEDING
          and 1 PRECEDING
      ),
      horse_weight::numeric
    )
    else horse_weight::numeric
  end as adjusted_horse_weight,
  budam_weight - lag(budam_weight) over (
    order by
      race_date
  ) as budam_weight_diff,
  count(*) filter (
    where
      final_rank > 0
  ) over (
    partition by
      horse_id,
      jk_no
    order by
      race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_jockey_total_races,
  count(*) filter (
    where
      final_rank = 1
  ) over (
    partition by
      horse_id,
      jk_no
    order by
      race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_jockey_wins,
  count(*) filter (
    where
      final_rank <= 3
  ) over (
    partition by
      horse_id,
      jk_no
    order by
      race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_jockey_top3
from
  race_analysis_step1;