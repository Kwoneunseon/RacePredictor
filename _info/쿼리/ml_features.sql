create materialized view public.ml_features as
select
  mf.race_date,
  mf.meet_code,
  mf.race_id,
  mf.horse_id,
  h.name,
  mf.humidity_percent,
  mf.race_distance,
  mf.total_horses,
  mf.entry_number,
  h.age::numeric - EXTRACT(
    year
    from
      age (
        CURRENT_DATE::timestamp with time zone,
        mf.race_date::timestamp with time zone
      )
  ) as age,
  mf.budam_weight,
  mf.horse_weight,
  mf.horse_weight_diff,
  mf.prev_race_time_std,
  mf.medical_count,
  mf.horse_total_races,
  mf.horse_first_count,
  mf.horse_second_count,
  mf.horse_third_count,
  mf.horse_performance,
  COALESCE(jts_jockey.jockey_total_races, 0::bigint) as jockey_total_races,
  COALESCE(jts_jockey.jockey_first_count, 0::bigint) as jockey_first_count,
  COALESCE(jts_jockey.jockey_second_count, 0::bigint) as jockey_second_count,
  COALESCE(jts_jockey.jockey_third_count, 0::bigint) as jockey_third_count,
  COALESCE(jts_jockey.jockey_performance, 0::numeric) as jockey_performance,
  COALESCE(jts_trainer.jockey_total_races, 0::bigint) as trainer_total_races,
  COALESCE(jts_trainer.jockey_first_count, 0::bigint) as trainer_first_count,
  COALESCE(jts_trainer.jockey_second_count, 0::bigint) as trainer_second_count,
  COALESCE(jts_trainer.jockey_third_count, 0::bigint) as trainer_third_count,
  COALESCE(jts_trainer.jockey_performance, 0::numeric) as trainer_performance,
  t.rc_cnt_t as trainer_race_count,
  mf.weather,
  mf.track_condition,
  h.gender as horse_sex,
  h.breed as horse_birthplace,
  h.rank,
  mf.standardized_race_time,
  mf.final_rank
from
  medical_features mf
  join horses h on mf.horse_id = h.horse_id
  left join jockeys j on mf.jk_no = j.jk_no
  left join trainers t on mf.trainer_id = t.trainer_id::text
  left join jockey_trainer_stats jts_jockey on mf.jk_no = jts_jockey.jk_no
  left join jockey_trainer_stats jts_trainer on mf.trainer_id = jts_trainer.jk_no
where
  mf.prev_race_time_std is not null;