create view public.medical_features as
select
  hps.race_date,
  hps.meet_code,
  hps.race_id,
  hps.horse_id,
  hps.jk_no,
  hps.trainer_id,
  hps.entry_number,
  hps.horse_weight,
  hps.final_rank,
  hps.finish_time,
  hps.horse_race_days,
  hps.horse_weight_diff,
  hps.budam,
  hps.budam_weight,
  hps.horse_rating,
  hps.prize_money,
  hps.created_at,
  hps.updated_at,
  hps.race_time,
  hps.race_distance,
  hps.weather,
  hps.track_condition,
  hps.humidity_percent,
  hps.total_horses,
  hps.standardized_race_time,
  hps.prev_race_time_std,
  hps.prev_race_date,
  hps.horse_total_races,
  hps.horse_first_count,
  hps.horse_second_count,
  hps.horse_third_count,
  hps.horse_performance,
  COALESCE(mc.medical_count, 0::bigint) as medical_count
from
  horse_performance_stats hps
  left join (
    select
      hps_1.horse_id,
      hps_1.race_date,
      count(hm.diagnosis_date) as medical_count
    from
      horse_performance_stats hps_1
      left join horse_medical hm on hps_1.horse_id = hm.horse_id::text
      and hm.diagnosis_date < hps_1.race_date
    group by
      hps_1.horse_id,
      hps_1.race_date
  ) mc on hps.horse_id = mc.horse_id
  and hps.race_date = mc.race_date;