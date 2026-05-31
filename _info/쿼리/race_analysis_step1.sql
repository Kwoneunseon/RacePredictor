create materialized view public.race_analysis_step1 as
select
  re.race_id,
  re.horse_id,
  re.race_date,
  re.meet_code,
  re.entry_number,
  re.horse_weight,
  re.final_rank,
  re.finish_time,
  re.horse_race_days,
  re.horse_weight_diff,
  re.budam,
  re.budam_weight,
  re.horse_rating,
  re.jk_no,
  re.trainer_id,
  re.race_time,
  h.age as horse_age,
  h.gender,
  h.rank as horse_class,
  h.name as horse_name,
  r.race_distance,
  r.total_horses,
  r.planned_horses,
  r.race_grade,
  r.track_condition,
  r.weather,
  r.weight_type
from
  race_entries re
  join horses h on re.horse_id = h.horse_id
  join races r on re.race_id = r.race_id
  and re.race_date = r.race_date
  and re.meet_code = r.meet_code;