create view public.standardized_results as
select
  rr.race_date,
  rr.meet_code,
  rr.race_id,
  rr.horse_id,
  rr.jk_no,
  rr.trainer_id,
  rr.entry_number,
  rr.horse_weight,
  rr.final_rank,
  rr.finish_time,
  rr.horse_race_days,
  rr.horse_weight_diff,
  rr.budam,
  rr.budam_weight,
  rr.horse_rating,
  rr.prize_money,
  rr.created_at,
  rr.updated_at,
  rr.race_time,
  r.race_distance,
  r.weather,
  regexp_replace(r.track_condition, '\s*\([^)]*\)'::text, ''::text) as track_condition,
  regexp_replace(
    r.track_condition,
    '.*\((\d+)%?\).*'::text,
    '\1'::text
  )::integer as humidity_percent,
  r.total_horses,
  (rr.race_time - ds.mean_time) / NULLIF(ds.std_time, 0::numeric) as standardized_race_time
from
  race_entries rr
  join races r on rr.race_id = r.race_id
  and rr.race_date = r.race_date
  and rr.meet_code = r.meet_code
  join distance_stats ds on r.race_distance = ds.race_distance
where
  ds.std_time > 0::numeric;