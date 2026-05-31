create view public.distance_performance as
select
  re.horse_id,
  r.race_distance,
  avg(re.final_rank) as avg_rank_at_distance,
  count(*) as races_at_distance,
  avg(re.race_time) as avg_time_at_distance,
  min(re.race_time) as best_time_at_distance,
  max(re.race_time) as wort_time_at_distance,
  COALESCE(stddev(re.race_time), 0::numeric) as time_consistency_at_distance
from
  race_entries re
  join races r on re.race_id = r.race_id
  and re.race_date = r.race_date
  and re.meet_code = r.meet_code
where
  COALESCE(re.final_rank, 0) <> 0
  and COALESCE(re.race_time, 0::numeric) <> 0::numeric
group by
  re.horse_id,
  r.race_distance;