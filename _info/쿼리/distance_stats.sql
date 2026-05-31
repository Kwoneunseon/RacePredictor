create view public.distance_stats as
select
  r.race_distance,
  avg(rr.race_time) as mean_time,
  stddev(rr.race_time) as std_time,
  count(*) as race_count
from
  race_entries rr
  join races r on rr.race_id = r.race_id
  and rr.race_date = r.race_date
  and rr.meet_code = r.meet_code
group by
  r.race_distance
order by
  r.race_distance;