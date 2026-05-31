create view public.horse_basic_stats as
select
  re.horse_id,
  re.race_date,
  count(*) over (
    partition by
      re.horse_id
    order by
      re.race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as prev_total_races,
  COALESCE(
    avg(re.final_rank) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between 5 PRECEDING
        and 1 PRECEDING
    ),
    16::numeric
  ) as prev_5_avg_rank,
  COALESCE(
    avg(re.final_rank) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between UNBOUNDED PRECEDING
        and 1 PRECEDING
    ),
    16::numeric
  ) as prev_total_avg_rank,
  COALESCE(
    sum(
      case
        when re.final_rank = 1 then 1
        else 0
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between UNBOUNDED PRECEDING
        and 1 PRECEDING
    ),
    0::bigint
  ) as prev_wins,
  COALESCE(
    sum(
      case
        when re.final_rank <= 3 then 1
        else 0
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between UNBOUNDED PRECEDING
        and 1 PRECEDING
    ),
    0::bigint
  ) as prev_top3,
  COALESCE(
    sum(
      case
        when re.final_rank <= 5 then 1
        else 0
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between UNBOUNDED PRECEDING
        and 1 PRECEDING
    ),
    0::bigint
  ) as prev_top5,
  COALESCE(
    sum(
      case
        when re.final_rank = 1
        and re.race_date >= (re.race_date - '1 year'::interval) then 1
        else 0
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between UNBOUNDED PRECEDING
        and 1 PRECEDING
    ),
    0::bigint
  ) as year_wins,
  COALESCE(
    sum(
      case
        when re.final_rank <= 3
        and re.race_date >= (re.race_date - '1 year'::interval) then 1
        else 0
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between UNBOUNDED PRECEDING
        and 1 PRECEDING
    ),
    0::bigint
  ) as year_top3,
  COALESCE(
    sum(
      case
        when re.final_rank <= 5
        and re.race_date >= (re.race_date - '1 year'::interval) then 1
        else 0
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between UNBOUNDED PRECEDING
        and 1 PRECEDING
    ),
    0::bigint
  ) as year_top5,
  COALESCE(h.total_races, 0) as total_races,
  COALESCE(h.total_win_rate, 0::numeric) as total_win_rate,
  COALESCE(h.total_place_rate, 0::numeric) as total_place_rate,
  COALESCE(h.year_races, 0) as year_races,
  COALESCE(h.year_win_rate, 0::numeric) as year_win_rate,
  COALESCE(h.year_place_rate, 0::numeric) as year_place_rate
from
  race_entries re
  join horses h on re.horse_id = h.horse_id
where
  re.final_rank is not null;