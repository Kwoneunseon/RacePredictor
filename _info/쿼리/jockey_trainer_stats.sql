create view public.jockey_trainer_stats as
select
  race_entries.jk_no,
  count(*) as jockey_total_races,
  sum(
    case
      when race_entries.final_rank = 1 then 1
      else 0
    end
  ) as jockey_first_count,
  sum(
    case
      when race_entries.final_rank = 2 then 1
      else 0
    end
  ) as jockey_second_count,
  sum(
    case
      when race_entries.final_rank = 3 then 1
      else 0
    end
  ) as jockey_third_count,
  sum(
    (6 - LEAST(race_entries.final_rank, 5))::numeric * 1.0
  ) / (15.0 * count(*)::numeric) as jockey_performance
from
  race_entries
group by
  race_entries.jk_no
union all
select
  race_entries.trainer_id as jk_no,
  count(*) as jockey_total_races,
  sum(
    case
      when race_entries.final_rank = 1 then 1
      else 0
    end
  ) as jockey_first_count,
  sum(
    case
      when race_entries.final_rank = 2 then 1
      else 0
    end
  ) as jockey_second_count,
  sum(
    case
      when race_entries.final_rank = 3 then 1
      else 0
    end
  ) as jockey_third_count,
  sum(
    (6 - LEAST(race_entries.final_rank, 5))::numeric * 1.0
  ) / (15.0 * count(*)::numeric) as jockey_performance
from
  race_entries
group by
  race_entries.trainer_id;