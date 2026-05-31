create materialized view public.race_analysis_complete_mater as
with
  distance_percentiles as (
    select
      r_1.race_distance,
      percentile_cont(0.75::double precision) within group (
        order by
          (re_1.race_time::double precision)
      ) as p75_race_time
    from
      races r_1
      join race_entries re_1 on r_1.race_id = re_1.race_id
      and r_1.race_date = re_1.race_date
      and r_1.meet_code = re_1.meet_code
    where
      re_1.race_time is not null
      and re_1.race_time > 0::numeric
    group by
      r_1.race_distance
  )
select
  re.race_id,
  re.horse_id,
  re.race_date,
  re.meet_code,
  re.entry_number,
  case
    when re.horse_weight = 0 then COALESCE(
      avg(re.horse_weight) filter (
        where
          re.horse_weight > 0
      ) over (
        partition by
          re.horse_id
        order by
          re.race_date rows between 3 PRECEDING
          and 1 PRECEDING
      ),
      re.horse_weight::numeric
    )
    else re.horse_weight::numeric
  end as horse_weight,
  re.final_rank,
  re.finish_time,
  re.horse_race_days,
  re.horse_weight_diff,
  re.budam,
  re.budam_weight,
  re.budam_weight - lag(re.budam_weight) over (
    order by
      re.race_date
  ) as budam_weight_diff,
  COALESCE(re.horse_rating, 31::numeric) as horse_rating,
  case
    when re.final_rank = 1 then 1
    else 0
  end as is_winner,
  case
    when re.final_rank <= 3 then 1
    else 0
  end as is_top3,
  h.age as horse_age,
  case
    when h.gender = '수'::text then 1
    else 0
  end as is_male,
  h.rank as horse_class,
  h.name as horse_name,
  r.race_distance,
  r.total_horses,
  r.planned_horses,
  r.race_grade,
  r.track_condition,
  r.weather,
  r.weight_type,
  hbs.prev_total_races,
  hbs.prev_5_avg_rank,
  hbs.prev_total_avg_rank,
  count(*) filter (
    where
      re.final_rank > 0
  ) over (
    partition by
      re.horse_id,
      re.jk_no
    order by
      re.race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_jockey_total_races,
  count(*) filter (
    where
      re.final_rank = 1
  ) over (
    partition by
      re.horse_id,
      re.jk_no
    order by
      re.race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_jockey_wins,
  count(*) filter (
    where
      re.final_rank <= 3
  ) over (
    partition by
      re.horse_id,
      re.jk_no
    order by
      re.race_date rows between UNBOUNDED PRECEDING
      and 1 PRECEDING
  ) as horse_jockey_top3,
  COALESCE(
    avg(
      case
        when COALESCE(re.race_time, 0::numeric) > 0::numeric then r.race_distance::numeric / re.race_time
        else 15.7
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between 5 PRECEDING
        and 1 PRECEDING
    ),
    15.7
  ) as prev_5_avg_speed_mps,
  COALESCE(
    avg(
      case
        when COALESCE(re.race_time, 0::numeric) > 0::numeric then r.race_distance::numeric / re.race_time
        else 15.7
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between 3 PRECEDING
        and 1 PRECEDING
    ),
    15.7
  ) as prev_3_avg_speed_mps,
  COALESCE(
    avg(
      case
        when COALESCE(re.race_time, 0::numeric) > 0::numeric then r.race_distance::numeric / re.race_time
        else 0::numeric
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between 3 PRECEDING
        and 1 PRECEDING
    ) - avg(
      case
        when COALESCE(re.race_time, 0::numeric) > 0::numeric then r.race_distance::numeric / re.race_time
        else 0::numeric
      end
    ) over (
      partition by
        re.horse_id
      order by
        re.race_date rows between 6 PRECEDING
        and 4 PRECEDING
    ),
    0::numeric
  ) as speed_improvement_trend,
  COALESCE(
    avg(re.race_time) over (
      partition by
        re.horse_id,
        r.race_distance
      order by
        re.race_date rows between 3 PRECEDING
        and 1 PRECEDING
    )::double precision,
    dp_fallback.p75_race_time
  ) as prev_3_avg_time_same_distance,
  COALESCE(
    lag(
      case
        when re.final_rank <= 3 then 1
        else 0
      end
    ) over (
      partition by
        re.horse_id,
        r.race_distance
      order by
        re.race_date
    ),
    0
  ) as prev_race_was_top3,
  COALESCE(
    lag(re.race_time) over (
      partition by
        re.horse_id,
        r.race_distance
      order by
        re.race_date
    )::double precision,
    dp_fallback.p75_race_time
  ) as prev_race_time,
  hbs.prev_wins,
  hbs.prev_top3,
  hbs.prev_top5,
  hbs.year_wins,
  hbs.year_top3,
  hbs.year_top5,
  hbs.total_races,
  hbs.total_win_rate,
  hbs.total_place_rate,
  hbs.year_races,
  hbs.year_win_rate,
  hbs.year_place_rate,
  j.total_races as jockey_total_races,
  j.total_wins as jockey_total_wins,
  j.year_races as jockey_year_races,
  j.year_wins as jockey_year_wins,
  t.rc_cnt_t as trainer_total_races,
  t.ord1_cnt_t as trainer_total_wins,
  t.rc_cnt_y as trainer_year_races,
  t.ord1_cnt_y as trainer_year_wins,
  COALESCE(dp.avg_rank_at_distance, 9::numeric) as avg_rank_at_distance,
  COALESCE(dp.races_at_distance, 0::bigint) as races_at_distance
from
  race_entries re
  join horses h on re.horse_id = h.horse_id
  join races r on re.race_id = r.race_id
  and re.race_date = r.race_date
  and re.meet_code = r.meet_code
  left join distance_percentiles dp_fallback on r.race_distance = dp_fallback.race_distance
  left join horse_basic_stats hbs on re.horse_id = hbs.horse_id
  and re.race_date = hbs.race_date
  left join jockeys j on re.jk_no = j.jk_no
  left join trainers t on re.trainer_id = t.trainer_id::text
  left join distance_performance dp on re.horse_id = dp.horse_id
  and r.race_distance = dp.race_distance;