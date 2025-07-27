-- 인덱스들 먼저 삭제
DROP INDEX IF EXISTS idx_race_analysis_date;
DROP INDEX IF EXISTS idx_race_analysis_horse_date;
DROP INDEX IF EXISTS idx_race_analysis_rank;
DROP INDEX IF EXISTS idx_race_analysis_date_rank;

-- Materialized View 삭제
DROP MATERIALIZED VIEW IF EXISTS race_analysis_complete;


-- meet_code 조건이 추가된 새로운 Materialized View
CREATE MATERIALIZED VIEW race_analysis_complete AS
SELECT 
    re.race_id,
    re.horse_id,
    re.race_date,
    re.meet_code,
    re.entry_number,
    re.horse_weight,
    re.final_rank,
    CASE WHEN re.final_rank = 1 THEN 1 ELSE 0 END as is_winner,
    
    -- 말 정보
    h.age as horse_age,
    CASE WHEN h.gender = '수컷' THEN 1 ELSE 0 END as is_male,
    h.rank as horse_class,
    h.name as horse_name,
    
    -- 경주 정보
    r.race_distance,
    r.total_horses,
    r.planned_horses,
    r.race_grade,
    r.track_condition,
    r.weather,
    r.weight_type,
    
    -- 말 과거 성적
    hbs.prev_total_races,
    hbs.prev_5_avg_rank,
    hbs.prev_total_avg_rank,
    hbs.prev_wins,
    hbs.prev_top3,
    
    -- 기수 정보
    j.total_races as jockey_total_races,
    j.total_wins as jockey_total_wins,
    j.year_races as jockey_year_races,
    j.year_wins as jockey_year_wins,
    
    -- 조교사 정보
    t.rc_cnt_t as trainer_total_races,
    t.ord1_cnt_t as trainer_total_wins,
    t.rc_cnt_y as trainer_year_races,
    t.ord1_cnt_y as trainer_year_wins,
    
    -- 거리별 성적
    dp.avg_rank_at_distance,
    dp.races_at_distance

FROM race_entries re
JOIN horses h ON re.horse_id = h.horse_id
JOIN races r ON re.race_id = r.race_id 
    AND re.race_date = r.race_date 
    AND re.meet_code = r.meet_code  -- 🎯 이 부분이 추가됨
LEFT JOIN horse_basic_stats hbs ON re.horse_id = hbs.horse_id AND re.race_date = hbs.race_date
LEFT JOIN jockeys j ON re.jk_no = j.jk_no
LEFT JOIN trainers t ON re.trainer_id = t.trainer_id
LEFT JOIN distance_performance dp ON re.horse_id = dp.horse_id AND r.race_distance = dp.race_distance
WHERE re.final_rank IS NOT NULL;



-- 성능을 위한 인덱스들 재생성
CREATE INDEX idx_race_analysis_date ON race_analysis_complete(race_date);
CREATE INDEX idx_race_analysis_horse_date ON race_analysis_complete(horse_id, race_date);
CREATE INDEX idx_race_analysis_rank ON race_analysis_complete(final_rank);
CREATE INDEX idx_race_analysis_date_rank ON race_analysis_complete(race_date, final_rank);

-- meet_code 관련 인덱스도 추가 (필요하다면)
CREATE INDEX idx_race_analysis_meet_code ON race_analysis_complete(meet_code);
CREATE INDEX idx_race_analysis_date_meet ON race_analysis_complete(race_date, meet_code);