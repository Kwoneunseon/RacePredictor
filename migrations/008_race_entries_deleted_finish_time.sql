-- =============================================
-- 008_race_entries_deleted_finish_time.sql
-- =============================================
-- 정상 착순(1~89)인데 완주기록이 없는 행도 deleted = 1 로 표시한다.
-- 미완주(90번대)·결과 미기입(0)의 기록 부재는 정당한 결측이라 제외.
-- 생성식 변경은 PG 17+ 문법(SET EXPRESSION), 테이블 재작성이 일어난다.
-- =============================================

-- 라이브에만 있던 컬럼을 마이그레이션에 기록 (이미 있으면 no-op)
ALTER TABLE race_entries
    ADD COLUMN IF NOT EXISTS finish_time DOUBLE PRECISION;

ALTER TABLE race_entries
    ALTER COLUMN deleted SET EXPRESSION AS (
        (CASE
            WHEN win_odds IS NULL OR win_odds = 0 OR win_odds >= 1000 THEN 1
            WHEN final_rank BETWEEN 1 AND 89 AND COALESCE(finish_time, 0) = 0 THEN 1
            ELSE 0
        END)::SMALLINT
    );
