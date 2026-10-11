-- =============================================
-- 007_horses_birth_year_gelded_since.sql
-- =============================================
-- 2026-10-11. 말 기본정보 API(API8_2)는 현역 말만 돌려줘서 은퇴마의
-- gender/age가 비어 있다. 경주결과 API(API4_3)의 age/sex로 채우기 위한 컬럼.
--
--   birth_year   : 출생연도 = 경주 연도 - 경주 당시 나이
--   gelded_since : '거'로 처음 관측된 경주일(실제 거세일이 아니라 관측 상한).
--                  경주 당시 성별은 race_date >= gelded_since 면 '거'로 복원.
--
-- 둘 다 NULL 허용, 재실행 안전.
-- =============================================

ALTER TABLE horses
    ADD COLUMN IF NOT EXISTS birth_year SMALLINT,
    ADD COLUMN IF NOT EXISTS gelded_since DATE;
