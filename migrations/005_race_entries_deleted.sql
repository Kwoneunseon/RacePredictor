-- =============================================
-- 005_race_entries_deleted.sql
-- =============================================
-- 2026-10-11 사용자 지시로 추가. race_entries.win_odds가 "쓸 수 없는 배당"인
-- 행을 표시하는 플래그 컬럼. 학습/백테스트는 deleted = 0 만 쓰도록 한다.
--
--   deleted = 1 : win_odds IS NULL 또는 win_odds = 0 또는 win_odds >= 1000
--   deleted = 0 : 그 외
--
-- * win_odds = 0  : 수집기 safe_float가 결측을 0.0으로 저장한 값(배당 없음).
-- * win_odds >= 1000 : 잠정 배당 자리 채움 값. 9999.9뿐 아니라 1338.4 같은
--   값도 있어 값 일치(= 9999.9) 대신 범위 조건으로 잡는다. 적용 시점 기준
--   1000 미만 최대값은 944, 1000 이상은 9999.9/1338.4/1294 세 종류뿐이었다.
-- * win_odds는 double precision이라 NaN이 들어오면 NaN >= 1000 이 참이 되어
--   1로 떨어진다(Postgres는 NaN을 모든 수보다 크게 취급).
--
-- 일반 컬럼이 아니라 STORED 생성 컬럼인 이유: win_odds는 재수집/백필로 계속
-- 바뀌는데(0 -> 실제 배당 등), 일반 컬럼이면 그때마다 따로 갱신해야 하고
-- 어긋나기 쉽다. 생성 컬럼은 win_odds가 바뀌면 DB가 자동으로 다시 계산한다.
--
-- 주의: 생성 컬럼은 INSERT/UPDATE/upsert 페이로드에 값을 실으면 에러가 난다
-- (428C9 / "cannot insert a non-DEFAULT value into column"). race_entries를
-- select("*")로 읽은 행을 그대로 다시 upsert하지 말 것 - deleted 키를 빼야 한다.
--
-- ADD COLUMN IF NOT EXISTS라 재실행해도 안전하다(이미 있으면 no-op).
-- 테이블 재작성이 일어나지만 적용 시점 60,775행이라 부담 없다.
-- =============================================

ALTER TABLE race_entries
    ADD COLUMN IF NOT EXISTS deleted SMALLINT NOT NULL
    GENERATED ALWAYS AS (
        (CASE
            WHEN win_odds IS NULL OR win_odds = 0 OR win_odds >= 1000 THEN 1
            ELSE 0
        END)::SMALLINT
    ) STORED;
