-- =============================================
-- 003_race_payouts.sql
-- =============================================
-- 경주 확정(정산) 배당 저장 — 단승/연승/복승/쌍승/복연승/삼복승/삼쌍승.
-- library/scraper_score.py의 fetch_score_info()/update_race_payouts_to_db()가
-- ScoreInfo.do 응답에서 파싱한 결과를 이 스키마로 upsert한다
-- ([[project_race_payouts_scraper]] 참고). 경주 종료 후 확정된 배당이며
-- 실시간 배당이 아니다 — 백테스트/EV 캘리브레이션용
-- ([[project_multi_bet_probability_model]] Harville 모델 Phase 2 블로커 해소).
--
-- 좁고 긴(narrow/long) 구조를 택한 이유: 마권 종류별로 콤보 개수(1~3두)와
-- 그룹 수(단승 1개 ~ 연승/복연승 3개)가 달라 와이드 테이블로 만들면 대부분
-- 컬럼이 NULL이 된다. 경주당 최대 11행(1+3+1+1+3+1+1)뿐이라 조회 성능
-- 문제도 없다.
-- =============================================

CREATE TABLE IF NOT EXISTS race_payouts (
    id              BIGSERIAL PRIMARY KEY,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    bet_type        TEXT NOT NULL CHECK (bet_type IN
                        ('win', 'place', 'quinella', 'exacta', 'quinella_place', 'trio', 'trifecta')),
    -- 파싱 순서 그대로 저장(원문 표기 순서). exacta/trifecta는 이 순서가
    -- 곧 착순이라 순서 보존이 필수. quinella/quinella_place/trio는 착순
    -- 무관 마권이지만 원문 순서를 그대로 저장해도 무방 — 사이트가 동일
    -- 경주를 재조회해도 같은 순서로 렌더링하는 한 UNIQUE 제약과 upsert
    -- on_conflict가 안정적으로 동작한다(현재 스크래퍼가 순서무관 결과를
    -- 재정렬하지 않는 점을 인지하고 유지: 순서가 재현되지 않는 사례가
    -- 발견되면 조회 전 정렬을 추가로 검토할 것).
    entry_numbers   INT[] NOT NULL CHECK (array_length(entry_numbers, 1) BETWEEN 1 AND 3),
    odds            FLOAT NOT NULL CHECK (odds > 0),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(race_id, bet_type, entry_numbers)
);

-- 주의: UNIQUE(race_id, bet_type, entry_numbers) 제약이 자동 생성하는
-- btree 인덱스가 이미 (race_id) 단독 및 (race_id, bet_type) 선행 컬럼
-- 조회를 커버한다. 별도 idx_race_payouts_race_bet(race_id, bet_type)
-- 인덱스는 이 unique 인덱스와 완전히 중복되므로 추가하지 않는다.

-- Harville 백테스트에서 predictions/betting_records를 race_id로 조인하는
-- 패턴이 늘어날 것에 대비해 인덱스 추가. predictions는 현재 코드베이스
-- 어디서도 실제로 INSERT되지 않는 미사용 테이블이지만(quant-model-expert
-- 검토 필요 항목), 인덱스 자체는 비용이 거의 없고 향후 예측 영속화를
-- 시작하면 바로 필요해진다.
CREATE INDEX IF NOT EXISTS idx_predictions_race_id ON predictions(race_id);
CREATE INDEX IF NOT EXISTS idx_betting_records_race_id ON betting_records(race_id);
