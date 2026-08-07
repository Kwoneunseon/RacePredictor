-- =============================================
-- 004_betting_records_bet_type_check.sql
-- =============================================
-- 2026-07-30 성능/무결성 점검(db-expert) 중 적용. betting_records.bet_type이
-- race_payouts.bet_type과 동일한 영문 snake_case("win"/"place") 어휘로
-- 통일되기로 결정됐는데([[project_predictions_bettype_dnf_decisions]] 참고,
-- core/ev_calculator.py 등 코드 변경 완료) 정작 컬럼에는 CHECK 제약이 없어
-- 값 드리프트를 막을 방법이 없었다. betting_records가 현재 0행(미사용
-- 테이블)이라 백필 없이 안전하게 추가 가능.
--
-- ALTER TABLE ... ADD CONSTRAINT는 IF NOT EXISTS 문법이 없으므로(Postgres
-- 미지원) DO 블록으로 idempotent하게 감싼다.
-- =============================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'betting_records_bet_type_check'
    ) THEN
        ALTER TABLE betting_records
            ADD CONSTRAINT betting_records_bet_type_check CHECK (bet_type IN ('win', 'place'));
    END IF;
END $$;
