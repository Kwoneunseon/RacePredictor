-- =============================================
-- 002_retroactive_new_signal_tables.sql
-- =============================================
-- 2026-07-19~20 신규 신호 피처 롤아웃(부담중량/구간기록/러닝스타일,
-- [[project_new_signal_features_rollout]] 메모리 참고) 작업 중 마이그레이션
-- 파일 없이 직접 라이브 Supabase(kjctbohokxdzxrsrmwlh)에 생성된 테이블
-- 3개를 사후 문서화한다.
--
-- *** 실질적인 스키마 변경은 없다 *** — 아래 3개 테이블은 이미 운영 DB에
-- 존재하며(2026-07-20 백필 완료), 이 파일은 001과 동일한 idempotent 스타일
-- (CREATE TABLE IF NOT EXISTS)로 재실행해도 no-op이다. 목적은 오직
-- migrations/ 디렉터리만 봐도 라이브 스키마를 재현할 수 있게 맞추는 것.
--
-- 최종 결론: 이 3개 테이블이 제공하는 신규 피처(부담중량/구간기록/
-- 러닝스타일)는 408경주 백테스트에서 개선 미확인으로 프로덕션 모델에서는
-- 전부 제외됐다(config.py 주석 처리). 그러나 재검토 가능성을 남기기 위해
-- 수집기·테이블·feature_builder 조인 로직은 코드베이스에 그대로 남아있다.
-- =============================================

-- 말별 구간기록(스피드 프로파일) 스냅샷 캐시.
-- 주의: horses(horse_id) FK가 의도적으로 없다 — 백필 시점에 horse_id가
-- horses 테이블에 항상 먼저 존재한다는 보장이 없었기 때문. 조인 시
-- PostgREST 관계 임베딩(races(...)/horses(...) 자동 조인)은 FK 부재로
-- PGRST200 에러가 나므로 반드시 수동 2단계 조회 사용할 것.
CREATE TABLE IF NOT EXISTS horse_section_records (
    horse_id            TEXT NOT NULL,
    race_distance        INT NOT NULL,
    snapshot_race_date   DATE NOT NULL,
    s1f_best  NUMERIC, s1f_avg  NUMERIC, s1f_worst  NUMERIC,
    c1_best   NUMERIC, c1_avg   NUMERIC, c1_worst   NUMERIC,
    c2_best   NUMERIC, c2_avg   NUMERIC, c2_worst   NUMERIC,
    c3_best   NUMERIC, c3_avg   NUMERIC, c3_worst   NUMERIC,
    c4_best   NUMERIC, c4_avg   NUMERIC, c4_worst   NUMERIC,
    g3f_best  NUMERIC, g3f_avg  NUMERIC, g3f_worst  NUMERIC,
    g1f_best  NUMERIC, g1f_avg  NUMERIC, g1f_worst  NUMERIC,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (horse_id, race_distance, snapshot_race_date)
);

-- race_entries 확장 상세: 부담중량 + 말/기수/조교사 통산·최근1년 성적 +
-- 통과순위(1~4코너). FK 없는 이유는 위 horse_section_records와 동일.
CREATE TABLE IF NOT EXISTS race_entry_detail (
    race_id                 TEXT NOT NULL,
    horse_id                TEXT NOT NULL,
    assigned_weight         NUMERIC,
    horse_career_wins       INT,
    horse_career_starts     INT,
    horse_recent1y_wins     INT,
    horse_recent1y_starts   INT,
    jockey_career_wins      INT,
    jockey_career_starts    INT,
    jockey_recent1y_wins    INT,
    jockey_recent1y_starts  INT,
    trainer_career_wins     INT,
    trainer_career_starts   INT,
    trainer_recent1y_wins   INT,
    trainer_recent1y_starts INT,
    sj_1c_ord INT, sj_2c_ord INT, sj_3c_ord INT, sj_4c_ord INT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (race_id, horse_id)
);

-- 조교(훈련) 세션 로그. horse_id가 아닌 말 이름(hrnm) 텍스트로 식별 —
-- horses FK 없음 (원본 API가 마번을 안 주고 이름만 줌).
CREATE TABLE IF NOT EXISTS horse_training_sessions (
    id            BIGSERIAL PRIMARY KEY,
    trng_dt       DATE NOT NULL,
    hrnm          TEXT NOT NULL,
    belo_no       TEXT,
    belo_trng_no  TEXT,
    ridr_nm       TEXT,
    remk_txt      TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trng_dt, hrnm, belo_trng_no)
);
CREATE INDEX IF NOT EXISTS idx_hts_hrnm ON horse_training_sessions(hrnm);
CREATE INDEX IF NOT EXISTS idx_hts_trng_dt ON horse_training_sessions(trng_dt);
