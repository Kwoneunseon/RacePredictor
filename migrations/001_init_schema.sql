-- =============================================
-- 경마 자동 투자 봇 - 새 스키마 (001_init_schema)
-- Supabase 프로젝트: kjctbohokxdzxrsrmwlh
-- =============================================

-- 말 기본 정보
CREATE TABLE IF NOT EXISTS horses (
    horse_id    TEXT PRIMARY KEY,
    name        TEXT,
    age         INT,
    gender      TEXT,
    rank        TEXT,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 경주 정보
CREATE TABLE IF NOT EXISTS races (
    race_id         TEXT PRIMARY KEY,
    race_date       DATE NOT NULL,
    meet_code       INT NOT NULL,
    race_no         INT,
    race_distance   INT,
    race_grade      TEXT,
    track_condition TEXT,
    weather         TEXT,
    total_horses    INT
);

-- 기수
CREATE TABLE IF NOT EXISTS jockeys (
    jk_no       TEXT PRIMARY KEY,
    name        TEXT,
    total_races INT DEFAULT 0,
    total_wins  INT DEFAULT 0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 조교사
CREATE TABLE IF NOT EXISTS trainers (
    trainer_id  TEXT PRIMARY KEY,
    name        TEXT,
    rc_cnt_t    INT DEFAULT 0,
    ord1_cnt_t  INT DEFAULT 0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 출전 기록 (핵심 조인 테이블)
CREATE TABLE IF NOT EXISTS race_entries (
    id              BIGSERIAL PRIMARY KEY,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    horse_id        TEXT NOT NULL REFERENCES horses(horse_id),
    race_date       DATE NOT NULL,
    entry_number    INT,
    horse_weight    INT,
    jk_no           TEXT REFERENCES jockeys(jk_no),
    trainer_id      TEXT REFERENCES trainers(trainer_id),
    final_rank      INT,
    win_odds        FLOAT,
    UNIQUE(race_id, horse_id)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_race_entries_date ON race_entries(race_date);
CREATE INDEX IF NOT EXISTS idx_race_entries_horse ON race_entries(horse_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);

-- 말별 성적 캐시
CREATE TABLE IF NOT EXISTS horse_race_history (
    horse_id            TEXT NOT NULL REFERENCES horses(horse_id),
    as_of_date          DATE NOT NULL,
    prev_total_races    INT DEFAULT 0,
    prev_wins           INT DEFAULT 0,
    prev_top3           INT DEFAULT 0,
    prev_5_avg_rank     FLOAT DEFAULT 6,
    prev_total_avg_rank FLOAT DEFAULT 6,
    PRIMARY KEY (horse_id, as_of_date)
);

-- 예측 결과
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    horse_id        TEXT NOT NULL REFERENCES horses(horse_id),
    predicted_rank  INT,
    win_probability FLOAT,
    model_version   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    actual_rank     INT
);

-- 베팅 기록
CREATE TABLE IF NOT EXISTS betting_records (
    id          BIGSERIAL PRIMARY KEY,
    race_id     TEXT NOT NULL REFERENCES races(race_id),
    horse_id    TEXT NOT NULL REFERENCES horses(horse_id),
    bet_amount  INT NOT NULL,
    bet_type    TEXT NOT NULL,
    odds        FLOAT,
    profit      INT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 스케줄러 실행 로그
CREATE TABLE IF NOT EXISTS scheduler_logs (
    id                  BIGSERIAL PRIMARY KEY,
    job_type            TEXT NOT NULL,
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              TEXT,
    records_processed   INT DEFAULT 0,
    error_message       TEXT
);
