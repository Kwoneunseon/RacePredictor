-- PostgREST prepared statement가 generic plan으로 바뀌면 범위 파라미터를 못 읽어
-- 페이지마다 전체 행을 조인·정렬한다(약 3초 = anon statement_timeout) - 매번 실제 값으로 계획하게 한다.
ALTER ROLE anon SET plan_cache_mode = 'force_custom_plan';
NOTIFY pgrst, 'reload config';
