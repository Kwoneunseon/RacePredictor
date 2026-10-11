from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from library import collecting_race_results
from core.history_rebuilder import rebuild_after_collect
from Utils.database import get_client
from Utils.logger import get_logger

logger = get_logger(__name__)


def morning_task():
    """매일 10:00 — 오늘 경주 예측 후 카카오톡 전송"""
    from core.predictor import predict_today, format_kakao_message
    from Utils.kakaotalk import send_message

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"아침 예측 시작: {today}")

    result = predict_today(today)
    if result is None:
        logger.info(f"{today} 경주 데이터 없음 — 카카오톡 생략")
        return

    msg = format_kakao_message(result, today)
    logger.info(f"예측 메시지 생성 완료 ({result['predictions']['race_id'].nunique()}경주)")

    ok = send_message(msg)
    if ok:
        logger.info("카카오톡 전송 완료")
    else:
        logger.warning("카카오톡 전송 실패 — 로그 확인 필요")


def _get_today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def collect_task():
    today = _get_today_str()
    logger.info(f"데이터 수집 시작: {today}")

    client = get_client()
    log_row = {
        "job_type": "collect",
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    log_id = None
    try:
        log_res = client.table("scheduler_logs").insert(log_row).execute()
        log_id = log_res.data[0]["id"] if log_res.data else None
    except Exception:
        pass

    try:
        first_date = collecting_race_results(
            start_page=1,
            max_pages=50,
            start_date=today,
            end_date=today,
        )
        rebuild_after_collect(first_date)

        # 마체중 스크래핑 (horse_weight=0인 항목 보완)
        from library.scraper_weights import update_weights_to_db
        today_dash = datetime.now().strftime("%Y-%m-%d")
        for meet_code in [1, 2, 3]:
            update_weights_to_db(meet_code, today_dash)

        status, error_msg = "success", None
    except Exception as e:
        status, error_msg = "error", str(e)
        logger.error(f"수집 실패: {e}")

    if log_id:
        try:
            client.table("scheduler_logs").update({
                "finished_at": datetime.now().isoformat(),
                "status": status,
                "error_message": error_msg,
            }).eq("id", log_id).execute()
        except Exception:
            pass

    logger.info(f"데이터 수집 완료: {status}")


def create_scheduler(background: bool = False):
    Scheduler = BackgroundScheduler if background else BlockingScheduler
    sched = Scheduler(timezone="Asia/Seoul")

    sched.add_job(
        collect_task,
        trigger=IntervalTrigger(hours=1),
        id="collect_race_data",
        name="경주일 시간별 수집",
        replace_existing=True,
    )

    sched.add_job(
        collect_task,
        trigger=CronTrigger(day_of_week="mon-fri", hour=6, minute=0),
        id="collect_weekday_data",
        name="평일 1일 1회 수집",
        replace_existing=True,
    )

    sched.add_job(
        morning_task,
        trigger=CronTrigger(hour=10, minute=0),
        id="morning_predict",
        name="매일 10시 예측 + 카카오톡",
        replace_existing=True,
    )

    return sched


if __name__ == "__main__":
    logger.info("스케줄러 시작")
    sched = create_scheduler(background=False)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료")
