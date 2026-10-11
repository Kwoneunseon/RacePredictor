import pytest
from unittest.mock import MagicMock, patch


def test_scheduler_has_collect_job():
    from scheduler import create_scheduler
    sched = create_scheduler(background=True)
    job_ids = [job.id for job in sched.get_jobs()]
    assert "collect_race_data" in job_ids


def test_scheduler_has_weekday_job():
    from scheduler import create_scheduler
    sched = create_scheduler(background=True)
    job_ids = [job.id for job in sched.get_jobs()]
    assert "collect_weekday_data" in job_ids


def test_collect_job_is_interval():
    from scheduler import create_scheduler
    from apscheduler.triggers.interval import IntervalTrigger
    sched = create_scheduler(background=True)
    job = next(j for j in sched.get_jobs() if j.id == "collect_race_data")
    assert isinstance(job.trigger, IntervalTrigger)


def test_collect_task_calls_collect(mocker):
    mocker.patch("scheduler.collecting_race_results")
    mocker.patch("scheduler.rebuild_after_collect")
    mock_client = MagicMock()
    mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[{"id": 1}])
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    mocker.patch("scheduler.get_client", return_value=mock_client)
    from scheduler import collect_task
    collect_task()


def test_collect_task_rebuilds_history_from_first_collected_date(mocker):
    mocker.patch("scheduler.collecting_race_results", return_value="2026-10-01")
    mocker.patch("library.scraper_weights.update_weights_to_db")
    mock_rebuild = mocker.patch("scheduler.rebuild_after_collect")
    mocker.patch("scheduler.get_client", return_value=MagicMock())
    from scheduler import collect_task
    collect_task()
    mock_rebuild.assert_called_once_with("2026-10-01")
