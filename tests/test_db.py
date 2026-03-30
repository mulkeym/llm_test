import os
import tempfile
import pytest
from src.db import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    database.close()
    os.unlink(path)


def test_create_run(db):
    run_id = db.create_run("test-model", {"tiers": [1, 2, 3]})
    assert run_id is not None
    assert isinstance(run_id, int)


def test_save_result(db):
    run_id = db.create_run("test-model", {})
    result_id = db.save_result(
        run_id=run_id,
        test_name="test_basic",
        category="tool_calling",
        tier=1,
        score=8.0,
        max_score=10.0,
        transcript=[{"role": "user", "content": "test"}],
    )
    assert result_id is not None


def test_save_judgement(db):
    run_id = db.create_run("test-model", {})
    result_id = db.save_result(run_id, "test_q", "knowledge", 1, 7.5, 10.0, [])
    db.save_judgement(result_id, "accuracy", 8, "Good answer")
    judgements = db.get_judgements(result_id)
    assert len(judgements) == 1
    assert judgements[0]["criterion"] == "accuracy"


def test_update_run_score(db):
    run_id = db.create_run("test-model", {})
    db.update_run_score(run_id, 85.5)
    run = db.get_run(run_id)
    assert run["composite_score"] == 85.5


def test_get_all_runs(db):
    db.create_run("model-a", {})
    db.create_run("model-b", {})
    runs = db.get_all_runs()
    assert len(runs) == 2


def test_get_results_by_run(db):
    run_id = db.create_run("test-model", {})
    db.save_result(run_id, "t1", "tool_calling", 1, 5.0, 10.0, [])
    db.save_result(run_id, "t2", "knowledge", 1, 8.0, 10.0, [])
    results = db.get_results_by_run(run_id)
    assert len(results) == 2


def test_get_results_by_category(db):
    run_id = db.create_run("test-model", {})
    db.save_result(run_id, "t1", "tool_calling", 1, 5.0, 10.0, [])
    db.save_result(run_id, "t2", "knowledge", 1, 8.0, 10.0, [])
    results = db.get_results_by_run(run_id, category="tool_calling")
    assert len(results) == 1
