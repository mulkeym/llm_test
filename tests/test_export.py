import json
import os
import tempfile
import pytest
from src.db import Database
from src.export import export_run


@pytest.fixture
def db_with_data():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    run_id = db.create_run("test-model", {"tiers": [1]})
    db.save_result(run_id, "test1", "tool_calling", 1, 8.0, 10.0, [{"role": "user", "content": "hi"}])
    db.save_result(run_id, "test2", "knowledge", 1, 7.5, 10.0, [])
    db.update_run_score(run_id, 77.5)
    yield db, run_id, path
    db.close()
    os.unlink(path)


def test_export_json(db_with_data, tmp_path):
    db, run_id, _ = db_with_data
    outfile = str(tmp_path / "results.json")
    export_run(db, run_id, "json", outfile)
    with open(outfile) as f:
        data = json.load(f)
    assert data["model_name"] == "test-model"
    assert len(data["results"]) == 2


def test_export_csv(db_with_data, tmp_path):
    db, run_id, _ = db_with_data
    outfile = str(tmp_path / "results.csv")
    export_run(db, run_id, "csv", outfile)
    with open(outfile) as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + 2 rows
