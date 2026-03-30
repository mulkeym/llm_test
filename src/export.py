import csv
import json
from typing import Optional
from .db import Database


def export_run(db: Database, run_id: int, fmt: str, output: Optional[str] = None):
    run = db.get_run(run_id)
    results = db.get_results_by_run(run_id)

    if fmt == "json":
        data = {
            "model_name": run["model_name"],
            "timestamp": run["timestamp"],
            "composite_score": run["composite_score"],
            "results": [
                {
                    "test_name": r["test_name"],
                    "category": r["category"],
                    "tier": r["tier"],
                    "score": r["score"],
                    "max_score": r["max_score"],
                    "percentage": round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0,
                }
                for r in results
            ],
        }
        path = output or "export_{}_{}.json".format(run["model_name"], run_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print("Exported to {}".format(path))

    elif fmt == "csv":
        path = output or "export_{}_{}.csv".format(run["model_name"], run_id)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "test_name", "category", "tier", "score", "max_score", "percentage"])
            for r in results:
                pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
                writer.writerow([run["model_name"], r["test_name"], r["category"], r["tier"], r["score"], r["max_score"], pct])
        print("Exported to {}".format(path))


def export_all(db: Database, fmt: str, output: Optional[str] = None):
    runs = db.get_all_runs()
    if fmt == "json":
        data = []
        for run in runs:
            results = db.get_results_by_run(run["id"])
            data.append({
                "model_name": run["model_name"],
                "timestamp": run["timestamp"],
                "composite_score": run["composite_score"],
                "results": [dict(r) for r in results],
            })
        path = output or "export_all.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print("Exported to {}".format(path))

    elif fmt == "csv":
        path = output or "export_all.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "timestamp", "composite_score", "test_name", "category", "tier", "score", "max_score", "percentage"])
            for run in runs:
                results = db.get_results_by_run(run["id"])
                for r in results:
                    pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
                    writer.writerow([run["model_name"], run["timestamp"], run["composite_score"], r["test_name"], r["category"], r["tier"], r["score"], r["max_score"], pct])
        print("Exported to {}".format(path))
