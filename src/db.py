import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional


class Database:
    def __init__(self, path: str = "results.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                config_snapshot TEXT,
                composite_score REAL
            );
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                test_name TEXT NOT NULL,
                category TEXT NOT NULL,
                tier INTEGER NOT NULL,
                score REAL NOT NULL,
                max_score REAL NOT NULL,
                transcript TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
            CREATE TABLE IF NOT EXISTS judgements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER NOT NULL,
                criterion TEXT NOT NULL,
                score REAL NOT NULL,
                explanation TEXT,
                FOREIGN KEY (result_id) REFERENCES results(id)
            );
        """)

    def create_run(self, model_name: str, config: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (model_name, timestamp, config_snapshot) VALUES (?, ?, ?)",
            (model_name, datetime.now(timezone.utc).isoformat(), json.dumps(config)),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_run_score(self, run_id: int, composite_score: float):
        self.conn.execute(
            "UPDATE runs SET composite_score = ? WHERE id = ?",
            (composite_score, run_id),
        )
        self.conn.commit()

    def save_result(
        self,
        run_id: int,
        test_name: str,
        category: str,
        tier: int,
        score: float,
        max_score: float,
        transcript: list,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO results (run_id, test_name, category, tier, score, max_score, transcript) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, test_name, category, tier, score, max_score, json.dumps(transcript)),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_judgement(self, result_id: int, criterion: str, score: float, explanation: str):
        self.conn.execute(
            "INSERT INTO judgements (result_id, criterion, score, explanation) VALUES (?, ?, ?, ?)",
            (result_id, criterion, score, explanation),
        )
        self.conn.commit()

    def get_run(self, run_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_all_runs(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY timestamp DESC").fetchall()
        return [dict(r) for r in rows]

    def get_results_by_run(self, run_id: int, category: Optional[str] = None) -> List[dict]:
        if category:
            rows = self.conn.execute(
                "SELECT * FROM results WHERE run_id = ? AND category = ?", (run_id, category)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_judgements(self, result_id: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM judgements WHERE result_id = ?", (result_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
