"""
SQLite persistence.

Two rules govern everything in this file.

**A database failure must never stop the belt.** Sorting waste is the job;
recording that it happened is bookkeeping. So every write is guarded, a
failure disables persistence and logs once rather than raising on every
subsequent item, and the Monitoring Agent keeps its live counters in memory
regardless of whether the disk is cooperating. A system that stops sorting
because a file is locked has its priorities backwards.

**Readers must not block the writer.** The Streamlit reporting view is a
separate process reading this same file while the sorting loop writes to it.
WAL mode is what makes that safe, and it is one line.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.config import Config
from core.messages import Event, WasteRecord

log = logging.getLogger("Database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    controller  TEXT,
    source      TEXT,
    frames      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER REFERENCES sessions(id),
    timestamp   TEXT    NOT NULL,
    track_id    INTEGER,
    object      TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    material    TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    destination TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    safety_rule TEXT    NOT NULL,
    reason      TEXT,
    latency_ms  REAL
);
CREATE INDEX IF NOT EXISTS idx_detections_time     ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_category ON detections(category);
CREATE INDEX IF NOT EXISTS idx_detections_rule     ON detections(safety_rule);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER REFERENCES sessions(id),
    timestamp   TEXT NOT NULL,
    agent       TEXT NOT NULL,
    event       TEXT NOT NULL,
    severity    TEXT NOT NULL,
    message     TEXT,
    payload     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_time     ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);

CREATE TABLE IF NOT EXISTS bin_status (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER REFERENCES sessions(id),
    bin           TEXT    NOT NULL,
    capacity      INTEGER NOT NULL,
    current_level INTEGER NOT NULL,
    status        TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bin_time ON bin_status(updated_at);
"""

# Events worth keeping forever. The per-frame chatter is already in the JSONL
# log; the database is for the record of what the system decided and when it
# complained, not for every heartbeat.
PERSISTED_EVENTS = {
    "WASTE_RECORDED", "SAFETY_VERDICT", "BIN_ALERT", "SYSTEM_FAULT",
    "SYSTEM_START", "SYSTEM_UPDATE", "SORT_EXECUTED",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, cfg: Config, read_only: bool = False):
        self.enabled = bool(cfg.get("database.enabled", True))
        self.path: Path = cfg.path("database.path", "data/waste.db")
        self.conn: sqlite3.Connection | None = None
        self.session_id: int | None = None
        self.read_only = read_only
        self.available = False          # can this connection WRITE?
        self._write_failures = 0

        if not self.enabled:
            log.info("Persistence disabled by config")
            return

        # The reporting view opens the same file read-only, so it can be left
        # open across a shift without any possibility of disturbing the line.
        if read_only:
            if not self.path.exists():
                return
            try:
                self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
                self.conn.row_factory = sqlite3.Row
            except sqlite3.Error:
                log.exception("Could not open %s read-only", self.path)
                self.conn = None
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # WAL lets the Streamlit view read while the sorting loop writes.
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self._migrate()
            self.conn.commit()
            self.available = True
            log.info("Database ready at %s", self.path)
        except sqlite3.Error:
            log.exception("Could not open %s; running without persistence", self.path)
            self.conn = None

    @property
    def readable(self) -> bool:
        return self.conn is not None

    # -- migration ---------------------------------------------------------

    # Columns added after the first release. CREATE TABLE IF NOT EXISTS does
    # nothing to a table that already exists, so a database written by an
    # earlier stage needs the new columns added rather than assumed.
    ADDED_COLUMNS = {
        "detections": [("route", "TEXT")],
    }

    def _migrate(self) -> None:
        if self.conn is None:
            return
        for table, columns in self.ADDED_COLUMNS.items():
            try:
                existing = {r["name"] for r in
                            self.conn.execute(f"PRAGMA table_info({table})")}
            except sqlite3.Error:
                continue
            if not existing:            # table absent; SCHEMA will have made it
                continue
            for name, sql_type in columns:
                if name in existing:
                    continue
                try:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
                    log.info("Migrated %s: added column %s", table, name)
                except sqlite3.Error:
                    log.exception("Could not add %s.%s", table, name)

    # -- guarded write -----------------------------------------------------

    def _write(self, sql: str, params: Iterable[Any]) -> int | None:
        """Every write goes through here, and none of them can raise.

        After the first failure persistence switches off for the run. Logging
        one exception per item for a locked file would fill the log with the
        same message and hide the events that matter.
        """
        if not self.available or self.conn is None:
            return None
        try:
            cursor = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error:
            self._write_failures += 1
            self.available = False
            log.exception(
                "Database write failed; persistence disabled for this run "
                "(the sorting loop is unaffected)"
            )
            return None

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        if self.conn is None:
            return []
        try:
            return list(self.conn.execute(sql, tuple(params)))
        except sqlite3.Error:
            log.exception("Query failed: %s", sql.strip().split("\n")[0])
            return []

    # -- sessions ----------------------------------------------------------

    def start_session(self, controller: str, source: str) -> int | None:
        self.session_id = self._write(
            "INSERT INTO sessions (started_at, controller, source) VALUES (?, ?, ?)",
            (_now(), controller, source),
        )
        return self.session_id

    def end_session(self, frames: int) -> None:
        if self.session_id is None:
            return
        self._write("UPDATE sessions SET ended_at = ?, frames = ? WHERE id = ?",
                    (_now(), frames, self.session_id))

    # -- writes ------------------------------------------------------------

    def insert_detection(self, record: WasteRecord) -> None:
        self._write(
            """INSERT INTO detections
               (session_id, timestamp, track_id, object, confidence, material,
                category, action, destination, status, safety_rule, reason,
                latency_ms, route)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session_id, record.timestamp, record.track_id, record.label,
             record.detection_confidence, record.material.value,
             record.category.value, record.action.value, record.destination.value,
             record.status.value, record.safety_rule, record.reason,
             record.latency_ms, record.route),
        )

    def insert_event(self, event: Event) -> None:
        if event.event not in PERSISTED_EVENTS:
            return
        payload = event.to_dict()["payload"]
        message = payload.get("reason") or payload.get("detail") or ""
        self._write(
            """INSERT INTO events
               (session_id, timestamp, agent, event, severity, message, payload)
               VALUES (?,?,?,?,?,?,?)""",
            (self.session_id, event.timestamp, event.agent, event.event,
             event.severity.value, str(message), json.dumps(payload)),
        )

    def snapshot_bins(self, bins, at: str | None = None) -> None:
        """`at` lets a simulated run stamp its own clock.

        The live system leaves it alone and gets wall time. A scripted run
        compresses minutes of belt time into a second of real time, so
        without this every sample would carry the same timestamp and the
        fill-over-time chart would collapse to a single point.
        """
        stamp = at or _now()
        for bin_ in bins.bins.values():
            self._write(
                """INSERT INTO bin_status
                   (session_id, bin, capacity, current_level, status, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (self.session_id, bin_.destination.value, bin_.capacity,
                 bin_.current_level, bin_.status, stamp),
            )

    # -- reads (the reporting view and, in Stage 4, the Analytics Agent) ----

    def totals_by_category(self, since: str | None = None) -> list[sqlite3.Row]:
        where = "WHERE timestamp >= ?" if since else ""
        return self.query(
            f"""SELECT category, COUNT(*) AS count, AVG(confidence) AS avg_confidence
                FROM detections {where}
                GROUP BY category ORDER BY count DESC""",
            (since,) if since else (),
        )

    def totals_by_rule(self, since: str | None = None) -> list[sqlite3.Row]:
        where = "WHERE timestamp >= ?" if since else ""
        return self.query(
            f"""SELECT safety_rule, COUNT(*) AS count
                FROM detections {where}
                GROUP BY safety_rule ORDER BY count DESC""",
            (since,) if since else (),
        )

    def daily_totals(self) -> list[sqlite3.Row]:
        return self.query(
            """SELECT substr(timestamp, 1, 10) AS day,
                      COUNT(*) AS processed,
                      SUM(CASE WHEN action = 'MANUAL_CHECK' THEN 1 ELSE 0 END) AS manual,
                      AVG(confidence) AS avg_confidence
               FROM detections GROUP BY day ORDER BY day"""
        )

    def recent_detections(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,))

    def recent_alerts(self, limit: int = 25) -> list[sqlite3.Row]:
        return self.query(
            """SELECT * FROM events
               WHERE severity IN ('WARNING', 'ALERT', 'CRITICAL')
               ORDER BY id DESC LIMIT ?""", (limit,))

    def bin_history(self, limit: int = 500) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM bin_status ORDER BY id DESC LIMIT ?", (limit,))

    def sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,))

    # -- windowed reads, for the Analytics Agent ---------------------------
    #
    # All three take an optional [since, until) window on the timestamp, so
    # one period can be compared with the one before it. Passing neither means
    # "everything on record".

    @staticmethod
    def _window(since: str | None, until: str | None) -> tuple[str, tuple]:
        clauses, params = [], []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp < ?")
            params.append(until)
        return ("WHERE " + " AND ".join(clauses) if clauses else ""), tuple(params)

    def window_summary(self, since: str | None = None,
                       until: str | None = None) -> dict[str, Any]:
        where, params = self._window(since, until)
        rows = self.query(
            f"""SELECT COUNT(*) AS processed,
                       AVG(confidence) AS avg_confidence,
                       SUM(CASE WHEN action = 'MANUAL_CHECK' THEN 1 ELSE 0 END) AS manual,
                       SUM(CASE WHEN status != 'OK' THEN 1 ELSE 0 END) AS failures
                FROM detections {where}""", params)
        row = rows[0] if rows else None
        processed = (row["processed"] if row else 0) or 0
        manual = (row["manual"] if row else 0) or 0
        return {
            "processed": processed,
            "manual": manual,
            "failures": (row["failures"] if row else 0) or 0,
            "avg_confidence": (row["avg_confidence"] if row else 0.0) or 0.0,
            "manual_rate": (manual / processed) if processed else 0.0,
        }

    def category_counts(self, since: str | None = None,
                        until: str | None = None) -> dict[str, int]:
        where, params = self._window(since, until)
        return {r["category"]: r["count"] for r in self.query(
            f"""SELECT category, COUNT(*) AS count FROM detections {where}
                GROUP BY category""", params)}

    def rule_counts(self, since: str | None = None,
                    until: str | None = None) -> dict[str, int]:
        where, params = self._window(since, until)
        return {r["safety_rule"]: r["count"] for r in self.query(
            f"""SELECT safety_rule, COUNT(*) AS count FROM detections {where}
                GROUP BY safety_rule""", params)}

    def days(self) -> list[str]:
        """Every day that has at least one decision, oldest first."""
        return [r["day"] for r in self.query(
            """SELECT DISTINCT substr(timestamp, 1, 10) AS day
               FROM detections ORDER BY day""")]

    def latest_bin_levels(self) -> dict[str, dict]:
        """The most recent sample for each bin."""
        rows = self.query(
            """SELECT b.bin, b.capacity, b.current_level, b.status, b.updated_at
               FROM bin_status b
               JOIN (SELECT bin, MAX(id) AS top FROM bin_status GROUP BY bin) latest
                 ON b.id = latest.top""")
        return {
            r["bin"]: {
                "capacity": r["capacity"], "level": r["current_level"],
                "status": r["status"], "updated_at": r["updated_at"],
                "fraction": (r["current_level"] / r["capacity"]) if r["capacity"] else 0.0,
            }
            for r in rows
        }

    def summary(self) -> dict[str, Any]:
        rows = self.query(
            """SELECT COUNT(*) AS processed,
                      AVG(confidence) AS avg_confidence,
                      SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) AS sorted_ok,
                      SUM(CASE WHEN action = 'MANUAL_CHECK' THEN 1 ELSE 0 END) AS manual
               FROM detections"""
        )
        if not rows:
            return {"processed": 0, "avg_confidence": 0.0, "sorted_ok": 0, "manual": 0}
        row = rows[0]
        return {
            "processed": row["processed"] or 0,
            "avg_confidence": row["avg_confidence"] or 0.0,
            "sorted_ok": row["sorted_ok"] or 0,
            "manual": row["manual"] or 0,
        }

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
            self.conn = None
            self.available = False
