"""SQLite persistence for papers, evidence and research tasks."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from config import LIBRARY_DB


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PaperStore:
    """Small SQLite repository with additive, startup-safe migrations."""

    def __init__(self, database_path: Path = LIBRARY_DB):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _add_column(connection: sqlite3.Connection, table: str, definition: str) -> None:
        name = definition.split()[0]
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if name not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    def initialize(self) -> None:
        with self._connection() as connection:
            fts_existed = bool(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_chunks_fts'"
                ).fetchone()
            )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    stored_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_pages (
                    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY (paper_id, page_number)
                );
                CREATE TABLE IF NOT EXISTS paper_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper_id ON paper_chunks(paper_id);
                CREATE TABLE IF NOT EXISTS research_tasks (
                    task_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    requirements_json TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_step TEXT,
                    error_message TEXT,
                    report_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES research_tasks(task_id) ON DELETE CASCADE,
                    step TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT NOT NULL,
                    task_id TEXT NOT NULL REFERENCES research_tasks(task_id) ON DELETE CASCADE,
                    paper_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_uri TEXT,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, evidence_id)
                );
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE REFERENCES research_tasks(task_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    reviewer_note TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS report_versions (
                    report_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES research_tasks(task_id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    reviewer_note TEXT,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (task_id, version)
                );
                CREATE TABLE IF NOT EXISTS claims (
                    report_id TEXT NOT NULL,
                    claim_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    claim_text TEXT NOT NULL,
                    claim_type TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    support_status TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    risk_note TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (report_id, claim_id)
                );
                """
            )
            for definition in (
                "source_type TEXT NOT NULL DEFAULT 'internal_upload'",
                "source_uri TEXT",
                "visibility TEXT NOT NULL DEFAULT 'internal'",
                "index_version TEXT",
                "indexed_at TEXT",
                "updated_at TEXT",
            ):
                self._add_column(connection, "papers", definition)
            for definition in (
                "cancel_requested INTEGER NOT NULL DEFAULT 0",
                "failed_step TEXT",
                "error_type TEXT",
                "budget_json TEXT NOT NULL DEFAULT '{}'",
                "revision_note TEXT",
            ):
                self._add_column(connection, "research_tasks", definition)
            self._migrate_evidence_primary_key(connection)
            connection.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS paper_chunks_fts USING fts5(
                    text, content='paper_chunks', content_rowid='rowid'
                );
                CREATE TRIGGER IF NOT EXISTS paper_chunks_ai AFTER INSERT ON paper_chunks BEGIN
                    INSERT INTO paper_chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS paper_chunks_ad AFTER DELETE ON paper_chunks BEGIN
                    INSERT INTO paper_chunks_fts(paper_chunks_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS paper_chunks_au AFTER UPDATE ON paper_chunks BEGIN
                    INSERT INTO paper_chunks_fts(paper_chunks_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                    INSERT INTO paper_chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END;
                """
            )
            connection.execute("INSERT OR IGNORE INTO schema_version VALUES (1, ?)", (_now(),))
            connection.execute("INSERT OR IGNORE INTO schema_version VALUES (2, ?)", (_now(),))
            # 新库由触发器实时同步 FTS；仅当 FTS 表是本次新建、且内容表已有分块（旧库升级）
            # 时做一次性回填，避免每次启动全量 rebuild，也保证旧库关键词检索可用。
            chunk_count = connection.execute("SELECT COUNT(*) FROM paper_chunks").fetchone()[0]
            if not fts_existed and chunk_count:
                connection.execute("INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES ('rebuild')")
            # 一次性把旧的单版本报告迁移到版本表，保证旧库升级后历史报告不丢。
            legacy_versions = connection.execute(
                """SELECT r.report_id, r.task_id, r.status, r.markdown, r.validation_json,
                          r.reviewer_note, r.version, r.created_at, r.updated_at
                   FROM reports r
                   WHERE NOT EXISTS (
                       SELECT 1 FROM report_versions v WHERE v.task_id = r.task_id
                   )"""
            ).fetchall()
            for row in legacy_versions:
                evidence_ids = json.dumps([], ensure_ascii=False)
                connection.execute(
                    """INSERT OR IGNORE INTO report_versions
                       (report_id, task_id, version, status, markdown, validation_json,
                        reviewer_note, evidence_ids_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["report_id"], row["task_id"], row["version"], row["status"],
                        row["markdown"], row["validation_json"], row["reviewer_note"],
                        evidence_ids, row["created_at"], row["updated_at"],
                    ),
                )

    @staticmethod
    def _migrate_evidence_primary_key(connection: sqlite3.Connection) -> None:
        """Upgrade legacy evidence(evidence_id PK) to task-scoped evidence IDs."""
        columns = connection.execute("PRAGMA table_info(evidence)").fetchall()
        primary_key = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
        if primary_key == ["task_id", "evidence_id"]:
            return
        connection.executescript(
            """
            ALTER TABLE evidence RENAME TO evidence_legacy;
            CREATE TABLE evidence (
                evidence_id TEXT NOT NULL,
                task_id TEXT NOT NULL REFERENCES research_tasks(task_id) ON DELETE CASCADE,
                paper_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                source_uri TEXT,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (task_id, evidence_id)
            );
            INSERT INTO evidence (
                evidence_id, task_id, paper_id, chunk_id, page_number, title,
                text, source_uri, score, created_at
            )
            SELECT evidence_id, task_id, paper_id, chunk_id, page_number, title,
                   text, source_uri, score, created_at
            FROM evidence_legacy;
            DROP TABLE evidence_legacy;
            """
        )

    def get_by_hash(self, content_hash: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return dict(row) if row else None

    def get_paper(self, paper_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def update_paper_title(self, paper_id: str, title: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE papers SET title = ?, updated_at = ? WHERE paper_id = ?",
                (title[:240], _now(), paper_id),
            )

    def list_papers(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, paper_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM paper_chunks WHERE paper_id = ? ORDER BY page_number, chunk_id",
                (paper_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_paper_status(
        self, paper_id: str, status: str, error_message: str | None = None,
        index_version: str | None = None,
    ) -> None:
        indexed_at = _now() if status == "ready" else None
        with self._connection() as connection:
            connection.execute(
                """UPDATE papers
                   SET status = ?, error_message = ?, index_version = COALESCE(?, index_version),
                       indexed_at = COALESCE(?, indexed_at), updated_at = ?
                   WHERE paper_id = ?""",
                (status, error_message, index_version, indexed_at, _now(), paper_id),
            )

    def save_processed_paper(self, paper: dict, pages: list[str], chunks: list[dict]) -> dict:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO papers (
                    paper_id, content_hash, filename, title, page_count, status,
                    error_message, stored_path, created_at, updated_at, source_type,
                    source_uri, visibility
                ) VALUES (?, ?, ?, ?, ?, 'parsed', NULL, ?, ?, ?, ?, ?, ?)""",
                (
                    paper["paper_id"], paper["content_hash"], paper["filename"], paper["title"],
                    len(pages), paper["stored_path"], now, now,
                    paper.get("source_type", "internal_upload"), paper.get("source_uri"),
                    paper.get("visibility", "internal"),
                ),
            )
            connection.executemany(
                "INSERT INTO paper_pages (paper_id, page_number, text) VALUES (?, ?, ?)",
                [(paper["paper_id"], index, text) for index, text in enumerate(pages, 1)],
            )
            connection.executemany(
                "INSERT INTO paper_chunks (chunk_id, paper_id, page_number, text) VALUES (?, ?, ?, ?)",
                [(c["chunk_id"], paper["paper_id"], c["page_number"], c["text"]) for c in chunks],
            )
        return self.get_by_hash(paper["content_hash"]) or paper

    def save_failed_paper(self, paper: dict, error_message: str) -> dict:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO papers (
                    paper_id, content_hash, filename, title, page_count, status,
                    error_message, stored_path, created_at, updated_at, source_type,
                    source_uri, visibility
                ) VALUES (?, ?, ?, ?, 0, 'failed', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper["paper_id"], paper["content_hash"], paper["filename"], paper["title"],
                    error_message, paper["stored_path"], now, now,
                    paper.get("source_type", "internal_upload"), paper.get("source_uri"),
                    paper.get("visibility", "internal"),
                ),
            )
        return self.get_by_hash(paper["content_hash"]) or paper

    def search_lexical(self, query: str, limit: int = 8) -> list[dict]:
        tokens = [token.replace('"', "") for token in query.split() if token.strip()]
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens[:12])
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT c.chunk_id, c.paper_id, c.page_number, c.text, p.title,
                          p.filename, p.source_type, p.source_uri, p.visibility,
                          bm25(paper_chunks_fts) AS rank
                   FROM paper_chunks_fts f
                   JOIN paper_chunks c ON c.rowid = f.rowid
                   JOIN papers p ON p.paper_id = c.paper_id
                   WHERE paper_chunks_fts MATCH ? AND p.status = 'ready'
                   ORDER BY rank LIMIT ?""",
                (expression, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_task(self, task: dict) -> dict:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO research_tasks (
                    task_id, question, requirements_json, candidates_json, weights_json,
                    status, current_step, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', 'queued', ?, ?)""",
                (
                    task["task_id"], task["question"], json.dumps(task["requirements"], ensure_ascii=False),
                    json.dumps(task["candidates"], ensure_ascii=False),
                    json.dumps(task["weights"], ensure_ascii=False), now, now,
                ),
            )
        return self.get_task(task["task_id"]) or task

    def get_task(self, task_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM research_tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            task = dict(row)
            events = connection.execute(
                "SELECT * FROM research_events WHERE task_id = ? ORDER BY event_id", (task_id,)
            ).fetchall()
        for field in ("requirements_json", "candidates_json", "weights_json"):
            task[field.removesuffix("_json")] = json.loads(task.pop(field))
        task["budget"] = json.loads(task.pop("budget_json") or "{}")
        task["events"] = [dict(event) for event in events]
        return task

    def active_task_exists(self) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM research_tasks WHERE status IN ('queued', 'running') LIMIT 1"
            ).fetchone()
        return row is not None

    def recover_interrupted_tasks(self) -> int:
        """Make process-local background work recoverable after an API restart."""
        now = _now()
        with self._connection() as connection:
            interrupted = connection.execute(
                """SELECT task_id, COALESCE(current_step, 'normalize_requirements') AS step
                   FROM research_tasks WHERE status IN ('queued', 'running')"""
            ).fetchall()
            for row in interrupted:
                connection.execute(
                    """UPDATE research_tasks
                       SET status='failed', failed_step=?, error_type='ProcessInterrupted',
                           error_message='API 进程中断，可从保留状态恢复。', updated_at=?
                       WHERE task_id=?""",
                    (row["step"], now, row["task_id"]),
                )
                connection.execute(
                    """INSERT INTO research_events
                       (task_id, step, event_type, summary, details_json, created_at)
                       VALUES (?, ?, 'interrupted', '检测到进程中断，任务已转为可恢复失败状态', '{}', ?)""",
                    (row["task_id"], row["step"], now),
                )
        return len(interrupted)

    def update_task(self, task_id: str, status: str, step: str, error: str | None = None) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE research_tasks SET status=?, current_step=?, error_message=?, updated_at=? WHERE task_id=?",
                (status, step, error, _now(), task_id),
            )

    def request_cancel(self, task_id: str) -> dict | None:
        """记录取消请求；queued 任务立即终止，running 任务在节点边界退出。"""
        task = self.get_task(task_id)
        if not task or task["status"] not in ("queued", "running"):
            return task
        with self._connection() as connection:
            if task["status"] == "queued":
                connection.execute(
                    """UPDATE research_tasks
                       SET status='cancelled', current_step='cancelled', cancel_requested=1,
                           updated_at=? WHERE task_id=?""",
                    (_now(), task_id),
                )
            else:
                connection.execute(
                    "UPDATE research_tasks SET cancel_requested=1, updated_at=? WHERE task_id=?",
                    (_now(), task_id),
                )
        self.add_event(task_id, task["current_step"] or "cancel", "cancel_requested", "评审人请求取消研究任务")
        return self.get_task(task_id)

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM research_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return bool(row and row[0])

    def mark_cancelled(self, task_id: str, step: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """UPDATE research_tasks
                   SET status='cancelled', current_step='cancelled', cancel_requested=1,
                       updated_at=? WHERE task_id=?""",
                (_now(), task_id),
            )
        self.add_event(task_id, step, "cancelled", "研究任务已取消，已保留证据与事件")

    def record_failure(self, task_id: str, step: str, error_type: str, message: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """UPDATE research_tasks
                   SET status='failed', current_step=?, failed_step=?, error_type=?,
                       error_message=?, updated_at=? WHERE task_id=?""",
                (step, step, error_type[:120], message[:500], _now(), task_id),
            )
        self.add_event(
            task_id, step, "failed", f"节点 {step} 失败：{error_type}",
            {"error_type": error_type, "message": message[:500]},
        )

    def reset_for_retry(self, task_id: str, entry_step: str, note: str | None = None) -> None:
        """失败恢复或修订时把任务重置为运行态，保留证据与历史报告。"""
        with self._connection() as connection:
            connection.execute(
                """UPDATE research_tasks
                   SET status='running', current_step=?, cancel_requested=0, failed_step=NULL,
                       error_type=NULL, error_message=NULL, revision_note=?, updated_at=?
                   WHERE task_id=?""",
                (entry_step, note, _now(), task_id),
            )

    def update_budget(self, task_id: str, budget: dict) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE research_tasks SET budget_json=?, updated_at=? WHERE task_id=?",
                (json.dumps(budget, ensure_ascii=False), _now(), task_id),
            )

    def begin_revision(self, task_id: str, reviewer_note: str) -> dict | None:
        """把待审批任务打回重做：保留旧版本，从报告起草节点重新执行。"""
        task = self.get_task(task_id)
        if not task or task["status"] != "awaiting_approval":
            return task
        self.reset_for_retry(task_id, "draft_report", reviewer_note)
        self.add_event(
            task_id, "draft_report", "revision_requested",
            "评审人要求修订报告，将复用已冻结证据重新起草",
            {"reviewer_note": reviewer_note},
        )
        return self.get_task(task_id)

    def add_event(
        self, task_id: str, step: str, event_type: str, summary: str,
        details: dict | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO research_events
                   (task_id, step, event_type, summary, details_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (task_id, step, event_type, summary, json.dumps(details or {}, ensure_ascii=False), _now()),
            )

    def save_evidence(self, task_id: str, items: list[dict]) -> None:
        now = _now()
        with self._connection() as connection:
            connection.executemany(
                """INSERT INTO evidence
                   (evidence_id, task_id, paper_id, chunk_id, page_number, title, text,
                    source_uri, score, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_id, evidence_id) DO UPDATE SET
                       paper_id=excluded.paper_id, chunk_id=excluded.chunk_id,
                       page_number=excluded.page_number, title=excluded.title,
                       text=excluded.text, source_uri=excluded.source_uri,
                       score=excluded.score, created_at=excluded.created_at""",
                [
                    (
                        item["evidence_id"], task_id, item["paper_id"], item["chunk_id"],
                        item["page_number"], item["title"], item["text"],
                        item.get("source_uri"), item["score"], now,
                    )
                    for item in items
                ],
            )

    def list_evidence(self, task_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE task_id = ? ORDER BY score DESC", (task_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def save_report(
        self, report: dict, evidence_ids: list[str] | None = None,
        reviewer_note: str | None = None, claims: list[dict] | None = None,
    ) -> dict:
        """保存新版本报告；历史版本标记为 superseded，任务回到待审批。"""
        now = _now()
        with self._connection() as connection:
            version_row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM report_versions WHERE task_id=?",
                (report["task_id"],),
            ).fetchone()
            version = (version_row["v"] or 0) + 1
            connection.execute(
                "UPDATE report_versions SET status='superseded', updated_at=? WHERE task_id=? AND status='draft'",
                (now, report["task_id"]),
            )
            connection.execute(
                """INSERT INTO report_versions
                   (report_id, task_id, version, status, markdown, validation_json,
                    reviewer_note, evidence_ids_json, created_at, updated_at)
                   VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?)""",
                (
                    report["report_id"], report["task_id"], version, report["markdown"],
                    json.dumps(report["validation"], ensure_ascii=False), reviewer_note,
                    json.dumps(evidence_ids or [], ensure_ascii=False), now, now,
                ),
            )
            if claims:
                connection.executemany(
                    """INSERT INTO claims
                       (report_id, claim_id, task_id, version, claim_text, claim_type,
                        evidence_ids_json, support_status, confidence, risk_note, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            report["report_id"], claim["claim_id"], report["task_id"], version,
                            claim["claim_text"], claim["claim_type"],
                            json.dumps(claim.get("evidence_ids", []), ensure_ascii=False),
                            claim["support_status"], float(claim.get("confidence", 0.0)),
                            claim.get("risk_note", ""), now,
                        )
                        for claim in claims
                    ],
                )
            connection.execute(
                """UPDATE research_tasks SET status='awaiting_approval',
                   current_step='await_human_approval', report_id=?, failed_step=NULL,
                   error_type=NULL, error_message=NULL, updated_at=? WHERE task_id=?""",
                (report["report_id"], now, report["task_id"]),
            )
        return self.get_report(report["task_id"]) or report

    def list_report_versions(self, task_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM report_versions WHERE task_id=? ORDER BY version", (task_id,)
            ).fetchall()
        reports = []
        for row in rows:
            item = dict(row)
            item["validation"] = json.loads(item.pop("validation_json"))
            item["evidence_ids"] = json.loads(item.pop("evidence_ids_json") or "[]")
            reports.append(item)
        return reports

    def get_report(self, task_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM report_versions WHERE task_id=? ORDER BY version DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if not row:
                return None
            claim_rows = connection.execute(
                "SELECT * FROM claims WHERE report_id=? ORDER BY claim_id", (row["report_id"],)
            ).fetchall()
        report = dict(row)
        report["validation"] = json.loads(report.pop("validation_json"))
        report["evidence_ids"] = json.loads(report.pop("evidence_ids_json") or "[]")
        report["claims"] = []
        for claim_row in claim_rows:
            claim = dict(claim_row)
            claim["evidence_ids"] = json.loads(claim.pop("evidence_ids_json") or "[]")
            report["claims"].append(claim)
        return report

    def approve_report(self, task_id: str, reviewer_note: str | None = None) -> dict | None:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """UPDATE report_versions
                   SET status='approved', reviewer_note=COALESCE(?, reviewer_note), updated_at=?
                   WHERE task_id=? AND version=(
                       SELECT MAX(version) FROM report_versions WHERE task_id=?)""",
                (reviewer_note, now, task_id, task_id),
            )
            connection.execute(
                """UPDATE research_tasks SET status='completed', current_step='finalized',
                   updated_at=? WHERE task_id=?""",
                (now, task_id),
            )
        self.add_event(task_id, "finalized", "approved", "报告已由人工确认，成为最终版本")
        return self.get_report(task_id)
