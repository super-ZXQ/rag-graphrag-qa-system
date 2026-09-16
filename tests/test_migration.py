"""增量迁移安全测试：旧库必须平滑升级，且 papers/chunks/FTS5 不被破坏。"""
from __future__ import annotations

import sqlite3

from paper_library.store import PaperStore


def _build_legacy_database(path) -> None:
    """模拟早期版本的库：缺少后加列、没有 FTS5，但已有论文与分块数据。"""
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE papers (
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
            CREATE TABLE paper_pages (
                paper_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (paper_id, page_number)
            );
            CREATE TABLE paper_chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE research_tasks (
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
            """
        )
        connection.execute(
            """INSERT INTO papers
               (paper_id, content_hash, filename, title, page_count, status,
                error_message, stored_path, created_at)
               VALUES ('legacy-1', ?, 'old.md', 'Old Doc', 1, 'ready', NULL, ?, 't0')""",
            ("h" * 64, "data/old.md"),
        )
        connection.execute(
            "INSERT INTO paper_chunks (chunk_id, paper_id, page_number, text) VALUES (?, ?, ?, ?)",
            ("legacy-1:p1:c0", "legacy-1", 1, "sensitive long document incremental indexing traceable"),
        )
        connection.commit()
    finally:
        connection.close()


def test_legacy_database_migrates_without_data_loss(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    _build_legacy_database(db_path)

    store = PaperStore(db_path)  # __init__ 触发增量迁移

    paper = store.get_paper("legacy-1")
    assert paper is not None
    assert paper["title"] == "Old Doc"
    assert paper["status"] == "ready"
    # 后加列必须存在并有默认值，而不是抛错
    assert paper["source_type"] == "internal_upload"
    assert paper["visibility"] == "internal"

    chunks = store.list_chunks("legacy-1")
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("sensitive long document")

    # 旧库已有的分块必须在迁移后进入 FTS5，可被关键词检索到
    results = store.search_lexical("incremental")
    assert any(item["chunk_id"] == "legacy-1:p1:c0" for item in results)


def test_reinitialize_does_not_duplicate_or_wipe_fts(tmp_path):
    db_path = tmp_path / "twice.sqlite3"
    store = PaperStore(db_path)
    paper = {
        "paper_id": "p-1",
        "content_hash": "c" * 64,
        "filename": "a.md",
        "title": "A",
        "stored_path": str(tmp_path / "a.md"),
    }
    store.save_processed_paper(
        paper,
        ["incremental indexing keeps evidence"],
        [{"chunk_id": "p-1:p1:c0", "page_number": 1, "text": "incremental indexing keeps evidence"}],
    )
    store.update_paper_status("p-1", "ready", index_version="v1")

    # 再次初始化（模拟重启），不应清空数据或重复 FTS 记录
    PaperStore(db_path)
    PaperStore(db_path)

    results = store.search_lexical("evidence")
    assert len(results) == 1
    assert results[0]["chunk_id"] == "p-1:p1:c0"


def test_blank_database_starts_cleanly(tmp_path):
    store = PaperStore(tmp_path / "blank.sqlite3")
    assert store.list_papers() == []
    assert store.search_lexical("anything") == []


def test_legacy_global_evidence_key_migrates_to_task_scoped_key(tmp_path):
    db_path = tmp_path / "evidence.sqlite3"
    store = PaperStore(db_path)
    task = {
        "question": "选择企业检索方案",
        "requirements": {},
        "candidates": ["RAPTOR", "CRAG"],
        "weights": {"质量": 100},
    }
    store.create_task({"task_id": "task-1", **task})
    store.create_task({"task_id": "task-2", **task})

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            ALTER TABLE evidence RENAME TO evidence_new;
            CREATE TABLE evidence (
                evidence_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES research_tasks(task_id) ON DELETE CASCADE,
                paper_id TEXT NOT NULL, chunk_id TEXT NOT NULL, page_number INTEGER NOT NULL,
                title TEXT NOT NULL, text TEXT NOT NULL, source_uri TEXT,
                score REAL NOT NULL, created_at TEXT NOT NULL
            );
            INSERT INTO evidence VALUES
                ('E-shared', 'task-1', 'p', 'p:p1:c0', 1, 'title', 'text', NULL, 1, 't0');
            DROP TABLE evidence_new;
            """
        )
        connection.commit()
    finally:
        connection.close()

    migrated = PaperStore(db_path)
    migrated.save_evidence(
        "task-2",
        [{
            "evidence_id": "E-shared", "paper_id": "p", "chunk_id": "p:p1:c0",
            "page_number": 1, "title": "title", "text": "text", "score": 1,
        }],
    )

    assert len(migrated.list_evidence("task-1")) == 1
    assert len(migrated.list_evidence("task-2")) == 1
