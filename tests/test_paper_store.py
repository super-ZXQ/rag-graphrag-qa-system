from paper_library.store import PaperStore


def _task(task_id):
    return {
        "task_id": task_id,
        "question": "选择企业检索方案",
        "requirements": {},
        "candidates": ["RAPTOR", "CRAG"],
        "weights": {"质量": 100},
    }


def test_processed_paper_keeps_page_and_chunk_provenance(tmp_path):
    store = PaperStore(tmp_path / "library.sqlite3")
    paper = {
        "paper_id": "local-123",
        "content_hash": "a" * 64,
        "filename": "example.pdf",
        "title": "Example paper",
        "stored_path": str(tmp_path / "example.pdf"),
    }

    saved = store.save_processed_paper(
        paper,
        ["first page", "second page"],
        [
            {"chunk_id": "local-123:p1:c0", "page_number": 1, "text": "first"},
            {"chunk_id": "local-123:p2:c0", "page_number": 2, "text": "second"},
        ],
    )

    assert saved["status"] == "parsed"
    assert saved["page_count"] == 2
    assert store.get_by_hash("a" * 64)["paper_id"] == "local-123"


def test_fts_search_only_returns_ready_papers(tmp_path):
    store = PaperStore(tmp_path / "library.sqlite3")
    paper = {
        "paper_id": "local-fts",
        "content_hash": "b" * 64,
        "filename": "requirements.md",
        "title": "Requirements",
        "stored_path": str(tmp_path / "requirements.md"),
    }
    store.save_processed_paper(
        paper,
        ["sensitive documents require traceable citations"],
        [{"chunk_id": "local-fts:p1:c0", "page_number": 1, "text": "sensitive documents require traceable citations"}],
    )
    assert store.search_lexical("traceable") == []

    store.update_paper_status("local-fts", "ready", index_version="test")

    results = store.search_lexical("traceable")
    assert results[0]["chunk_id"] == "local-fts:p1:c0"
    assert results[0]["filename"] == "requirements.md"


def test_same_evidence_is_scoped_to_each_task(tmp_path):
    store = PaperStore(tmp_path / "library.sqlite3")
    store.create_task(_task("task-1"))
    store.create_task(_task("task-2"))
    evidence = {
        "evidence_id": "E-shared",
        "paper_id": "paper-1",
        "chunk_id": "paper-1:p1:c0",
        "page_number": 1,
        "title": "Shared evidence",
        "text": "same chunk",
        "score": 0.8,
    }

    store.save_evidence("task-1", [evidence])
    store.save_evidence("task-2", [evidence])

    assert [item["evidence_id"] for item in store.list_evidence("task-1")] == ["E-shared"]
    assert [item["evidence_id"] for item in store.list_evidence("task-2")] == ["E-shared"]


def test_recover_interrupted_tasks_makes_them_resumable(tmp_path):
    store = PaperStore(tmp_path / "library.sqlite3")
    store.create_task(_task("task-running"))
    store.update_task("task-running", "running", "draft_report")

    assert store.recover_interrupted_tasks() == 1

    task = store.get_task("task-running")
    assert task["status"] == "failed"
    assert task["failed_step"] == "draft_report"
    assert task["error_type"] == "ProcessInterrupted"
    assert task["events"][-1]["event_type"] == "interrupted"
