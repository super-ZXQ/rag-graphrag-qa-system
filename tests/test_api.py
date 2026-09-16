from fastapi.testclient import TestClient

import api.app as api_app


WEIGHTS = {
    "检索质量与可追溯性": 25,
    "在线延迟": 15,
    "数据更新复杂度": 15,
    "实施成本": 15,
    "运行成本": 10,
    "隐私与部署适配": 10,
    "运维与团队适配": 10,
}


class _Msg:
    content = "# Report\n\nThe method improves retrieval quality [E-paper-1-p2-c0]."
    usage_metadata = None


class _Retriever:
    def search(self, query, limit):
        return [
            {
                "chunk_id": "paper-1:p2:c0",
                "paper_id": "paper-1",
                "page_number": 2,
                "title": "RAG Evidence",
                "text": "The method improves retrieval quality.",
                "source_uri": "https://example.test/paper-1",
                "score": 0.5,
            }
        ]


class _Model:
    def __init__(self, fail=False):
        self.fail = fail

    def invoke(self, messages):
        if self.fail:
            raise RuntimeError("model down")
        return _Msg()


def _isolate(monkeypatch, tmp_path, model=None):
    from paper_library.store import PaperStore
    from research.workflow import ResearchWorkflow

    store = PaperStore(tmp_path / "api.sqlite3")
    store.save_processed_paper(
        {
            "paper_id": "paper-1",
            "content_hash": "p" * 64,
            "filename": "paper-1.md",
            "title": "RAG Evidence",
            "stored_path": "paper-1.md",
            "source_type": "public_arxiv",
            "source_uri": "https://example.test/paper-1",
            "visibility": "public",
        },
        ["", "The method improves retrieval quality."],
        [{"chunk_id": "paper-1:p2:c0", "page_number": 2,
          "text": "The method improves retrieval quality."}],
    )
    store.update_paper_status("paper-1", "ready", index_version="test")
    monkeypatch.setattr(api_app, "PaperStore", lambda *args, **kwargs: store)

    class _StubWorkflow:
        def __init__(self):
            self.wf = ResearchWorkflow(
                store, _Retriever(), model or _Model(), max_retries=0,
                sleeper=lambda _: None, claim_soft_llm=False,
            )

        def run(self, task_id, mode="run"):
            return self.wf.run(task_id, mode=mode)

    monkeypatch.setattr(api_app, "ResearchWorkflow", _StubWorkflow)
    return TestClient(api_app.app), store


def _create_payload():
    return {
        "question": "为企业知识库选择可落地的 RAG 技术路线",
        "requirements": {"隐私要求": "内部敏感资料"},
        "candidates": ["RAPTOR", "CRAG"],
        "weights": WEIGHTS,
    }



def test_liveness_does_not_need_external_services():
    client = TestClient(api_app.app)

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_accepts_local_qdrant_and_optional_neo4j(monkeypatch, tmp_path):
    import neo4j

    monkeypatch.setattr(api_app, "QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(
        neo4j.GraphDatabase,
        "driver",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    response = TestClient(api_app.app).get("/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {
        "qdrant": "ok_local",
        "neo4j": "optional_unavailable",
    }


def test_research_task_rejects_duplicate_candidates():
    payload = _create_payload()
    payload["candidates"] = ["CRAG", " CRAG "]

    response = TestClient(api_app.app).post("/research-tasks", json=payload)

    assert response.status_code == 422


def test_query_delegates_to_selected_pipeline(monkeypatch):
    client = TestClient(api_app.app)
    monkeypatch.setattr(
        api_app,
        "execute_query",
        lambda question, mode: {"answer": question, "route": mode},
    )

    response = client.post("/queries", json={"question": "  测试问题  ", "mode": "rag"})

    assert response.status_code == 200
    assert response.json() == {"answer": "测试问题", "route": "rag"}


def test_query_rejects_unknown_mode():
    client = TestClient(api_app.app)

    response = client.post("/queries", json={"question": "测试", "mode": "unsafe"})

    assert response.status_code == 422


def test_upload_rejects_non_pdf():
    client = TestClient(api_app.app)

    response = client.post(
        "/papers",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 415


def test_full_lifecycle_create_approve_blocks_cancel_and_revise(monkeypatch, tmp_path):
    client, store = _isolate(monkeypatch, tmp_path)

    created = client.post("/research-tasks", json=_create_payload())
    assert created.status_code == 202
    task_id = created.json()["task_id"]

    task = client.get(f"/research-tasks/{task_id}").json()
    assert task["status"] == "awaiting_approval"
    assert client.get(f"/research-tasks/{task_id}/report").json()["version"] == 1

    approved = client.post(f"/research-tasks/{task_id}/approve", json={"reviewer_note": "确认"})
    assert approved.status_code == 200
    assert client.get(f"/research-tasks/{task_id}").json()["status"] == "completed"

    assert client.post(f"/research-tasks/{task_id}/cancel").status_code == 409
    assert client.post(f"/research-tasks/{task_id}/resume").status_code == 409
    assert client.post(
        f"/research-tasks/{task_id}/revise", json={"reviewer_note": "补充延迟对比"}
    ).status_code == 409


def test_revise_creates_version_two(monkeypatch, tmp_path):
    client, store = _isolate(monkeypatch, tmp_path)
    task_id = client.post("/research-tasks", json=_create_payload()).json()["task_id"]

    response = client.post(
        f"/research-tasks/{task_id}/revise", json={"reviewer_note": "重点比较 CRAG 延迟"}
    )
    assert response.status_code == 202
    assert response.json()["current_step"] == "draft_report"

    report = client.get(f"/research-tasks/{task_id}/report").json()
    assert report["version"] == 2
    versions = client.get(f"/research-tasks/{task_id}/report/versions").json()
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["status"] == "superseded"

    assert client.post(f"/research-tasks/{task_id}/revise", json={"reviewer_note": ""}).status_code == 422


def test_failed_task_can_resume_via_api(monkeypatch, tmp_path):
    client, store = _isolate(monkeypatch, tmp_path, model=_Model(fail=True))
    task_id = client.post("/research-tasks", json=_create_payload()).json()["task_id"]
    failed = client.get(f"/research-tasks/{task_id}").json()
    assert failed["status"] == "failed"
    assert failed["failed_step"] == "draft_report"
    assert failed["error_type"] == "RuntimeError"

    # 恢复时换用可用模型
    from research.workflow import ResearchWorkflow
    class _GoodStub:
        def run(self, task_id, mode="run"):
            return ResearchWorkflow(
                store, _Retriever(), _Model(), max_retries=0, sleeper=lambda _: None,
                claim_soft_llm=False,
            ).run(task_id, mode=mode)

    monkeypatch.setattr(api_app, "ResearchWorkflow", _GoodStub)
    resumed = client.post(f"/research-tasks/{task_id}/resume")
    assert resumed.status_code == 202
    assert client.get(f"/research-tasks/{task_id}").json()["status"] == "awaiting_approval"


def test_cancel_queued_and_running_tasks(monkeypatch, tmp_path):
    client, store = _isolate(monkeypatch, tmp_path)

    # queued 任务立即取消
    queued = store.create_task(
        {
            "task_id": "t-queued", "question": "排队中的任务问题示例",
            "requirements": {}, "candidates": ["RAPTOR", "CRAG"], "weights": WEIGHTS,
        }
    )
    response = client.post("/research-tasks/t-queued/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    # running 任务记录取消请求，等待工作流节点边界退出
    running = store.create_task(
        {
            "task_id": "t-running", "question": "运行中的任务问题示例",
            "requirements": {}, "candidates": ["RAPTOR", "CRAG"], "weights": WEIGHTS,
        }
    )
    store.update_task("t-running", "running", "retrieve_evidence")
    response = client.post("/research-tasks/t-running/cancel")
    assert response.status_code == 200
    assert response.json()["cancel_requested"] in (1, True)
    assert store.is_cancel_requested("t-running")

    # 已取消不能再次取消
    assert client.post("/research-tasks/t-queued/cancel").status_code == 409
    assert client.post("/research-tasks/missing/cancel").status_code == 404


def test_cancel_and_resume_require_existing_task(monkeypatch, tmp_path):
    client, _ = _isolate(monkeypatch, tmp_path)
    assert client.post("/research-tasks/missing/resume").status_code == 404
    assert client.post("/research-tasks/missing/revise", json={"reviewer_note": "x"}).status_code == 404
