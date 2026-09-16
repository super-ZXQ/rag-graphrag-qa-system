"""阶段 1：任务取消、失败恢复、修订版本与预算记录的工作流测试。"""
import pytest

from paper_library.store import PaperStore
from research.workflow import ResearchWorkflow, WorkflowFailed, looks_like_prompt_injection, validate_report

WEIGHTS = {
    "检索质量与可追溯性": 25,
    "在线延迟": 15,
    "数据更新复杂度": 15,
    "实施成本": 15,
    "运行成本": 10,
    "隐私与部署适配": 10,
    "运维与团队适配": 10,
}

VALID_REPORT = "# Report\n\nThe method improves retrieval quality and latency [E-paper-1-p2-c0]."
REVISED_REPORT = "# 报告 v2\n\n已重点比较 CRAG 延迟与改造成本 [E-paper-1-p2-c0]。"


class FakeRetriever:
    def __init__(self, store=None, on_search=None):
        self.calls = 0
        self.store = store
        self.on_search = on_search

    def search(self, query, limit):
        self.calls += 1
        if self.on_search:
            self.on_search(query)
        return [
            {
                "chunk_id": "paper-1:p2:c0",
                "paper_id": "paper-1",
                "page_number": 2,
                "title": "RAG Evidence",
                "text": "The method improves retrieval quality and latency.",
                "source_uri": "https://example.test/paper-1",
                "score": 0.5,
            }
        ]


class FakeMessage:
    def __init__(self, content=VALID_REPORT):
        self.content = content
        self.usage_metadata = None


class ScriptedModel:
    def __init__(self, contents=None, error=None, fail_times=0, record=None):
        self.contents = list(contents or [VALID_REPORT])
        self.error = error or RuntimeError("model unavailable")
        self.fail_times = fail_times
        self.calls = 0
        self.record = record if record is not None else []

    def invoke(self, messages):
        self.calls += 1
        combined = "\n".join(getattr(m, "content", "") for m in messages)
        self.record.append(combined)
        if self.calls <= self.fail_times:
            raise self.error
        content = self.contents.pop(0) if self.contents else VALID_REPORT
        return FakeMessage(content)


def _seed_paper(store):
    """种入与 FakeRetriever 证据一致的公开论文与分块（第二页 c0）。"""
    paper = {
        "paper_id": "paper-1",
        "content_hash": "p" * 64,
        "filename": "paper-1.md",
        "title": "RAG Evidence",
        "stored_path": "paper-1.md",
        "source_type": "public_arxiv",
        "source_uri": "https://example.test/paper-1",
        "visibility": "public",
    }
    store.save_processed_paper(
        paper,
        ["", "The method improves retrieval quality and latency."],
        [{"chunk_id": "paper-1:p2:c0", "page_number": 2,
          "text": "The method improves retrieval quality and latency."}],
    )
    store.update_paper_status("paper-1", "ready", index_version="test")


def _make_task(store, task_id="task-1"):
    _seed_paper(store)
    return store.create_task(
        {
            "task_id": task_id,
            "question": "选择企业知识库检索方案",
            "requirements": {"privacy": "internal"},
            "candidates": ["RAPTOR", "CRAG"],
            "weights": WEIGHTS,
        }
    )


def _workflow(store, retriever=None, model=None, max_retries=0):
    return ResearchWorkflow(
        store, retriever or FakeRetriever(), model or ScriptedModel(),
        max_retries=max_retries, sleeper=lambda _: None, claim_soft_llm=False,
    )


def test_cancel_queued_task_before_run_is_cancelled_without_model(tmp_path):
    store = PaperStore(tmp_path / "t.sqlite3")
    _make_task(store)
    store.request_cancel("task-1")

    model = ScriptedModel()
    report = _workflow(store, model=model).run("task-1")

    assert report["status"] == "cancelled"
    assert model.calls == 0
    assert store.get_report("task-1") is None


def test_cancel_running_task_stops_at_retrieval_boundary(tmp_path):
    store = PaperStore(tmp_path / "t.sqlite3")
    _make_task(store)

    def cancel_on_first(query):
        store.request_cancel("task-1")

    retriever = FakeRetriever(store=store, on_search=cancel_on_first)
    report = _workflow(store, retriever=retriever, model=ScriptedModel()).run("task-1")

    assert report["status"] == "cancelled"
    assert retriever.calls == 1  # 第一次检索后即请求取消，第二次边界退出
    assert store.list_evidence("task-1") == []  # 未完成检索，不保存半成品
    assert store.get_report("task-1") is None


def test_failure_persists_step_and_error_type_then_resume_reuses_evidence(tmp_path):
    store = PaperStore(tmp_path / "t.sqlite3")
    _make_task(store)

    failing = _workflow(store, model=ScriptedModel(fail_times=99), max_retries=0)
    with pytest.raises(WorkflowFailed):
        failing.run("task-1")

    failed = store.get_task("task-1")
    assert failed["status"] == "failed"
    assert failed["failed_step"] == "draft_report"
    assert failed["error_type"] == "RuntimeError"
    assert "model unavailable" in failed["error_message"]
    assert len(store.list_evidence("task-1")) == 1  # 证据已保留

    # 恢复：换用可用模型，检索器不应被再次调用
    resume_retriever = FakeRetriever()
    good_model = ScriptedModel()
    report = ResearchWorkflow(
        store, resume_retriever, good_model, max_retries=0, sleeper=lambda _: None,
        claim_soft_llm=False,
    ).run("task-1", mode="resume")

    assert report["status"] == "draft"
    assert resume_retriever.calls == 0  # 复用冻结证据，不重新检索
    assert store.get_task("task-1")["status"] == "awaiting_approval"
    assert good_model.calls == 1


def test_revise_increments_version_keeps_history_and_passes_note(tmp_path):
    store = PaperStore(tmp_path / "t.sqlite3")
    _make_task(store)
    wf = _workflow(store, model=ScriptedModel(contents=[VALID_REPORT]))
    wf.run("task-1")
    first = store.get_report("task-1")
    assert first["version"] == 1

    store.begin_revision("task-1", "请重点比较 CRAG 的延迟和内部化改造成本")
    captured = []
    revise_model = ScriptedModel(contents=[REVISED_REPORT], record=captured)
    revise_retriever = FakeRetriever()
    ResearchWorkflow(
        store, revise_retriever, revise_model, max_retries=0, sleeper=lambda _: None,
        claim_soft_llm=False,
    ).run("task-1", mode="revise")

    latest = store.get_report("task-1")
    assert latest["version"] == 2
    assert latest["status"] == "draft"
    assert "CRAG 延迟" in latest["markdown"]
    assert revise_retriever.calls == 0  # 修订复用证据
    assert "请重点比较 CRAG 的延迟和内部化改造成本" in captured[0]

    versions = store.list_report_versions("task-1")
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["status"] == "superseded"
    assert versions[1]["status"] == "draft"
    assert versions[1]["evidence_ids"] == ["E-paper-1-p2-c0"]


def test_budget_tracks_calls_queries_snippets_and_durations(tmp_path):
    store = PaperStore(tmp_path / "t.sqlite3")
    _make_task(store)
    _workflow(store, model=ScriptedModel()).run("task-1")

    budget = store.get_task("task-1")["budget"]
    assert budget["model_calls"] == 1
    assert budget["retrieval_queries"] == 4  # 1 业务问题 + 1 约束 + 2 候选
    assert budget["sent_snippets"] == 1
    assert budget["retrieved_hits"] >= 1
    assert "draft_report" in budget["step_durations_seconds"]
    assert budget["token_source"] == "estimated_len"
    assert budget["input_tokens"] > 0


def test_transient_model_failure_is_retried_then_succeeds(tmp_path):
    store = PaperStore(tmp_path / "t.sqlite3")
    _make_task(store)
    model = ScriptedModel(fail_times=1)
    report = _workflow(store, model=model, max_retries=2).run("task-1")

    assert report["status"] == "draft"
    assert model.calls == 2
    task = store.get_task("task-1")
    assert task["status"] == "awaiting_approval"
    assert any(event["event_type"] == "retry" for event in task["events"])


def test_insufficient_evidence_report_is_not_valid_for_approval(tmp_path):
    store = PaperStore(tmp_path / "t.sqlite3")
    _make_task(store)

    class EmptyRetriever(FakeRetriever):
        def search(self, query, limit):
            self.calls += 1
            return []

    report = _workflow(store, retriever=EmptyRetriever(), model=ScriptedModel()).run("task-1")
    assert report["validation"]["valid"] is False
    assert store.get_task("task-1")["status"] == "awaiting_approval"


def test_prompt_template_inside_retrieved_paper_is_filtered():
    assert looks_like_prompt_injection(
        "Return output as a well-formed JSON-formatted string with the following format"
    )
    assert not looks_like_prompt_injection(
        "Experiments show the retrieval method improves answer quality."
    )
