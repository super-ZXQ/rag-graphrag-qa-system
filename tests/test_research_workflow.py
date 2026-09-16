from paper_library.store import PaperStore
from research.workflow import ResearchWorkflow, validate_report


DEFAULT_WEIGHTS = {
    "检索质量与可追溯性": 25,
    "在线延迟": 15,
    "数据更新复杂度": 15,
    "实施成本": 15,
    "运行成本": 10,
    "隐私与部署适配": 10,
    "运维与团队适配": 10,
}


class FakeRetriever:
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


class FakeMessage:
    content = "# Report\n\nThe method improves retrieval quality [E-paper-1-p2-c0]."


class FakeModel:
    def invoke(self, messages):
        combined = "\n".join(message.content for message in messages)
        assert "The method improves retrieval quality." in combined
        return FakeMessage()


def test_research_workflow_persists_evidence_and_waits_for_approval(tmp_path):
    store = PaperStore(tmp_path / "research.sqlite3")
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
        ["", "The method improves retrieval quality."],
        [{"chunk_id": "paper-1:p2:c0", "page_number": 2,
          "text": "The method improves retrieval quality."}],
    )
    store.update_paper_status("paper-1", "ready", index_version="test")
    task = store.create_task(
        {
            "task_id": "task-1",
            "question": "选择企业知识库检索方案",
            "requirements": {"privacy": "internal"},
            "candidates": ["RAPTOR", "CRAG"],
            "weights": DEFAULT_WEIGHTS,
        }
    )

    report = ResearchWorkflow(store, FakeRetriever(), FakeModel()).run(task["task_id"])

    assert report["status"] == "draft"
    assert report["validation"]["valid"] is True
    assert store.get_task("task-1")["status"] == "awaiting_approval"
    assert len(store.list_evidence("task-1")) == 1


def test_validate_report_detects_fabricated_citation():
    validation = validate_report("claim [E-missing]", [])

    assert validation["valid"] is False
    assert validation["missing_evidence"] == ["E-missing"]


def test_validate_report_ignores_placeholder_and_allows_annotation():
    evidence = [{"evidence_id": "E-paper-1-p2-c0"}]

    validation = validate_report(
        "格式示例 [E-...]，有效引用 [E-paper-1-p2-c0 支持该结论]。", evidence
    )

    assert validation["valid"] is True
    assert validation["missing_evidence"] == []
