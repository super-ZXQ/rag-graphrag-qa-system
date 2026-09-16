"""阶段 2：claim 证据可信度、硬/软校验与审批门槛测试。"""
import json

from paper_library.store import PaperStore
from research.claims import evaluate_claims, extract_claims, hard_validate

EVIDENCE = [
    {
        "evidence_id": "E-paper-1-p2-c0",
        "paper_id": "paper-1",
        "chunk_id": "paper-1:p2:c0",
        "page_number": 2,
        "title": "RAG Evidence",
        "text": "The hybrid method improves retrieval quality and reduces online latency.",
        "source_uri": "https://example.test/paper-1",
        "score": 0.9,
    }
]


def _store_with_chunk(tmp_path):
    store = PaperStore(tmp_path / "c.sqlite3")
    paper = {
        "paper_id": "paper-1",
        "content_hash": "a" * 64,
        "filename": "p.md",
        "title": "RAG Evidence",
        "stored_path": str(tmp_path / "p.md"),
        "source_type": "public_arxiv",
        "source_uri": "https://example.test/paper-1",
        "visibility": "public",
    }
    store.save_processed_paper(
        paper,
        ["The hybrid method improves retrieval quality and reduces online latency."],
        [{
            "chunk_id": "paper-1:p2:c0", "page_number": 1,
            "text": "The hybrid method improves retrieval quality and reduces online latency.",
        }],
    )
    return store


def test_supported_paper_fact_passes_approval(tmp_path):
    store = _store_with_chunk(tmp_path)
    markdown = "The hybrid method improves retrieval quality and latency [E-paper-1-p2-c0]."

    result = evaluate_claims(markdown, EVIDENCE, store=store, use_llm=False)

    assert result["citation_existence_rate"] == 1.0
    assert result["hard_errors"] == []
    assert result["claim_count"] == 1
    claim = result["claims"][0]
    assert claim["claim_type"] == "paper_fact"
    assert claim["support_status"] == "supported"
    assert result["claim_evidence_coverage"] == 1.0
    assert result["approval_allowed"] is True
    assert result["valid"] is True


def test_fabricated_citation_is_blocked():
    markdown = "A made up conclusion [E-ghost]."

    result = evaluate_claims(markdown, EVIDENCE, use_llm=False)

    assert result["citation_existence_rate"] == 0.0
    assert "E-ghost" in result["missing_evidence"]
    assert result["fabricated_citations"] == ["E-ghost"]
    assert any(e["code"] == "fabricated_citation" for e in result["hard_errors"])
    assert result["approval_allowed"] is False


def test_uncited_fact_is_extracted_and_blocks_approval():
    markdown = (
        "The hybrid method improves retrieval quality and reduces online latency "
        "[E-paper-1-p2-c0].\n"
        "Atlas 已通过 SOC2 安全认证。"
    )

    result = evaluate_claims(markdown, EVIDENCE, use_llm=False)

    assert result["claim_count"] == 2
    assert any(error["code"] == "missing_citation" for error in result["hard_errors"])
    assert result["claim_evidence_coverage"] == 0.5
    assert result["approval_allowed"] is False


def test_reference_only_line_is_not_treated_as_claim():
    markdown = (
        "The hybrid method improves retrieval quality and reduces online latency "
        "[E-paper-1-p2-c0].\n\n## 参考资料\n1. [E-paper-1-p2-c0]"
    )

    result = evaluate_claims(markdown, EVIDENCE, use_llm=False)

    assert result["claim_count"] == 1
    assert result["approval_allowed"] is True


def test_code_fenced_report_is_blocked_as_possible_prompt_injection():
    markdown = (
        "```json\n{\"claim\": \"The hybrid method improves retrieval quality "
        "[E-paper-1-p2-c0].\"}\n```"
    )

    result = evaluate_claims(markdown, EVIDENCE, use_llm=False)

    assert any(error["code"] == "invalid_report_format" for error in result["hard_errors"])
    assert result["approval_allowed"] is False


def test_existing_citation_that_does_not_support_claim_is_refused():
    # 证据讲检索质量，结论却讲无关的图谱社区总结
    markdown = (
        "GraphRAG builds a knowledge graph and uses community summaries for global questions "
        "[E-paper-1-p2-c0]."
    )

    result = evaluate_claims(markdown, EVIDENCE, use_llm=False)
    claim = result["claims"][0]

    assert claim["support_status"] in ("insufficient", "partially_supported")
    assert result["claim_evidence_coverage"] == 0.0
    assert result["approval_allowed"] is False


def test_missing_evidence_refuses_recommendation():
    markdown = "# 技术选型报告（证据不足）\n\n证据不足，无法给出推荐。"

    result = evaluate_claims(markdown, evidence=[], use_llm=False)

    assert result["insufficient_only"] is True
    assert result["approval_allowed"] is False
    assert result["cited_evidence"] == 0


def test_engineering_inference_is_typed_and_separate_from_fact():
    markdown = (
        "The hybrid method improves retrieval quality [E-paper-1-p2-c0].\n"
        "建议先用 4 人团队在 8 周内做 POC 以控制实施成本 [E-paper-1-p2-c0]."
    )

    result = evaluate_claims(markdown, EVIDENCE, use_llm=False)
    types = {c["claim_type"] for c in result["claims"]}

    assert "paper_fact" in types
    assert "engineering_inference" in types
    assert result["engineering_inference_count"] == 1
    assert result["approval_allowed"] is True


def test_evidence_locates_paper_chunk_and_page(tmp_path):
    store = _store_with_chunk(tmp_path)
    claims = extract_claims(
        "The hybrid method improves retrieval quality [E-paper-1-p2-c0].", EVIDENCE
    )
    errors = hard_validate(claims, EVIDENCE, store=store)
    assert errors == []

    # 指向不存在的分块必须报 chunk_missing
    bad_evidence = [{**EVIDENCE[0], "chunk_id": "paper-1:p9:c9", "page_number": 9}]
    bad_claims = extract_claims(
        "The hybrid method improves retrieval quality [E-paper-1-p2-c0].", bad_evidence
    )
    errors = hard_validate(bad_claims, bad_evidence, store=store)
    assert any(e["code"] == "chunk_missing" for e in errors)


def test_llm_cannot_upgrade_unsupported_or_fabricated_claim():
    class RogueModel:
        def invoke(self, messages):
            class Response:
                content = json.dumps([
                    {
                        "claim_id": "C-001",
                        "support_status": "supported",
                        "confidence": 0.99,
                        "risk_note": "强行支持",
                    }
                ])
            return Response()

    markdown = "GraphRAG community summaries answer global questions [E-paper-1-p2-c0]."
    result = evaluate_claims(markdown, EVIDENCE, model=RogueModel(), use_llm=True)

    # 启发式判定不支持，LLM 无权翻案为 supported
    assert result["claims"][0]["support_status"] in ("insufficient", "partially_supported")
    assert result["approval_allowed"] is False
    # LLM 确实被调用，但它的“支持”翻案被硬规则否决
    assert result["soft_checker"] == "llm+heuristic"


def test_llm_can_downgrade_an_overlap_match():
    class DowngradeModel:
        def invoke(self, messages):
            class Response:
                content = json.dumps([
                    {
                        "claim_id": "C-001",
                        "support_status": "conflicted",
                        "confidence": 0.8,
                        "risk_note": "证据与结论冲突",
                    }
                ])
            return Response()

    markdown = "The hybrid method improves retrieval quality and latency [E-paper-1-p2-c0]."
    result = evaluate_claims(markdown, EVIDENCE, model=DowngradeModel(), use_llm=True)

    assert result["claims"][0]["support_status"] == "conflicted"
    assert result["approval_allowed"] is False
