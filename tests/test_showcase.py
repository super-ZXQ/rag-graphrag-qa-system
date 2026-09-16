from research.claims import evaluate_claims
from research.decision import build_decision_matrix
from research.showcase import build_validated_showcase


def test_fail_closed_showcase_contains_only_supported_facts(monkeypatch):
    monkeypatch.setattr("research.showcase.build_evidence_chain_section", lambda candidates: "")
    task = {
        "question": "为敏感长文档知识库选择方案",
        "requirements": {"隐私要求": "内部敏感资料", "团队规模": "4 人，8 周"},
        "candidates": ["基础混合检索 RAG", "CRAG"],
        "weights": {"隐私与部署适配": 50, "实施成本": 50},
    }
    evidence = [
        {
            "evidence_id": "E-local-bcba105fe48f-p1-c0", "paper_id": "internal-1",
            "chunk_id": "internal-1:p1:c0", "page_number": 1, "title": "Requirements",
            "text": "Atlas 初期语料约 180,000 份文档，其中约 35% 是超过 30 页的长文档。",
            "score": 1.0,
        },
        {
            "evidence_id": "E-local-f4c9fff07eb3-p1-c0", "paper_id": "internal-2",
            "chunk_id": "internal-2:p1:c0", "page_number": 1, "title": "Constraints",
            "text": "现有环境提供 16 核 CPU、64 GB 内存和单张 24 GB 显存 GPU。首期由 4 人小组在 8 周内交付。",
            "score": 1.0,
        },
    ]
    matrix = build_decision_matrix(task)

    markdown = build_validated_showcase(task, evidence, matrix)
    core = markdown.split("## 系统确定性业务评分", 1)[0].rstrip()
    validation = evaluate_claims(core, evidence, use_llm=False)

    assert validation["valid"] is True
    assert validation["claim_evidence_coverage"] == 1.0
    assert matrix["recommended_candidate"] == "基础混合检索 RAG"
