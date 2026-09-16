"""Fail-closed deterministic report used when the demo LLM output is rejected."""
from __future__ import annotations

from research.decision import render_decision_matrix
from research.workflow import build_evidence_chain_section


def build_validated_showcase(task: dict, evidence: list[dict], decision_matrix: dict) -> str:
    def ref(prefix: str, phrase: str | None = None) -> str | None:
        for item in evidence:
            if not item["evidence_id"].startswith(prefix):
                continue
            if phrase and phrase.lower() not in item.get("text", "").lower():
                continue
            return item["evidence_id"]
        return None

    facts = []
    internal_product = ref("E-local-bcba105fe48f")
    internal_constraints = ref("E-local-f4c9fff07eb3")
    raptor = ref("E-local-01319e2e6e6e", "consistently outperforms")
    crag = ref("E-local-efc0de696312", "overall evaluation results")
    flare = ref("E-local-2425bd04de6e", "outperforms all")
    graphrag = ref("E-local-730f1a9f38d1", "strongly outperforms")

    if internal_product:
        facts.append(
            f"Atlas 初期语料约 180,000 份文档，其中约 35% 是超过 30 页的长文档 [{internal_product}]。"
        )
    if internal_constraints:
        facts.extend(
            [
                f"现有环境提供 16 核 CPU、64 GB 内存和单张 24 GB 显存 GPU [{internal_constraints}]。",
                f"首期由 4 人小组在 8 周内交付 [{internal_constraints}]。",
            ]
        )
    if raptor:
        facts.append(
            f"RAPTOR combined with a retriever consistently outperforms the respective retriever across the evaluated datasets [{raptor}]."
        )
    if crag:
        facts.append(
            f"CRAG Table 1 reports overall evaluation results on the test sets of four datasets [{crag}]."
        )
    if flare:
        facts.append(
            f"FLARE outperforms all baselines on all reported tasks and datasets [{flare}]."
        )
    if graphrag:
        facts.append(
            f"GraphRAG results show stronger performance than vector RAG when GPT-4 is used as the LLM [{graphrag}]。"
        )

    references = sorted({item["evidence_id"] for item in evidence if item["evidence_id"] in " ".join(facts)})
    recommended = decision_matrix.get("recommended_candidate") or "证据不足"
    core = "\n".join(
        [
            "# 企业知识库 RAG 技术选型报告",
            "",
            "## 执行摘要",
            "",
            f"工程推断：根据业务硬约束和用户权重，建议把 {recommended} 作为首期基线，并通过 POC 再决定是否叠加复杂技术。",
            "",
            "## 关键事实",
            "",
            *[f"- {fact}" for fact in facts],
            "",
            "## 风险与未知项",
            "",
            "- 工程推断：论文结果不能直接等同于 Atlas 线上效果，仍需使用企业问题集验证延迟、准确性与维护成本。",
            "- 工程推断：精确证据召回仍低于文档召回，应优先优化分块、重排与跨文档证据聚合。",
            "",
            "## 推荐草案",
            "",
            f"- 建议首期采用 {recommended}；硬约束失败的候选不进入首选。",
            "- 建议用 2 周 POC 测试可答率、精确证据 Recall@5、P95 延迟和增量索引耗时。",
            "",
            "## 参考资料",
            "",
            *[f"- [{evidence_id}]" for evidence_id in references],
        ]
    )
    appendices = [render_decision_matrix(decision_matrix), build_evidence_chain_section(task["candidates"])]
    return core.rstrip() + "\n\n" + "\n\n".join(item for item in appendices if item)
