"""可复现评测：检索消融 + 无答案拒答 + claim 证据校验，一次运行输出完整结果。

真实结果优先：所有指标由当前索引与代码现场计算，不写死“提升幅度”。
无答案拒答阈值是显式产品门限（记录在实验上下文里），并同时输出逐例 top 分数便于审计。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    EMBED_MODEL,
    HYBRID_RETRIEVER_K,
    LEXICAL_RETRIEVER_K,
    QDRANT_COLLECTION,
    QDRANT_COLLECTION_VERSION,
    QDRANT_PATH,
    QDRANT_URL,
    VECTOR_RETRIEVER_K,
    VECTOR_SIZE,
)
from paper_library.store import PaperStore  # noqa: E402
from research.claims import evaluate_claims  # noqa: E402
from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion  # noqa: E402

TOP_K = 5
# 说明：cosine/RRF/词项重叠分数在跨语言语义检索下无法可靠区分无答案问题，
# 拒答统一交给证据/claim 语义层（answerability 探针 + 报告审批门槛），不在检索层设硬阈值。

GOLDEN_PAPER_VERSIONS = {
    "2401.18059v1.pdf": "RAPTOR arXiv:2401.18059v1",
    "2401.15884v1.pdf": "CRAG arXiv:2401.15884v1",
    "2305.06983v2.pdf": "FLARE arXiv:2305.06983v2",
    "2404.16130v2.pdf": "GraphRAG arXiv:2404.16130v2",
}


def is_relevant(item: dict, case: dict) -> bool:
    chunk_ids = case.get("relevant_chunk_ids", [])
    if chunk_ids:
        return item.get("chunk_id") in chunk_ids
    return bool(case.get("relevant_source")) and item.get("filename") == case["relevant_source"]


def _rank_of_complete_evidence(results: list[dict], case: dict) -> int | None:
    required = set(case.get("relevant_chunk_ids", []))
    require_all = case.get("query_type") in {"cross_document", "conflict"} and len(required) > 1
    if required and require_all:
        found = set()
        for rank, item in enumerate(results, 1):
            if item.get("chunk_id") in required:
                found.add(item["chunk_id"])
            if found == required:
                return rank
        return None
    for rank, item in enumerate(results, 1):
        if is_relevant(item, case):
            return rank
    return None


def _rank_of_complete_document_set(results: list[dict], case: dict) -> int | None:
    required = set(case.get("relevant_sources") or [])
    if not required and case.get("relevant_source"):
        required = {case["relevant_source"]}
    if not required:
        return None
    require_all = case.get("query_type") in {"cross_document", "conflict"} and len(required) > 1
    found = set()
    for rank, item in enumerate(results, 1):
        if item.get("filename") in required:
            if not require_all:
                return rank
            found.add(item["filename"])
        if found == required:
            return rank
    return None


def _metrics_from_pairs(pairs) -> dict:
    def recall(k: int, ranker) -> float:
        if not pairs:
            return 0.0
        hits = sum(1 for case, found in pairs if ranker(found[:k], case) is not None)
        return round(hits / len(pairs), 4)

    document_mrr_total = 0.0
    evidence_mrr_total = 0.0
    for case, found in pairs:
        document_rank = _rank_of_complete_document_set(found, case)
        evidence_rank = _rank_of_complete_evidence(found, case)
        if document_rank:
            document_mrr_total += 1 / document_rank
        if evidence_rank:
            evidence_mrr_total += 1 / evidence_rank
    return {
        "case_count": len(pairs),
        "recall_at_1": recall(1, _rank_of_complete_document_set),
        "recall_at_5": recall(5, _rank_of_complete_document_set),
        "mrr": round(document_mrr_total / len(pairs), 4) if pairs else 0.0,
        "evidence_recall_at_1": recall(1, _rank_of_complete_evidence),
        "evidence_recall_at_5": recall(5, _rank_of_complete_evidence),
        "evidence_mrr": round(evidence_mrr_total / len(pairs), 4) if pairs else 0.0,
    }


def retrieval_metrics(cases: list[dict], results: list[list[dict]]) -> dict:
    answer_pairs = [
        (case, found) for case, found in zip(cases, results)
        if case["expected_behavior"] == "answer"
    ]
    # 论文/技术类问题子集（relevant_source 为 PDF），用于专门比较图谱扩展收益
    paper_pairs = [
        (case, found) for case, found in answer_pairs
        if str(case.get("relevant_source", "")).lower().endswith(".pdf")
    ]
    base = _metrics_from_pairs(answer_pairs)
    paper = _metrics_from_pairs(paper_pairs)
    return {
        "answer_case_count": base["case_count"],
        "refuse_case_count": len(cases) - base["case_count"],
        "recall_at_1": base["recall_at_1"],
        "recall_at_5": base["recall_at_5"],
        "mrr": base["mrr"],
        "evidence_recall_at_1": base["evidence_recall_at_1"],
        "evidence_recall_at_5": base["evidence_recall_at_5"],
        "evidence_mrr": base["evidence_mrr"],
        "paper_question_count": paper["case_count"],
        "paper_recall_at_1": paper["recall_at_1"],
        "paper_recall_at_5": paper["recall_at_5"],
        "paper_mrr": paper["mrr"],
        "paper_evidence_recall_at_1": paper["evidence_recall_at_1"],
        "paper_evidence_recall_at_5": paper["evidence_recall_at_5"],
        "paper_evidence_mrr": paper["evidence_mrr"],
    }


def evaluate_mode(name, cases, search) -> tuple[dict, list[list[dict]]]:
    results = [search(case["query"]) for case in cases]
    metrics = retrieval_metrics(cases, results)
    # 仅记录拒答类问题的 top-1 分数用于审计；检索分数本身不作为拒答门限。
    metrics["refuse_case_top_scores"] = [
        {"id": case["id"], "top_score": round(float(found[0]["score"]), 5) if found else 0.0}
        for case, found in zip(cases, results)
        if case["expected_behavior"] == "refuse"
    ]
    per_case = [
        {
            "id": case["id"],
            "query": case["query"],
            "query_type": case["query_type"],
            "expected_behavior": case["expected_behavior"],
            "relevant_source": case.get("relevant_source", ""),
            "relevant_sources": case.get("relevant_sources", []),
            "top_chunk": (found[0]["chunk_id"] if found else None),
            "top_source": (found[0].get("filename") if found else None),
            "top_score": round(float(found[0]["score"]), 5) if found else 0.0,
            "retrieved": [item.get("filename") or item.get("title") for item in found[:TOP_K]],
            "document_complete_rank": _rank_of_complete_document_set(found, case),
            "evidence_complete_rank": _rank_of_complete_evidence(found, case),
        }
        for case, found in zip(cases, results)
    ]
    return metrics, per_case


def dense_search_factory(retriever: HybridRetriever):
    def search(query: str):
        try:
            rows = retriever.vector_search(query, VECTOR_RETRIEVER_K)
        except Exception:
            rows = []
        ordered = sorted(rows, key=lambda r: float(r.get("vector_score", 0.0)), reverse=True)
        return [{**row, "score": float(row.get("vector_score", 0.0))} for row in ordered[:TOP_K]]
    return search


def fts_search_factory(store: PaperStore):
    def search(query: str):
        rows = store.search_lexical(query, LEXICAL_RETRIEVER_K)
        for row in rows:
            row.setdefault("score", 1.0)  # FTS 命中即视为强匹配；无命中即拒答
        return rows[:TOP_K]
    return search


def hybrid_search_factory(retriever: HybridRetriever):
    def search(query: str):
        return retriever.search(query, TOP_K)
    return search


def evaluate_claim_cases(path: Path) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    rows, passed = [], 0
    pos_existence, pos_coverage = [], []
    negatives_rejected = 0
    negatives_total = 0
    for case in cases:
        result = evaluate_claims(case["markdown"], case["evidence"], use_llm=False)
        ok_approval = result["approval_allowed"] == case["expected"]["approval_allowed"]
        expected_support = case["expected"].get("support_status")
        statuses = {c["support_status"] for c in result["claims"]}
        ok_support = (expected_support in statuses) if expected_support else True
        ok = ok_approval and ok_support
        passed += int(ok)
        if case["expected"]["approval_allowed"]:
            pos_existence.append(result["citation_existence_rate"])
            if result["key_claim_count"]:
                pos_coverage.append(result["claim_evidence_coverage"])
        else:
            negatives_total += 1
            negatives_rejected += int(not result["approval_allowed"])
        rows.append({
            "id": case["id"],
            "expected_approval": case["expected"]["approval_allowed"],
            "actual_approval": result["approval_allowed"],
            "citation_existence_rate": result["citation_existence_rate"],
            "claim_evidence_coverage": result["claim_evidence_coverage"],
            "support_counts": result["support_counts"],
            "hard_errors": [e["code"] for e in result["hard_errors"]],
            "passed": ok,
        })
    return {
        "case_count": len(cases),
        "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
        "positive_citation_existence_rate": (
            round(sum(pos_existence) / len(pos_existence), 4) if pos_existence else None),
        "positive_claim_evidence_coverage": (
            round(sum(pos_coverage) / len(pos_coverage), 4) if pos_coverage else None),
        "negative_rejection_rate": (
            round(negatives_rejected / negatives_total, 4) if negatives_total else None),
        "cases": rows,
    }


def experiment_context(store: PaperStore) -> dict:
    papers = store.list_papers()
    ready = [p for p in papers if p["status"] == "ready"]
    return {
        "run_date_utc": datetime.now(UTC).isoformat(),
        "corpus": {
            "papers_total": len(papers),
            "papers_ready": len(ready),
            "chunks_total": sum(len(store.list_chunks(p["paper_id"])) for p in ready),
            "golden_paper_versions": GOLDEN_PAPER_VERSIONS,
        },
        "embedding_model": EMBED_MODEL,
        "vector_size": VECTOR_SIZE,
        "qdrant_collection": QDRANT_COLLECTION,
        "collection_version": QDRANT_COLLECTION_VERSION,
        "qdrant_backend": ("local_persistence:" + QDRANT_PATH) if QDRANT_PATH else QDRANT_URL,
        "top_k": TOP_K,
        "vector_candidate_k": VECTOR_RETRIEVER_K,
        "fts_candidate_k": LEXICAL_RETRIEVER_K,
        "hybrid_final_k": HYBRID_RETRIEVER_K,
        "rrf_k": 60,
        "refusal_mechanism": "answerability probe (LLM) + report claim gate; retrieval score is not a refusal threshold",
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("retrieval_cases.json"))
    parser.add_argument("--claim-cases", type=Path, default=Path(__file__).with_name("claim_cases.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 可答性探针（离线）")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    store = PaperStore()
    retriever = HybridRetriever(store=store)

    modes = {}
    for name, factory in (
        ("dense_only", dense_search_factory(retriever)),
        ("fts5_only", fts_search_factory(store)),
        ("hybrid_rrf", hybrid_search_factory(retriever)),
    ):
        metrics, per_case = evaluate_mode(name, cases, factory)
        modes[name] = {"metrics": metrics, "cases": per_case}

    # hybrid + graph expansion 在阶段 4 接入；未就绪时显式标记，不伪造结果。
    try:
        from graph.expansion_eval import graph_search_factory  # noqa: F401
        metrics, per_case = evaluate_mode(
            "hybrid_graph", cases, graph_search_factory(retriever, store)
        )
        modes["hybrid_graph"] = {"metrics": metrics, "cases": per_case}
    except Exception as exc:  # 图谱未就绪时降级并说明
        modes["hybrid_graph"] = {"status": "not_available", "reason": type(exc).__name__}

    answerability = {"status": "skipped", "reason": "--no-llm"}
    if not args.no_llm:
        try:
            from llm import get_chat_model, model_label
            from research.answerability import run_answerability_probe
            answerability = run_answerability_probe(
                cases, hybrid_search_factory(retriever), get_chat_model()
            )
            answerability["model"] = model_label()
        except Exception as exc:  # noqa: BLE001
            answerability = {"status": "unavailable", "reason": type(exc).__name__}

    report = {
        "experiment_context": experiment_context(store),
        "retrieval_ablation": modes,
        "answerability": answerability,
        "claim_validation": evaluate_claim_cases(args.claim_cases),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["retrieval_ablation"]["hybrid_rrf"]["metrics"]["recall_at_5"] >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
