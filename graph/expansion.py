"""hybrid 检索后的引用图背书重排：宽召回、弱信号重排、跨论文去重、限界。

设计原则（避免图谱信号伤害通用检索）：
- 仅当查询显式提到某个候选技术时才启用图重排，否则原样返回 hybrid 结果；
- 引用背书只作为近似分数的小幅 tie-break（alpha 很小），不压过检索相关性；
- 图谱只用于重排与证据链展示，不把未下载全文的邻居论文当作证据；
- 图后端缺失或为空时安全降级为普通 hybrid。
"""
from __future__ import annotations

import math

from graph.domain import ARXIV_FILENAME_TO_TECHNIQUE, TECHNIQUES
from graph.store import CitationGraph, get_graph
from retrieval.hybrid import HybridRetriever

WIDE_K = 8
FINAL_K = 5
RERANK_ALPHA = 0.005
PER_PAPER_CAP = 4

# 查询中出现这些 token 才认为与候选技术相关（图谱扩展的触发门）
TECHNIQUE_TOKENS = {
    "raptor": ("raptor", "RAPTOR"),
    "crag": ("crag", "CRAG"),
    "flare": ("flare", "FLARE"),
    "graphrag": ("graphrag", "GraphRAG", "图谱", "社区摘要", "Leiden"),
}


def detect_techniques(query: str) -> set[str]:
    q = query or ""
    hits = set()
    for key, tokens in TECHNIQUE_TOKENS.items():
        if any(token in q for token in tokens):
            hits.add(key)
    return hits


class GraphExpander:
    def __init__(self, retriever: HybridRetriever | None = None, graph: CitationGraph | None = None):
        # retriever 懒加载：evidence_chain 只依赖引用图，不应强制初始化 Qdrant/embedding
        self._retriever = retriever
        self.graph = graph if graph is not None else get_graph()
        self._cited_cache: dict[str, int] = {}

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def _cited(self, filename: str) -> int:
        technique = ARXIV_FILENAME_TO_TECHNIQUE.get(filename or "")
        if not technique:
            return 0
        if technique not in self._cited_cache:
            try:
                self._cited_cache[technique] = self.graph.cited_by_count(technique)
            except Exception:
                self._cited_cache[technique] = 0
        return self._cited_cache[technique]

    def graph_ready(self) -> bool:
        try:
            with self.graph._conn() as connection:
                return connection.execute("SELECT COUNT(*) FROM graph_papers").fetchone()[0] > 0
        except Exception:
            return False

    def search(self, query: str, limit: int = FINAL_K, wide_k: int = WIDE_K) -> list[dict]:
        candidates = self.retriever.search(query, wide_k)
        if not candidates:
            return []
        active = detect_techniques(query)
        if not self.graph_ready() or not active:
            return candidates[:limit]

        max_cited = max((self._cited(c.get("filename", "")) for c in candidates), default=0)
        denom = math.log1p(max(1, max_cited))
        for item in candidates:
            cited = self._cited(item.get("filename", ""))
            # 仅对查询命中的候选技术给引用背书，避免抬高无关论文
            technique = ARXIV_FILENAME_TO_TECHNIQUE.get(item.get("filename", ""), "")
            applicable = cited if technique in active else 0
            boost = RERANK_ALPHA * (math.log1p(applicable) / denom) if denom and applicable else 0.0
            item["graph_boost"] = round(boost, 5)
            item["graph_cited_by"] = cited
            item["score"] = float(item.get("score", 0.0)) + boost

        ordered = sorted(candidates, key=lambda x: x["score"], reverse=True)
        # 软去重：仅在结果充足时限制单篇论文占比，避免挤掉具体论文事实
        chosen, seen, per_paper = [], set(), {}
        for item in ordered:
            if item["chunk_id"] in seen:
                continue
            paper_id = item.get("paper_id", "")
            if per_paper.get(paper_id, 0) >= PER_PAPER_CAP and len(ordered) > limit:
                continue
            seen.add(item["chunk_id"])
            per_paper[paper_id] = per_paper.get(paper_id, 0) + 1
            chosen.append(item)
            if len(chosen) >= limit:
                break
        return chosen

    def evidence_chain(self, technique_key: str, limit: int = 5) -> dict:
        """引用网络：种子论文、元数据邻居及独立证据验证状态。"""
        try:
            breadth = self.graph.evidence_breadth(technique_key)
            related = self.graph.related_papers(technique_key, limit)
        except Exception:
            return {"technique_key": technique_key, "related": [], "single_paper_dependency": True}
        return {**breadth, "related": related}
