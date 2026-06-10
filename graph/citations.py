"""
12 篇论文的元数据 + 引用边（graph/citations.py）。
- 边数据来自 ingest/detect_citations.py 的自动检测（arXiv ID + 标题关键词 + 作者姓+年份）
- 如需校对/修正，直接修改 CITATIONS 列表后重跑 build_neo4j.py 即可
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT

# ============== 论文元数据 ==============
# arxiv_id 是唯一主键；短键 short_id 用于在 Neo4j 中按简称查询
PAPERS = {
    "2005.11401": {
        "short_id": "lewis2020",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "first_author": "Patrick Lewis",
        "year": 2020,
        "short_name": "Lewis et al. 2020 (RAG 奠基论文)",
    },
    "2305.06983": {
        "short_id": "jiang2023",
        "title": "Active Retrieval Augmented Generation",
        "first_author": "Zhengbao Jiang",
        "year": 2023,
        "short_name": "Jiang et al. 2023 (FLARE)",
    },
    "2309.15217": {
        "short_id": "es2023",
        "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "first_author": "Shahul Es",
        "year": 2023,
        "short_name": "Es et al. 2023 (RAGAS)",
    },
    "2401.15884": {
        "short_id": "yan2024",
        "title": "Corrective Retrieval Augmented Generation",
        "first_author": "Shi-Qi Yan",
        "year": 2024,
        "short_name": "Yan et al. 2024 (CRAG)",
    },
    "2401.18059": {
        "short_id": "sarthi2024",
        "title": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "first_author": "Parth Sarthi",
        "year": 2024,
        "short_name": "Sarthi et al. 2024 (RAPTOR)",
    },
    "2402.01767": {
        "short_id": "chen2024",
        "title": "HiQA: A Hierarchical Contextual Augmentation RAG for Multi-Documents QA",
        "first_author": "Xinyue Chen",
        "year": 2024,
        "short_name": "Chen et al. 2024 (HiQA)",
    },
    "2404.16130": {
        "short_id": "edge2024",
        "title": "From Local to Global: A GraphRAG Approach to Query-Focused Summarization",
        "first_author": "Darren Edge",
        "year": 2024,
        "short_name": "Edge et al. 2024 (Microsoft GraphRAG)",
    },
    "2409.14924": {
        "short_id": "zhao2024",
        "title": "RAG and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely",
        "first_author": "Siyun Zhao",
        "year": 2024,
        "short_name": "Zhao et al. 2024 (RAG and Beyond)",
    },
    "2410.12837": {
        "short_id": "gupta2024",
        "title": "A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions",
        "first_author": "Shailja Gupta",
        "year": 2024,
        "short_name": "Gupta et al. 2024 (RAG Survey)",
    },
    "2503.10677": {
        "short_id": "cheng2025",
        "title": "A Survey on Knowledge-Oriented Retrieval-Augmented Generation",
        "first_author": "Mingyue Cheng",
        "year": 2025,
        "short_name": "Cheng et al. 2025 (Knowledge-Oriented RAG Survey)",
    },
    "2601.08773": {
        "short_id": "chinthareddy2026",
        "title": "Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs",
        "first_author": "Manideep Reddy Chinthareddy",
        "year": 2026,
        "short_name": "Chinthareddy 2026 (Code GraphRAG)",
    },
    "2605.18765": {
        "short_id": "li2026",
        "title": "STAR: Semantic-Tuned and Tail-Adaptive Retriever for Graph-Augmented Generation",
        "first_author": "Shuai Li",
        "year": 2026,
        "short_name": "Li et al. 2026 (STAR)",
    },
}

# ============== 引用边 (src -> dst: src 引用了 dst) ==============
# 来自 detect_citations.py 自动检测；用户可手动校对
CITATIONS_FILE = PROJECT_ROOT / "data" / "detected_citations.json"


def get_citations() -> list[tuple[str, str]]:
    """加载引用边；优先读 detected_citations.json，否则用空列表并打印警告"""
    if CITATIONS_FILE.exists():
        data = json.loads(CITATIONS_FILE.read_text(encoding="utf-8"))
        return [(e["src"], e["dst"]) for e in data["edges"]]
    print(f"[WARNING] 引用数据文件不存在: {CITATIONS_FILE}，将使用空引用列表")
    return []


if __name__ == "__main__":
    edges = get_citations()
    print(f"[citations] {len(PAPERS)} papers, {len(edges)} edges")
    for arxiv, meta in PAPERS.items():
        out = sum(1 for s, _ in edges if s == arxiv)
        inn = sum(1 for _, d in edges if d == arxiv)
        print(f"  {arxiv}  {meta['short_name']:<48}  out={out:2d}  in={inn:2d}")
