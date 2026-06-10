"""
增强版引用检测：
- arXiv ID 匹配（直接出现在 ref 章节里）
- 论文标题关键词匹配（如 "GraphRAG", "RAPTOR", "FLARE", "RAGAS", "CRAG", "HiQA", "STAR"）
- 第一作者姓 + 年份（如 "Lewis et al., 2020", "Edge et al. 2024"）
- 论文全名（去停用词后子串匹配）
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAPERS_TEXT_DIR, PAPER_MAP_JSON

ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")

# 12 篇论文的元数据：(arxiv_id, first_author_surname, year, title_keywords)
# 关键词必须是论文独有的标识：避免用 "retrieval-augmented generation" 这种通用词
# （会误匹配 GRAG / GraphRAG Survey 等其他论文的标题）
PAPER_META = {
    "2005.11401": ("Lewis",       2020, ["knowledge-intensive nlp tasks"]),
    "2305.06983": ("Jiang",       2023, ["active retrieval augmented generation", "forward-looking", "flare"]),
    "2309.15217": ("Es",          2023, ["ragas", "automated evaluation of retrieval"]),
    "2401.15884": ("Yan",         2024, ["corrective retrieval augmented generation", "corrective rag"]),
    "2401.18059": ("Sarthi",      2024, ["raptor", "recursive abstractive processing", "tree-organized retrieval"]),
    "2402.01767": ("Chen",        2024, ["hiqa", "hierarchical contextual augmentation"]),
    "2404.16130": ("Edge",        2024, ["query-focused summarization", "graphrag approach"]),
    "2409.14924": ("Zhao",        2024, ["external data more wisely"]),
    "2410.12837": ("Gupta",       2024, ["comprehensive survey of retrieval", "current landscape and future"]),
    "2503.10677": ("Cheng",       2025, ["knowledge-oriented retrieval-augmented"]),
    "2601.08773": ("Chinthareddy",2026, ["ast-derived graphs", "reliable graph-rag for codebases"]),
    "2605.18765": ("Li",          2026, ["semantic-tuned", "tail-adaptive retriever", "star"]),
}

# 停用词（不在标题关键词中做匹配）
STOP_WORDS = {"a", "an", "the", "of", "for", "to", "in", "on", "and", "or", "is", "are"}


def find_references_section(text: str) -> tuple[int, str]:
    m = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    if m:
        return m.end(), text[m.end():]
    return 0, text  # fallback 扫全篇


def paper_matches(paper_id: str, ref_text: str) -> bool:
    """判断 ref_text 中是否引用了 paper_id 对应的论文。
    注意：关键词匹配必须用 \b 单词边界，避免 "Chen" 误匹配 "Cheng"，
    或 "retrieval-augmented" 误匹配其他论文的标题子串。
    """
    ref_lower = ref_text.lower()
    # 1) arXiv ID（已经是带边界的，保留）
    if re.search(rf"\b{re.escape(paper_id)}\b", ref_text):
        return True
    author, year, keywords = PAPER_META[paper_id]
    # 2) 作者姓 + 年份（同一条 ref 行内 40 字符内）
    if re.search(rf"\b{author}\b[^\n]{{0,40}}\b{year}\b", ref_text, re.IGNORECASE):
        return True
    if re.search(rf"\b{year}\b[^\n]{{0,40}}\b{author}\b", ref_text, re.IGNORECASE):
        return True
    # 3) 标题关键词 — 必须整词匹配，避免子串误命中
    for kw in keywords:
        # 用 \b...\b；关键词里的连字符 / 空格要正确处理
        if re.search(rf"(?<![\w]){re.escape(kw.lower())}(?![\w])", ref_lower):
            return True
    return False


def main():
    arxiv_ids = list(PAPER_META.keys())
    edges = []
    for src in arxiv_ids:
        text = (PAPERS_TEXT_DIR / f"{src}.txt").read_text(encoding="utf-8")
        _, ref = find_references_section(text)
        for dst in arxiv_ids:
            if dst == src:
                continue
            if paper_matches(dst, ref):
                edges.append((src, dst))
    edges = sorted(set(edges))

    out = {
        "arxiv_ids": arxiv_ids,
        "edges": [{"src": s, "dst": d} for s, d in edges],
    }
    out_path = PAPERS_TEXT_DIR.parent / "detected_citations.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[detect] {len(edges)} edges (enhanced) -> {out_path}")
    in_deg = {a: 0 for a in arxiv_ids}
    out_deg = {a: 0 for a in arxiv_ids}
    for s, d in edges:
        out_deg[s] += 1
        in_deg[d] += 1
    print(f"{'arxiv_id':<12} {'first_author':<14} {'out':>4} {'in':>4}")
    for a in arxiv_ids:
        author = PAPER_META[a][0]
        print(f"  {a:<10} {author:<14} {out_deg[a]:>4} {in_deg[a]:>4}")


if __name__ == "__main__":
    main()
