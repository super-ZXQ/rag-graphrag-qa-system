"""从 OpenAlex 构建黄金论文的一跳引用图（带缓存、限流、有界扩展）。

用法（本地 Qdrant 模式无关，仅访问 OpenAlex）：
    python scripts/build_citation_graph.py [--neo4j] [--limit 6]

默认写入 data/citation_graph.sqlite3；--neo4j 且 Neo4j 连通时额外写入固定模型节点。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.domain import FITS, REQUIREMENTS, TECHNIQUES  # noqa: E402
from graph.store import CitationGraph, get_graph  # noqa: E402
from sources.openalex_graph import (  # noqa: E402
    CitationGraphError,
    citation_neighbors,
    find_work_by_arxiv,
)


def build_sqlite(limit: int = 6) -> CitationGraph:
    graph = CitationGraph()
    for key, meta in TECHNIQUES.items():
        arxiv_id = meta.get("arxiv_id")
        if not arxiv_id:
            continue
        work = find_work_by_arxiv(arxiv_id, title_hint=meta.get("title_hint") or meta["name"])
        if not work:
            print(f"[graph] {meta['name']} 未在 OpenAlex 定位，跳过")
            continue
        wid = work["id"].rsplit("/", 1)[-1]
        graph.upsert_paper(
            {
                "openalex_id": wid, "arxiv_id": arxiv_id, "title": work.get("title", meta["name"]),
                "year": work.get("publication_year"), "cited_by_count": work.get("cited_by_count", 0),
                "doi": work.get("doi") or "",
            },
            is_seed=True, technique_key=key,
        )
        try:
            neighbors = citation_neighbors(wid, limit=limit)
        except CitationGraphError as exc:
            print(f"[graph] {meta['name']} 邻居获取失败：{exc}")
            continue
        for item in neighbors["citing"]:
            graph.upsert_paper({
                "openalex_id": item["openalex_id"], "arxiv_id": None, "title": item["title"],
                "year": item["year"], "cited_by_count": item["cited_by_count"], "doi": item["doi"] or "",
            })
            graph.upsert_edge(item["openalex_id"], wid, "CITES")
        for item in neighbors["referenced"]:
            graph.upsert_paper({
                "openalex_id": item["openalex_id"], "arxiv_id": None, "title": item["title"],
                "year": item["year"], "cited_by_count": item["cited_by_count"], "doi": item["doi"] or "",
            })
            graph.upsert_edge(wid, item["openalex_id"], "CITES")
        breadth = graph.evidence_breadth(key)
        print(f"[graph] {meta['name']:<10} seed={wid} neighbors={breadth['neighbor_count']} "
              f"cited_by={work.get('cited_by_count', 0)}")
    return graph


def write_neo4j() -> bool:
    """把固定模型（Paper/Technique/Requirement 及关系）写入 Neo4j；不可用返回 False。"""
    try:
        from neo4j import GraphDatabase
        from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
                                      connection_timeout=3)
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        print(f"[graph] Neo4j 不可用，跳过写入并保留 SQLite 图：{type(exc).__name__}")
        return False

    sqlite = get_graph()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for key, meta in TECHNIQUES.items():
            session.run("MERGE (t:Technique {key:$key}) SET t.name=$name",
                        key=key, name=meta["name"])
        for req_id, desc in REQUIREMENTS.items():
            session.run("MERGE (r:Requirement {id:$id}) SET r.description=$desc",
                        id=req_id, desc=desc)
        for key, reqs in FITS.items():
            for req_id, (stance, rationale) in reqs.items():
                session.run(
                    "MATCH (t:Technique {key:$key}), (r:Requirement {id:$req}) "
                    "MERGE (t)-[f:FITS]->(r) SET f.stance=$stance, f.rationale=$rationale",
                    key=key, req=req_id, stance=stance, rationale=rationale,
                )
        with sqlite._conn() as connection:  # 固定模板写入，仅离线构建时执行
            papers = connection.execute("SELECT * FROM graph_papers").fetchall()
            edges = connection.execute("SELECT * FROM graph_edges").fetchall()
        for p in papers:
            session.run(
                "MERGE (p:Paper {openalex_id:$id}) SET p.title=$title, p.arxiv_id=$arxiv, "
                "p.cited_by_count=$cited, p.technique_key=$technique",
                id=p["openalex_id"], title=p["title"], arxiv=p["arxiv_id"] or "",
                cited=p["cited_by_count"], technique=p["technique_key"],
            )
        for key in TECHNIQUES:
            session.run(
                "MATCH (p:Paper {technique_key:$key}), (t:Technique {key:$key}) "
                "MERGE (p)-[:DISCUSSES]->(t)", key=key,
            )
        for e in edges:
            session.run(
                "MATCH (a:Paper {openalex_id:$s}), (b:Paper {openalex_id:$d}) "
                f"MERGE (a)-[:{e['relation']}]->(b)", s=e["src_id"], d=e["dst_id"],
            )
    driver.close()
    print("[graph] Neo4j 固定模型写入完成")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--neo4j", action="store_true")
    args = parser.parse_args()
    build_sqlite(limit=args.limit)
    if args.neo4j:
        write_neo4j()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
