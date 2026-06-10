"""
验收：打印 Neo4j 图谱的概览（节点数、边数、出入度、连通分量）+ Qdrant 集合状态。
- 节点数 = 12（硬性）
- 边数 ≥ 30（硬性，计划要求）
- 强连通 / 可达性检查：Lewis 2005.11401 → STAR 2605.18765 的路径（考试题 6）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER,
    PROJECT_ROOT, QDRANT_COLLECTION, QDRANT_URL,
)

sys.stdout.reconfigure(encoding="utf-8")
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from graph.citations import PAPERS, get_citations


def verify_neo4j():
    print("=" * 60)
    print("[verify] Neo4j graph summary")
    print("=" * 60)
    edges = get_citations()
    print(f"[verify] expected: 12 nodes, {len(edges)} edges (auto-detected)")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as s:
        n_nodes = s.run("MATCH (n:Paper) RETURN count(n) AS c").single()["c"]
        n_edges = s.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
        print(f"[verify] actual  : {n_nodes} nodes, {n_edges} edges")
        assert n_nodes == 12, f"节点数应为 12，实际 {n_nodes}"
        assert n_edges >= 30, f"边数应 ≥ 30，实际 {n_edges}"
        print("[verify] node/edge counts PASS")

        # 出入度
        print(f"\n[verify] {'arxiv_id':<12} {'short_name':<48} {'out':>4} {'in':>4}")
        rows = s.run("""
            MATCH (n:Paper)
            OPTIONAL MATCH (n)-[r:CITES]->(m)
            WITH n, count(DISTINCT r) AS out_deg
            OPTIONAL MATCH (k)-[r2:CITES]->(n)
            WITH n, out_deg, count(DISTINCT r2) AS in_deg
            RETURN n.arxiv_id AS id, n.short_name AS name,
                   out_deg, in_deg
            ORDER BY n.year, n.arxiv_id
        """).data()
        for r in rows:
            print(f"  {r['id']:<10} {r['name']:<48} {r['out_deg']:>4} {r['in_deg']:>4}")

        # 路径检查（考试题 6: Lewis 2020 → STAR 2605.18765）
        print("\n[verify] PATH 2005.11401 (Lewis) <-> 2605.18765 (STAR)")
        # 1) 有向：Lewis 沿 CITES 出度能否到达 STAR（Lewis 是基础论文，预计无）
        result = s.run("""
            MATCH p = shortestPath((a:Paper {arxiv_id:'2005.11401'})-[:CITES*]->(b:Paper {arxiv_id:'2605.18765'}))
            RETURN [n IN nodes(p) | n.short_name] AS titles
        """).single()
        if result:
            print(f"  有向 (Lewis → STAR): {' -> '.join(result['titles'])}")
        else:
            print("  有向 (Lewis → STAR): 无路径（Lewis 是基础论文，0 出度）")
        # 2) 有向：STAR 沿 CITES 出度能否到达 Lewis
        result = s.run("""
            MATCH p = shortestPath((a:Paper {arxiv_id:'2605.18765'})-[:CITES*]->(b:Paper {arxiv_id:'2005.11401'}))
            RETURN [n IN nodes(p) | n.short_name] AS titles
        """).single()
        if result:
            print(f"  有向 (STAR → Lewis): {' -> '.join(result['titles'])}")
        # 3) 无向
        result = s.run("""
            MATCH p = shortestPath((a:Paper {arxiv_id:'2005.11401'})-[:CITES*]-(b:Paper {arxiv_id:'2605.18765'}))
            RETURN [n IN nodes(p) | n.short_name] AS titles
        """).single()
        if result:
            print(f"  无向  (Lewis ↔ STAR): {' -> '.join(result['titles'])}")

        # 考试题 4：STAR 引用了哪些论文？
        print("\n[verify] Q4: STAR (2605.18765) cites?")
        for r in s.run("""
            MATCH (a:Paper {arxiv_id:'2605.18765'})-[:CITES]->(b:Paper)
            RETURN b.short_name AS name ORDER BY b.year
        """):
            print(f"  -> {r['name']}")

        # 考试题 5：HiQA 是否引用了 RAPTOR？
        print("\n[verify] Q5: HiQA (2402.01767) cites RAPTOR (2401.18059)?")
        r = s.run("""
            MATCH (a:Paper {arxiv_id:'2402.01767'})-[:CITES]->(b:Paper {arxiv_id:'2401.18059'})
            RETURN b.short_name AS name
        """).single()
        print(f"  {'YES: ' + r['name'] if r else 'NO'}")

    driver.close()


def verify_qdrant():
    print("\n" + "=" * 60)
    print("[verify] Qdrant collection summary")
    print("=" * 60)
    client = QdrantClient(url=QDRANT_URL)
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"[verify] collection: {QDRANT_COLLECTION}")
    print(f"[verify] points    : {info.points_count}")
    print(f"[verify] dim       : {info.config.params.vectors.size}")
    print(f"[verify] distance  : {info.config.params.vectors.distance}")
    # 按 arxiv 统计
    by_paper = {}
    offset = None
    while True:
        recs, offset = client.scroll(
            QDRANT_COLLECTION, limit=200, with_payload=True, with_vectors=False, offset=offset
        )
        for r in recs:
            aid = r.payload.get("arxiv_id", "?")
            by_paper[aid] = by_paper.get(aid, 0) + 1
        if offset is None:
            break
    print(f"[verify] chunks per paper:")
    for aid, meta in PAPERS.items():
        print(f"  {aid}  {meta['short_name']:<48}  {by_paper.get(aid, 0):>3} chunks")


if __name__ == "__main__":
    verify_neo4j()
    verify_qdrant()
    print("\n[verify] ALL DONE")
