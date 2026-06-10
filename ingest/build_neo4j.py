"""
把 12 篇论文 + 引用边写入 Neo4j
- 节点：(:Paper {arxiv_id, short_id, title, first_author, year, short_name})
- 关系：(:Paper)-[:CITES]->(:Paper)
- 节点 merge 模式（重复运行幂等）
- 关系用 UNWIND 批量写入
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

sys.stdout.reconfigure(encoding="utf-8")
from neo4j import GraphDatabase

from graph.citations import PAPERS, get_citations


def main():
    edges = get_citations()
    print(f"[build_neo4j] {len(PAPERS)} papers, {len(edges)} edges -> {NEO4J_URI}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as s:
        # 清空旧数据（避免测试残留）
        s.run("MATCH (n) DETACH DELETE n")
        print("[build_neo4j] cleared existing graph")

        # 1) 建节点
        s.run(
            """
            UNWIND $papers AS p
            MERGE (n:Paper {arxiv_id: p.arxiv_id})
            SET n.short_id    = p.short_id,
                n.title       = p.title,
                n.first_author= p.first_author,
                n.year        = p.year,
                n.short_name  = p.short_name
            """,
            papers=[
                {"arxiv_id": k, **v} for k, v in PAPERS.items()
            ],
        )
        n_nodes = s.run("MATCH (n:Paper) RETURN count(n) AS c").single()["c"]
        print(f"[build_neo4j] {n_nodes} Paper nodes created")

        # 2) 建关系
        s.run(
            """
            UNWIND $edges AS e
            MATCH (a:Paper {arxiv_id: e.src})
            MATCH (b:Paper {arxiv_id: e.dst})
            MERGE (a)-[:CITES]->(b)
            """,
            edges=[{"src": s_id, "dst": d_id} for s_id, d_id in edges],
        )
        n_edges = s.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
        print(f"[build_neo4j] {n_edges} CITES relations created")

    driver.close()
    print("[build_neo4j] DONE")


if __name__ == "__main__":
    main()
