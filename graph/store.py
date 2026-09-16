"""引用图存储与固定模板业务查询。

- 默认 SQLite 后端（data/citation_graph.sqlite3），离线可跑、可测试；
- 可选 Neo4j 后端，所有 Cypher 都是固定模板 + 参数化输入、只读、带 LIMIT；
- 不提供任何自由文本 Cypher 执行接口；Neo4j 不可用时调用方降级到 SQLite/hybrid。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DATA_DIR, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from graph.domain import FITS, REQUIREMENTS, TECHNIQUES

GRAPH_DB = DATA_DIR / "citation_graph.sqlite3"

# ---------- 固定 Cypher 模板（只读、参数化、限界），禁止拼接用户输入 ----------
CYPHER_RELATED = """
MATCH (p:Paper)-[r:CITES|RELATED_TO]->(n:Paper)
WHERE p.technique_key = $technique_key OR p.arxiv_id = $arxiv_id
RETURN n.openalex_id AS openalex_id, n.title AS title, n.year AS year,
       n.cited_by_count AS cited_by_count, type(r) AS relation
ORDER BY n.cited_by_count DESC
LIMIT $limit
"""
CYPHER_BREADTH = """
MATCH (p:Paper)
WHERE p.technique_key = $technique_key
OPTIONAL MATCH (p)-[:CITES|RELATED_TO]-(n:Paper)
RETURN count(DISTINCT p) AS seed_count, count(DISTINCT n) AS neighbor_count
"""
CYPHER_TECH_REQ = """
MATCH (p:Paper)-[:DISCUSSES]->(t:Technique {key:$technique_key})
OPTIONAL MATCH (t)-[f:FITS]->(r:Requirement {id:$requirement_id})
RETURN p.openalex_id AS openalex_id, p.title AS title, f.stance AS stance,
       f.rationale AS rationale
LIMIT $limit
"""


class CitationGraph:
    """SQLite 引用图后端。"""

    def __init__(self, path: Path = GRAPH_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _conn(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._conn() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_papers (
                    openalex_id TEXT PRIMARY KEY,
                    arxiv_id TEXT,
                    title TEXT NOT NULL,
                    year INTEGER,
                    cited_by_count INTEGER NOT NULL DEFAULT 0,
                    doi TEXT,
                    is_seed INTEGER NOT NULL DEFAULT 0,
                    technique_key TEXT
                );
                CREATE TABLE IF NOT EXISTS graph_edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    PRIMARY KEY (src_id, dst_id, relation)
                );
                """
            )

    def upsert_paper(self, paper: dict, is_seed: bool = False, technique_key: str | None = None) -> None:
        with self._conn() as connection:
            connection.execute(
                """INSERT INTO graph_papers
                   (openalex_id, arxiv_id, title, year, cited_by_count, doi, is_seed, technique_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(openalex_id) DO UPDATE SET
                     title=excluded.title, year=excluded.year,
                     cited_by_count=excluded.cited_by_count, doi=excluded.doi,
                     is_seed=MAX(is_seed, excluded.is_seed),
                     technique_key=COALESCE(excluded.technique_key, technique_key)""",
                (
                    paper["openalex_id"], paper.get("arxiv_id"), paper.get("title", ""),
                    paper.get("year"), int(paper.get("cited_by_count", 0)), paper.get("doi", ""),
                    1 if is_seed else 0, technique_key,
                ),
            )

    def upsert_edge(self, src_id: str, dst_id: str, relation: str) -> None:
        if not src_id or not dst_id or src_id == dst_id:
            return
        with self._conn() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO graph_edges (src_id, dst_id, relation) VALUES (?, ?, ?)",
                (src_id, dst_id, relation),
            )

    def seed_paper(self, arxiv_id: str, technique_key: str) -> None:
        with self._conn() as connection:
            connection.execute(
                "UPDATE graph_papers SET is_seed=1, technique_key=?, arxiv_id=COALESCE(arxiv_id, ?) "
                "WHERE arxiv_id=? OR doi LIKE ?",
                (technique_key, arxiv_id, arxiv_id, f"%{arxiv_id}%"),
            )

    def seed_paper_by_openalex(self, openalex_id: str, arxiv_id: str, technique_key: str) -> None:
        with self._conn() as connection:
            connection.execute(
                "UPDATE graph_papers SET is_seed=1, technique_key=?, arxiv_id=? WHERE openalex_id=?",
                (technique_key, arxiv_id, openalex_id),
            )

    def _seed(self, technique_key: str) -> dict | None:
        with self._conn() as connection:
            row = connection.execute(
                "SELECT * FROM graph_papers WHERE technique_key=? AND is_seed=1 LIMIT 1",
                (technique_key,),
            ).fetchone()
        return dict(row) if row else None

    def related_papers(self, technique_key: str, limit: int = 8) -> list[dict]:
        """业务查询 1：某候选技术的高相关引用/被引论文（按被引数排序）。"""
        seed = self._seed(technique_key)
        if not seed:
            return []
        with self._conn() as connection:
            rows = connection.execute(
                """SELECT n.openalex_id, n.title, n.year, n.cited_by_count, e.relation
                   FROM graph_edges e
                   JOIN graph_papers n ON n.openalex_id =
                       CASE WHEN e.src_id=? THEN e.dst_id ELSE e.src_id END
                   WHERE (e.src_id=? OR e.dst_id=?)
                   ORDER BY n.cited_by_count DESC
                   LIMIT ?""",
                (seed["openalex_id"], seed["openalex_id"], seed["openalex_id"], limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def evidence_breadth(self, technique_key: str) -> dict:
        """Return citation-neighbor breadth without treating metadata as evidence."""
        seed = self._seed(technique_key)
        if not seed:
            return {"technique_key": technique_key, "seed_count": 0, "neighbor_count": 0,
                    "independent_evidence_count": 0, "single_paper_dependency": True,
                    "evidence_breadth_status": "unverified"}
        with self._conn() as connection:
            neighbors = connection.execute(
                "SELECT COUNT(DISTINCT CASE WHEN src_id=? THEN dst_id ELSE src_id END) AS c "
                "FROM graph_edges WHERE src_id=? OR dst_id=?",
                (seed["openalex_id"], seed["openalex_id"], seed["openalex_id"]),
            ).fetchone()["c"]
        return {
            "technique_key": technique_key,
            "technique_name": TECHNIQUES.get(technique_key, {}).get("name", technique_key),
            "seed_count": 1,
            "neighbor_count": neighbors,
            "independent_evidence_count": 0,
            "single_paper_dependency": True,
            "evidence_breadth_status": "unverified",
        }

    def technique_requirement_papers(self, technique_key: str, requirement_id: str, limit: int = 8) -> dict:
        """业务查询 3：讨论某技术且关联某类企业约束的论文与契合度。"""
        seed = self._seed(technique_key)
        fit = FITS.get(technique_key, {}).get(requirement_id)
        return {
            "technique": TECHNIQUES.get(technique_key, {}).get("name", technique_key),
            "requirement": REQUIREMENTS.get(requirement_id, requirement_id),
            "stance": fit[0] if fit else "unknown",
            "rationale": fit[1] if fit else "图谱中没有该技术对此约束的确定性映射。",
            "papers": ([{
                "openalex_id": seed["openalex_id"], "title": seed["title"],
                "year": seed["year"], "cited_by_count": seed["cited_by_count"],
            }] if seed else []),
            "related": self.related_papers(technique_key, limit),
        }

    def cited_by_count(self, technique_key: str) -> int:
        seed = self._seed(technique_key)
        return int(seed["cited_by_count"]) if seed else 0


class Neo4jCitationGraph:
    """可选 Neo4j 后端：只用固定模板，只读、参数化、限界；不可用即抛错由调用方降级。"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=3)
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def related_papers(self, technique_key: str, limit: int = 8) -> list[dict]:
        arxiv_id = TECHNIQUES.get(technique_key, {}).get("arxiv_id") or ""
        with self.driver.session() as session:
            result = session.run(CYPHER_RELATED, technique_key=technique_key,
                                 arxiv_id=arxiv_id, limit=limit)
            return [dict(record) for record in result]

    def evidence_breadth(self, technique_key: str) -> dict:
        with self.driver.session() as session:
            record = session.run(CYPHER_BREADTH, technique_key=technique_key).single()
            data = dict(record) if record else {"seed_count": 0, "neighbor_count": 0}
        data["technique_key"] = technique_key
        data["independent_evidence_count"] = 0
        data["single_paper_dependency"] = True
        data["evidence_breadth_status"] = "unverified"
        return data

    def technique_requirement_papers(self, technique_key: str, requirement_id: str, limit: int = 8) -> dict:
        with self.driver.session() as session:
            records = [dict(r) for r in session.run(
                CYPHER_TECH_REQ, technique_key=technique_key,
                requirement_id=requirement_id, limit=limit)]
        fit = FITS.get(technique_key, {}).get(requirement_id)
        return {
            "technique": TECHNIQUES.get(technique_key, {}).get("name", technique_key),
            "requirement": REQUIREMENTS.get(requirement_id, requirement_id),
            "stance": fit[0] if fit else "unknown",
            "rationale": fit[1] if fit else "",
            "papers": records, "related": [],
        }


def get_graph(prefer_neo4j: bool = False):
    """优先尝试 Neo4j（显式要求且连通），否则返回 SQLite；任何异常都安全降级。"""
    if prefer_neo4j:
        try:
            return Neo4jCitationGraph()
        except Exception:
            pass
    return CitationGraph()
