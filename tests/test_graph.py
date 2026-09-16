from types import SimpleNamespace

from graph.domain import ARXIV_FILENAME_TO_TECHNIQUE
from graph.expansion import GraphExpander
from graph.store import CYPHER_RELATED, CitationGraph, Neo4jCitationGraph, get_graph


def _seed_graph(path) -> CitationGraph:
    graph = CitationGraph(path)
    seeds = [
        ("Wraptor", "2401.18059", "raptor", "RAPTOR", 3858),
        ("Wcrag", "2401.15884", "crag", "CRAG", 275),
    ]
    for wid, arxiv, key, title, cited in seeds:
        graph.upsert_paper(
            {"openalex_id": wid, "arxiv_id": arxiv, "title": title,
             "year": 2024, "cited_by_count": cited, "doi": ""},
            is_seed=True, technique_key=key)
    # RAPTOR 被一篇高被引论文引用，且引用一篇参考文献
    graph.upsert_paper({"openalex_id": "Wciting", "arxiv_id": None, "title": "High Impact Survey",
                        "year": 2025, "cited_by_count": 500, "doi": ""})
    graph.upsert_paper({"openalex_id": "Wref", "arxiv_id": None, "title": "Older Ref",
                        "year": 2020, "cited_by_count": 12, "doi": ""})
    graph.upsert_edge("Wciting", "Wraptor", "CITES")
    graph.upsert_edge("Wraptor", "Wref", "CITES")
    return graph


def test_related_papers_ordered_and_bounded(tmp_path):
    graph = _seed_graph(tmp_path / "g.db")
    related = graph.related_papers("raptor", limit=1)
    assert len(related) == 1
    assert related[0]["openalex_id"] == "Wciting"  # cited_by_count 500 排第一


def test_evidence_breadth_single_paper_dependency(tmp_path):
    graph = _seed_graph(tmp_path / "g.db")
    assert graph.evidence_breadth("raptor")["neighbor_count"] == 2
    assert graph.evidence_breadth("raptor")["single_paper_dependency"] is True
    assert graph.evidence_breadth("raptor")["evidence_breadth_status"] == "unverified"
    assert graph.evidence_breadth("crag")["single_paper_dependency"] is True
    assert graph.evidence_breadth("flare")["seed_count"] == 0


def test_technique_requirement_papers_stance(tmp_path):
    graph = _seed_graph(tmp_path / "g.db")
    conflict = graph.technique_requirement_papers("crag", "req_privacy")
    assert conflict["stance"] == "conflict"
    assert conflict["papers"][0]["openalex_id"] == "Wcrag"
    unknown = graph.technique_requirement_papers("crag", "req_global_questions")
    assert unknown["stance"] == "unknown"


def test_fixed_cypher_is_parameterized():
    assert "$technique_key" in CYPHER_RELATED and "$limit" in CYPHER_RELATED


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, limit=8):
        return [dict(r) for r in self.rows[:limit]]


def _rows():
    # CRAG 论文 3 块 + RAPTOR 1 块；CRAG 初始分数略高
    rows = []
    for i in range(3):
        rows.append({"chunk_id": f"crag-{i}", "paper_id": "pcrag", "filename": "2401.15884v1.pdf",
                     "title": "CRAG", "score": 0.10 - i * 0.01, "text": "crag"})
    rows.append({"chunk_id": "raptor-0", "paper_id": "praptor", "filename": "2401.18059v1.pdf",
                 "title": "RAPTOR", "score": 0.07, "text": "raptor"})
    return rows


def test_graph_expansion_applies_citation_boost_on_technique_query(tmp_path):
    graph = _seed_graph(tmp_path / "g.db")
    expander = GraphExpander(retriever=FakeRetriever(_rows()), graph=graph)
    out = expander.search("RAPTOR 与 CRAG 怎么选", limit=4)
    assert len(out) == 4
    assert len({c["chunk_id"] for c in out}) == 4
    # 技术类问题：所有结果都带引用背书，且引用更高的 RAPTOR 获得更高背书
    assert all("graph_boost" in c for c in out)
    raptor = next(c for c in out if c["filename"] == "2401.18059v1.pdf")
    crag = next(c for c in out if c["filename"] == "2401.15884v1.pdf")
    assert raptor["graph_boost"] > crag["graph_boost"]
    assert raptor["graph_cited_by"] == 3858
    # 背书是弱信号：不压过检索分数，CRAG 首块仍排第一
    assert out[0]["chunk_id"] == "crag-0"


def test_graph_expansion_gated_off_for_non_technique_query(tmp_path):
    graph = _seed_graph(tmp_path / "g.db")
    rows = _rows()
    expander = GraphExpander(retriever=FakeRetriever(rows), graph=graph)
    out = expander.search("公司明年的市场预算是多少", limit=3)
    # 非技术问题即使图谱就绪也不改变 hybrid 排序
    assert [c["chunk_id"] for c in out] == [c["chunk_id"] for c in rows[:3]]
    assert all("graph_boost" not in c for c in out)


def test_graph_expansion_degrades_without_graph(tmp_path):
    empty = CitationGraph(tmp_path / "empty.db")
    rows = _rows()
    expander = GraphExpander(retriever=FakeRetriever(rows), graph=empty)
    out = expander.search("RAPTOR CRAG", limit=5)
    assert [c["chunk_id"] for c in out] == [c["chunk_id"] for c in rows]  # 退回原 hybrid 排序


def test_get_graph_falls_back_to_sqlite_when_neo4j_down():
    graph = get_graph(prefer_neo4j=True)
    assert isinstance(graph, CitationGraph)
    assert not isinstance(graph, Neo4jCitationGraph)


def test_neo4j_constructor_raises_when_unreachable():
    import pytest
    with pytest.raises(Exception):
        Neo4jCitationGraph(uri="bolt://127.0.0.1:9999", user="neo4j", password="x")


def test_evidence_chain_section_renders_from_graph(tmp_path):
    from research.workflow import build_evidence_chain_section

    graph = _seed_graph(tmp_path / "g.db")
    expander = GraphExpander(retriever=FakeRetriever(_rows()), graph=graph)
    section = build_evidence_chain_section(
        ["基础混合检索 RAG", "RAPTOR", "CRAG", "FLARE", "GraphRAG"], expander=expander
    )
    assert "## 推荐证据链（引用图）" in section
    assert "### RAPTOR" in section
    assert "独立多论文支持：未验证" in section
    assert "Wciting" in section
    # 无种子的候选不渲染
    assert "### FLARE" not in section


def test_evidence_chain_section_empty_when_graph_missing(tmp_path):
    from research.workflow import build_evidence_chain_section

    empty = CitationGraph(tmp_path / "empty.db")
    expander = GraphExpander(retriever=FakeRetriever(_rows()), graph=empty)
    section = build_evidence_chain_section(["RAPTOR", "CRAG"], expander=expander)
    assert section == ""


def test_title_match_rejects_homonym_noise():
    from sources.openalex_graph import _title_matches

    assert _title_matches(
        {"title": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"},
        "RAPTOR",
    )
    assert not _title_matches(
        {"title": "AMPK Phosphorylation of Raptor Mediates a Metabolic Checkpoint"},
        "RAPTOR",
    )
    assert not _title_matches(
        {"title": "Soil and karst aquifer ... Crag Cave ..."},
        "CRAG",
    )
    assert not _title_matches(
        {"title": "Crag Is a GEF for Rab11 Required for Rhodopsin Trafficking"},
        "Corrective Retrieval",
    )
    assert _title_matches(
        {"title": "Corrective Retrieval Augmented Generation"},
        "Corrective Retrieval",
    )
    # 多词短语禁止词袋宽松匹配
    assert not _title_matches(
        {"title": "From local explanations to global understanding with explainable AI for trees"},
        "From Local to Global",
    )
    assert _title_matches(
        {"title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"},
        "From Local to Global",
    )


def test_year_plausible_rejects_old_homonyms():
    from sources.openalex_graph import _year_plausible

    assert _year_plausible({"publication_year": 2024}, "2404.16130")
    assert not _year_plausible({"publication_year": 2012}, "2404.16130")
    assert not _year_plausible({"publication_year": 2020}, "2404.16130")


def test_graph_expander_lazy_retriever(tmp_path):
    graph = _seed_graph(tmp_path / "g.db")
    expander = GraphExpander(graph=graph)
    # evidence_chain 不应触发 HybridRetriever / Qdrant
    chain = expander.evidence_chain("raptor")
    assert chain["seed_count"] == 1
    assert expander._retriever is None
