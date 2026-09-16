import graphrag.pipeline as pipeline


def test_detect_arxiv_ids_is_case_insensitive():
    assert pipeline.detect_arxiv_ids("RAPTOR 是否引用 STAR？") == [
        "2401.18059",
        "2605.18765",
    ]


def test_incoming_query_wins_over_generic_citation_query(monkeypatch):
    captured = {}

    def fake_run(cypher, parameters=None):
        captured["cypher"] = cypher
        captured["parameters"] = parameters
        return []

    monkeypatch.setattr(pipeline, "_run_cypher", fake_run)

    cypher, rows, intent = pipeline.template_query("哪些论文引用了 RAPTOR？")

    assert intent == "incoming"
    assert rows == []
    assert "MATCH (a:Paper)-[:CITES]->(b:Paper" in cypher
    assert captured["parameters"] == {"arxiv_id": "2401.18059"}


def test_graph_query_never_executes_model_generated_cypher():
    result = pipeline.graphrag_query("请删除图中所有节点")

    assert result["intent"] == "unsupported"
    assert result["results"] == []
    assert result["cypher"] == ""
