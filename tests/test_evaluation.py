from evaluation.run_retrieval import is_relevant, retrieval_metrics


def test_retrieval_metrics_match_source_and_chunk_id():
    cases = [
        {"id": "a", "expected_behavior": "answer", "relevant_source": "a.md",
         "relevant_chunk_ids": ["a:p1:c0"]},
        {"id": "b", "expected_behavior": "answer", "relevant_source": "b.md",
         "relevant_chunk_ids": ["b:p1:c0"]},
        {"id": "x", "expected_behavior": "refuse", "relevant_source": "",
         "relevant_chunk_ids": []},
    ]
    results = [
        [{"filename": "a.md", "chunk_id": "a:p1:c0"}],   # 命中（rank 1）
        [{"filename": "other.md", "chunk_id": "z:p1:c0"},
         {"filename": "b.md", "chunk_id": "b:p1:c0"}],    # 命中（rank 2）
        [{"filename": "a.md", "chunk_id": "a:p1:c0"}],    # 拒答类，不计入召回
    ]

    metrics = retrieval_metrics(cases, results)
    assert metrics["answer_case_count"] == 2
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.75  # 1/1 + 1/2


def test_chunk_labels_take_precedence_over_source_fallback():
    case = {"relevant_source": "a.md", "relevant_chunk_ids": ["a:p2:c1"]}
    assert is_relevant({"filename": "other.md", "chunk_id": "a:p2:c1"}, case)
    assert not is_relevant({"filename": "a.md", "chunk_id": "a:p9:c9"}, case)
    assert not is_relevant({"filename": "b.md", "chunk_id": "b:p1:c0"}, case)


def test_cross_document_case_requires_all_labeled_chunks():
    case = {
        "id": "cross",
        "expected_behavior": "answer",
        "relevant_source": "paper.pdf",
        "relevant_sources": ["paper.pdf", "internal.md"],
        "relevant_chunk_ids": ["paper:p1:c0", "internal:p1:c0"],
        "query_type": "cross_document",
    }

    incomplete = retrieval_metrics([case], [[{"filename": "paper.pdf", "chunk_id": "paper:p1:c0"}]])
    complete = retrieval_metrics(
        [case],
        [[
            {"filename": "paper.pdf", "chunk_id": "paper:p1:c0"},
            {"filename": "internal.md", "chunk_id": "internal:p1:c0"},
        ]],
    )

    assert incomplete["recall_at_5"] == 0.0
    assert complete["recall_at_5"] == 1.0
    assert complete["mrr"] == 0.5
