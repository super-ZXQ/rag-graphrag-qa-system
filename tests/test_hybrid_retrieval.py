from retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_rewards_chunks_found_by_both_retrievers():
    vector = [{"chunk_id": "shared"}, {"chunk_id": "vector-only"}]
    lexical = [{"chunk_id": "lexical-only"}, {"chunk_id": "shared"}]

    fused = reciprocal_rank_fusion([vector, lexical], limit=3)

    assert fused[0]["chunk_id"] == "shared"
    assert {item["chunk_id"] for item in fused} == {"shared", "vector-only", "lexical-only"}
