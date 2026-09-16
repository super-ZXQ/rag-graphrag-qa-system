"""Local hybrid retrieval with reciprocal-rank fusion."""
from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from config import (
    EMBED_MODEL,
    HYBRID_RETRIEVER_K,
    LEXICAL_RETRIEVER_K,
    OLLAMA_BASE_URL,
    QDRANT_COLLECTION,
    VECTOR_RETRIEVER_K,
)
from paper_library.store import PaperStore
from paper_library.vector_client import get_qdrant_client


def reciprocal_rank_fusion(result_lists: list[list[dict]], limit: int, k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, 1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            items[chunk_id] = {**items.get(chunk_id, {}), **item}
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [{**items[chunk_id], "score": scores[chunk_id]} for chunk_id in ordered]


class HybridRetriever:
    def __init__(self, store: PaperStore | None = None, client=None, embeddings=None):
        self.store = store or PaperStore()
        self.client = client or get_qdrant_client()
        self.embeddings = embeddings or OllamaEmbeddings(
            model=EMBED_MODEL, base_url=OLLAMA_BASE_URL
        )

    def vector_search(self, query: str, limit: int = VECTOR_RETRIEVER_K) -> list[dict]:
        vector = self.embeddings.embed_query(query)
        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id", ""),
                    "paper_id": payload.get("paper_id") or payload.get("arxiv_id", ""),
                    "page_number": int(payload.get("page_number", 0)),
                    "title": payload.get("title") or payload.get("filename", ""),
                    "filename": payload.get("filename", ""),
                    "text": payload.get("text", ""),
                    "source_uri": payload.get("source_uri"),
                    "vector_score": float(hit.score),
                }
            )
        return results

    def search(self, query: str, limit: int = HYBRID_RETRIEVER_K) -> list[dict]:
        lexical = self.store.search_lexical(query, LEXICAL_RETRIEVER_K)
        try:
            vector = self.vector_search(query, VECTOR_RETRIEVER_K)
        except Exception:
            vector = []
        return reciprocal_rank_fusion([vector, lexical], limit=limit)
