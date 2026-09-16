"""Incremental paper indexing. It never drops an existing collection."""
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_COLLECTION,
    QDRANT_COLLECTION_VERSION,
    VECTOR_SIZE,
)
from paper_library.store import PaperStore
from paper_library.vector_client import get_qdrant_client


class IndexingError(RuntimeError):
    pass


class PaperIndexer:
    def __init__(self, store: PaperStore | None = None, client=None, embeddings=None):
        self.store = store or PaperStore()
        self.client = client or get_qdrant_client()
        self.embeddings = embeddings or OllamaEmbeddings(
            model=EMBED_MODEL, base_url=OLLAMA_BASE_URL
        )

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(QDRANT_COLLECTION):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            return
        size = self.client.get_collection(QDRANT_COLLECTION).config.params.vectors.size
        if size != VECTOR_SIZE:
            raise IndexingError(
                f"向量集合维度为 {size}，当前嵌入模型要求 {VECTOR_SIZE}；请更换集合版本。"
            )

    def index_paper(self, paper_id: str) -> dict:
        paper = self.store.get_paper(paper_id)
        if not paper:
            raise IndexingError("论文不存在。")
        chunks = self.store.list_chunks(paper_id)
        if not chunks:
            raise IndexingError("论文没有可索引分块。")

        self.store.update_paper_status(paper_id, "indexing")
        try:
            self._ensure_collection()
            vectors = self.embeddings.embed_documents([chunk["text"] for chunk in chunks])
            if any(len(vector) != VECTOR_SIZE for vector in vectors):
                raise IndexingError("嵌入维度与配置不一致。")
            points = []
            for chunk, vector in zip(chunks, vectors):
                points.append(
                    PointStruct(
                        id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])),
                        vector=vector,
                        payload={
                            "paper_id": paper_id,
                            "arxiv_id": paper_id if paper["source_type"] == "public_arxiv" else "",
                            "chunk_id": chunk["chunk_id"],
                            "page_number": chunk["page_number"],
                            "title": paper["title"],
                            "filename": paper["filename"],
                            "source_type": paper["source_type"],
                            "source_uri": paper["source_uri"],
                            "visibility": paper["visibility"],
                            "content_hash": paper["content_hash"],
                            "index_version": QDRANT_COLLECTION_VERSION,
                            "text": chunk["text"],
                        },
                    )
                )
            self.client.upsert(QDRANT_COLLECTION, points=points, wait=True)
            self.store.update_paper_status(
                paper_id, "ready", index_version=QDRANT_COLLECTION_VERSION
            )
            return self.store.get_paper(paper_id) or paper
        except Exception as exc:
            self.store.update_paper_status(paper_id, "index_failed", str(exc)[:500])
            if isinstance(exc, IndexingError):
                raise
            raise IndexingError("向量索引失败，请检查 Ollama 与 Qdrant。") from exc


def index_paper(paper_id: str, store: PaperStore | None = None) -> dict:
    return PaperIndexer(store=store).index_paper(paper_id)
