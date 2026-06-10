"""
向量化 + 入 Qdrant
- 读 chunks.json
- 用 OllamaEmbeddings(qwen3-embedding:4b) 嵌入
- 写入 Qdrant collection "papers"，向量维度 2560，距离用 cosine
- metadata (arxiv_id, chunk_id, filename) 一起存，方便溯源
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHUNKS_FILE, EMBED_MODEL, PROJECT_ROOT, QDRANT_COLLECTION, QDRANT_URL, VECTOR_SIZE,
)

sys.stdout.reconfigure(encoding="utf-8")
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def main():
    print(f"[build_qdrant] reading {CHUNKS_FILE}")
    chunks = json.loads(Path(CHUNKS_FILE).read_text(encoding="utf-8"))
    print(f"[build_qdrant] {len(chunks)} chunks")

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(QDRANT_COLLECTION):
        print(f"[build_qdrant] dropping existing '{QDRANT_COLLECTION}'")
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[build_qdrant] created '{QDRANT_COLLECTION}' (dim={VECTOR_SIZE}, cosine)")

    print(f"[build_qdrant] embedding model = {EMBED_MODEL}")
    # 4b embedding 模型在 GPU 上 ~0.1s/chunk；CPU 上要 1s+/chunk，差异巨大
    # RTX 4060 8GB 跑 4b 模型余量充足，默认 GPU 跑即可
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    BATCH = 16  # GPU 批大点吞吐高

    # 缓存：避免重跑要重新嵌入（每次 2~3 分钟）
    import numpy as np
    cache = PROJECT_ROOT / "data" / "vectors.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape == (len(texts), VECTOR_SIZE):
        all_vecs = np.load(cache).tolist()
        print(f"[build_qdrant] loaded {len(all_vecs)} vectors from cache ({cache})")
    else:
        all_vecs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            for attempt in range(3):
                try:
                    vecs = emb.embed_documents(batch)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"  [retry {attempt+1}/3] batch {i//BATCH}: {e}")
                    time.sleep(2 ** attempt)
            all_vecs.extend(vecs)
            done = len(all_vecs)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  embedded {done}/{len(texts)} ({elapsed:.1f}s, {rate:.1f} chunks/s)")
        # 存缓存
        np.save(cache, np.asarray(all_vecs, dtype=np.float32))
        print(f"[build_qdrant] cached {len(all_vecs)} vectors -> {cache}")
    print(f"[build_qdrant] {len(all_vecs)} vectors in {time.time()-t0:.1f}s")

    # upsert（分批，避免单次请求过大）
    # Qdrant 只接受 uint64 或 UUID 作 point id，不能用字符串
    # 用 chunk 的全局下标做 id，并把可读的 chunk_id 存在 payload 里
    UPSERT_BATCH = 100
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch_chunks = chunks[i:i+UPSERT_BATCH]
        batch_vecs = all_vecs[i:i+UPSERT_BATCH]
        points = [
            PointStruct(
                id=i + j,  # uint64
                vector=v,
                payload={
                    "arxiv_id": c["arxiv_id"],
                    "chunk_id": c["chunk_id"],
                    "filename": c["filename"],
                    "text": c["text"],
                },
            )
            for j, (c, v) in enumerate(zip(batch_chunks, batch_vecs))
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        print(f"  upserted {i+len(points)}/{len(chunks)}")

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"[build_qdrant] DONE. collection has {info.points_count} points")


if __name__ == "__main__":
    main()
