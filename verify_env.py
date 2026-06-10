"""临时小工具：验证 config.py 里的所有依赖连接"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMBED_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    QDRANT_URL,
    self_check,
)


def check_qdrant() -> None:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        info = client.get_collections()
        print(f"[qdrant] OK, collections: {[c.name for c in info.collections]}")
    except Exception as e:
        print(f"[qdrant] FAIL: {e}")


def check_neo4j() -> None:
    try:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            with driver.session() as s:
                r = s.run("RETURN 1 AS x").single()
                print(f"[neo4j] OK, test query = {r['x']}")
    except Exception as e:
        print(f"[neo4j] FAIL: {e}")


def check_ollama() -> None:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            print(f"[ollama] OK, models: {models}")
    except Exception as e:
        print(f"[ollama] FAIL: {e} (请先安装并启动 Ollama)")


def check_embedding() -> None:
    try:
        from langchain_ollama import OllamaEmbeddings
        emb = OllamaEmbeddings(model=EMBED_MODEL)
        vec = emb.embed_query("test")
        print(f"[embedding] {EMBED_MODEL} dim = {len(vec)}")
        if len(vec) != 2560:
            print(f"[embedding] ⚠️  config.VECTOR_SIZE 应该是 {len(vec)}，不是 2560")
    except Exception as e:
        print(f"[embedding] FAIL: {e}")


def main() -> None:
    self_check()
    check_qdrant()
    check_neo4j()
    check_ollama()
    check_embedding()


if __name__ == "__main__":
    main()
