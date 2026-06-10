"""临时小工具：验证 config.py 里的所有依赖连接"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import self_check, QDRANT_URL

self_check()

# Qdrant ping
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, timeout=5)
    info = client.get_collections()
    print(f"[qdrant] OK, collections: {[c.name for c in info.collections]}")
except Exception as e:
    print(f"[qdrant] FAIL: {e}")

# Neo4j ping
try:
    from neo4j import GraphDatabase
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        with driver.session() as s:
            r = s.run("RETURN 1 AS x").single()
            print(f"[neo4j] OK, test query = {r['x']}")
except Exception as e:
    print(f"[neo4j] FAIL: {e}")

# Ollama ping
try:
    import urllib.request, json
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"[ollama] OK, models: {models}")
except Exception as e:
    print(f"[ollama] FAIL: {e} (请先安装并启动 Ollama)")

# Ollama embeddings dimension
try:
    from langchain_ollama import OllamaEmbeddings
    from config import EMBED_MODEL
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vec = emb.embed_query("test")
    print(f"[embedding] {EMBED_MODEL} dim = {len(vec)}")
    if len(vec) != 2560:
        print(f"[embedding] ⚠️  config.VECTOR_SIZE 应该是 {len(vec)}，不是 2560")
except Exception as e:
    print(f"[embedding] FAIL: {e}")
