"""Qdrant client factory: server in production, local persistence for development/CI."""
from qdrant_client import QdrantClient

from config import QDRANT_PATH, QDRANT_URL


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH) if QDRANT_PATH else QdrantClient(url=QDRANT_URL)
