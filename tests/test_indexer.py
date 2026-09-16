from ingest.indexer import PaperIndexer
from paper_library.store import PaperStore


class FakeClient:
    def __init__(self):
        self.points = []

    def collection_exists(self, name):
        return False

    def create_collection(self, **kwargs):
        self.created = kwargs

    def upsert(self, collection_name, points, wait):
        self.points.extend(points)


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1] * 2560 for _ in texts]


def test_incremental_indexer_sets_ready_and_keeps_provenance(tmp_path):
    store = PaperStore(tmp_path / "library.sqlite3")
    store.save_processed_paper(
        {
            "paper_id": "paper-1",
            "content_hash": "c" * 64,
            "filename": "paper.md",
            "title": "Paper",
            "stored_path": str(tmp_path / "paper.md"),
            "source_type": "demo_internal",
            "visibility": "internal",
        },
        ["evidence"],
        [{"chunk_id": "paper-1:p1:c0", "page_number": 1, "text": "evidence"}],
    )
    client = FakeClient()

    paper = PaperIndexer(store, client, FakeEmbeddings()).index_paper("paper-1")

    assert paper["status"] == "ready"
    assert len(client.points) == 1
    assert client.points[0].payload["page_number"] == 1
    assert client.points[0].payload["visibility"] == "internal"
