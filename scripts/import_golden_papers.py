"""Import the version-pinned public papers used by the golden demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.indexer import index_paper
from ingest.library import ingest_pdf_bytes
from paper_library.store import PaperStore
from sources.public import download_arxiv_pdf

PAPERS = {
    "RAPTOR": ("2401.18059v1", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"),
    "CRAG": ("2401.15884v1", "Corrective Retrieval Augmented Generation"),
    "FLARE": ("2305.06983v2", "Active Retrieval Augmented Generation"),
    "GraphRAG": ("2404.16130v2", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    store = PaperStore()
    for name, (arxiv_id, title) in PAPERS.items():
        print(f"downloading {name} ({arxiv_id})", flush=True)
        content = download_arxiv_pdf(arxiv_id)
        paper, created = ingest_pdf_bytes(
            content,
            f"{arxiv_id}.pdf",
            store=store,
            source_type="public_arxiv",
            source_uri=f"https://arxiv.org/abs/{arxiv_id}",
            visibility="public",
        )
        store.update_paper_title(paper["paper_id"], title)
        paper = store.get_paper(paper["paper_id"]) or paper
        print(f"{'imported' if created else 'exists'}: {paper['title']}", flush=True)
        if args.index and paper["status"] != "ready":
            index_paper(paper["paper_id"], store)
            print(f"indexed: {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
