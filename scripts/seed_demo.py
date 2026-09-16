"""Load the fictional enterprise materials; optionally index them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.indexer import IndexingError, index_paper
from ingest.library import ingest_markdown_bytes
from paper_library.store import PaperStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="also embed and upsert to Qdrant")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "demo_data" / "internal"
    store = PaperStore()
    imported = []
    for path in sorted(root.glob("*.md")):
        if path.name == "README.md":
            continue
        paper, created = ingest_markdown_bytes(
            path.read_bytes(),
            path.name,
            store=store,
            source_type="demo_internal",
            visibility="internal",
        )
        imported.append(paper["paper_id"])
        print(f"{'imported' if created else 'exists'}: {paper['title']}")
    if args.index:
        for paper_id in imported:
            try:
                index_paper(paper_id, store)
                print(f"indexed: {paper_id}")
            except IndexingError as exc:
                print(f"index failed: {paper_id}: {exc}")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
