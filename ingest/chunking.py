"""
分块：对 12 篇论文的纯文本做切片，存为 data/chunks.json
- 使用 langchain_text_splitters.RecursiveCharacterTextSplitter
- 参数：chunk_size=800, overlap=100（来自 config.py）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_OVERLAP, CHUNK_SIZE, PAPER_MAP_JSON, PAPERS_TEXT_DIR

sys.stdout.reconfigure(encoding="utf-8")
from langchain_text_splitters import RecursiveCharacterTextSplitter

OUT = PAPERS_TEXT_DIR.parent / "chunks.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

meta = json.loads(PAPER_MAP_JSON.read_text(encoding="utf-8"))
all_chunks = []
for m in meta:
    text = (PAPERS_TEXT_DIR / f"{m['arxiv_id']}.txt").read_text(encoding="utf-8")
    docs = splitter.create_documents([text], metadatas=[{
        "arxiv_id": m["arxiv_id"],
        "filename": m["filename"],
    }])
    for i, d in enumerate(docs):
        all_chunks.append({
            "chunk_id": f"{m['arxiv_id']}_chunk_{i:04d}",
            "arxiv_id": m["arxiv_id"],
            "filename": m["filename"],
            "text": d.page_content,
            "char_len": len(d.page_content),
        })
    print(f"  {m['arxiv_id']}: {len(docs)} chunks")

OUT.write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
)
total_chars = sum(c["char_len"] for c in all_chunks)
print(f"[chunking] {len(all_chunks)} chunks, {total_chars} chars total -> {OUT}")
print(f"[chunking] avg chunk = {total_chars // max(len(all_chunks), 1)} chars")
