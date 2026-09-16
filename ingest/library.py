"""动态资料摄取：PDF/Markdown → 带页码的本地证据分块。"""
import hashlib
import re
from io import BytesIO
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOADS_DIR
from paper_library.store import PaperStore

ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")


class PdfIngestionError(ValueError):
    """用户上传的文件无法安全转换为可检索文本。"""


def clean_text(text: str) -> str:
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def infer_title(pages: list[str], fallback: str) -> str:
    for page in pages:
        for line in page.splitlines():
            compact = " ".join(line.split())
            lowered = compact.lower()
            if len(compact) >= 8 and not lowered.startswith(("published as", "arxiv:")):
                return compact[:240]
    return Path(fallback).stem[:240] or "Untitled paper"


def build_page_chunks(paper_id: str, pages: list[str]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for page_number, page_text in enumerate(pages, 1):
        for index, text in enumerate(splitter.split_text(page_text)):
            chunks.append(
                {
                    "chunk_id": f"{paper_id}:p{page_number}:c{index}",
                    "page_number": page_number,
                    "text": text,
                }
            )
    return chunks


def ingest_pdf_bytes(
    content: bytes,
    filename: str,
    store: PaperStore | None = None,
    source_type: str = "internal_upload",
    source_uri: str | None = None,
    visibility: str = "internal",
) -> tuple[dict, bool]:
    """保存一篇 PDF，返回 (论文记录, 是否为首次导入)。"""
    if not content:
        raise PdfIngestionError("上传的文件为空。")
    if not filename.lower().endswith(".pdf"):
        raise PdfIngestionError("仅支持 PDF 文件。")

    store = store or PaperStore()
    content_hash = hashlib.sha256(content).hexdigest()
    existing = store.get_by_hash(content_hash)
    if existing:
        return existing, False

    match = ARXIV_RE.search(filename)
    paper_id = match.group(1) if match else f"local-{content_hash[:12]}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOADS_DIR / f"{content_hash}.pdf"
    paper = {
        "paper_id": paper_id,
        "content_hash": content_hash,
        "filename": Path(filename).name,
        "title": Path(filename).stem,
        "stored_path": str(stored_path),
        "source_type": source_type,
        "source_uri": source_uri,
        "visibility": visibility,
    }

    try:
        reader = PdfReader(BytesIO(content))
        pages = [clean_text(page.extract_text() or "") for page in reader.pages]
        if not any(page.strip() for page in pages):
            raise PdfIngestionError("无法从 PDF 提取文本；扫描件暂不支持。")

        paper["title"] = infer_title(pages, filename)
        chunks = build_page_chunks(paper_id, pages)
        if not chunks:
            raise PdfIngestionError("PDF 没有可供检索的文本块。")

        stored_path.write_bytes(content)
        return store.save_processed_paper(paper, pages, chunks), True
    except PdfIngestionError:
        raise
    except Exception as exc:
        store.save_failed_paper(paper, "PDF 解析失败")
        raise PdfIngestionError("PDF 解析失败，请确认文件未损坏且包含可选中文本。") from exc


def ingest_markdown_bytes(
    content: bytes,
    filename: str,
    store: PaperStore | None = None,
    source_type: str = "internal_upload",
    source_uri: str | None = None,
    visibility: str = "internal",
) -> tuple[dict, bool]:
    """保存 Markdown；逻辑页用一级标题分隔，便于生成稳定证据位置。"""
    if not content:
        raise PdfIngestionError("上传的文件为空。")
    if not filename.lower().endswith((".md", ".markdown")):
        raise PdfIngestionError("仅支持 PDF 或 Markdown 文件。")
    try:
        text = clean_text(content.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PdfIngestionError("Markdown 必须使用 UTF-8 编码。") from exc
    if not text.strip():
        raise PdfIngestionError("Markdown 没有可供检索的文本。")

    store = store or PaperStore()
    content_hash = hashlib.sha256(content).hexdigest()
    existing = store.get_by_hash(content_hash)
    if existing:
        return existing, False

    paper_id = f"local-{content_hash[:12]}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOADS_DIR / f"{content_hash}.md"
    sections = re.split(r"(?m)(?=^#\s+)", text)
    pages = [section.strip() for section in sections if section.strip()]
    paper = {
        "paper_id": paper_id,
        "content_hash": content_hash,
        "filename": Path(filename).name,
        "title": infer_title(pages, filename).lstrip("# "),
        "stored_path": str(stored_path),
        "source_type": source_type,
        "source_uri": source_uri,
        "visibility": visibility,
    }
    chunks = build_page_chunks(paper_id, pages)
    stored_path.write_bytes(content)
    return store.save_processed_paper(paper, pages, chunks), True


def ingest_document_bytes(
    content: bytes,
    filename: str,
    store: PaperStore | None = None,
    **metadata,
) -> tuple[dict, bool]:
    if filename.lower().endswith(".pdf"):
        return ingest_pdf_bytes(content, filename, store=store, **metadata)
    if filename.lower().endswith((".md", ".markdown")):
        return ingest_markdown_bytes(content, filename, store=store, **metadata)
    raise PdfIngestionError("仅支持 PDF 或 Markdown 文件。")
