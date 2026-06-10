"""
从 12 篇 PDF 提取纯文本，每篇一个 .txt 文件
- 输入: ARTICLES_DIR 下的 *.pdf
- 输出: data/papers_text/{arxiv_id}.txt
- 同时输出 data/papers_meta.json，包含每篇的 arxiv_id、页数、首段（用于识别标题）
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTICLES_DIR, PAPERS_TEXT_DIR, PAPER_MAP_JSON

sys.stdout.reconfigure(encoding="utf-8")
PAPERS_TEXT_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")  # 抓 arxiv 编号


def clean_text(text: str) -> str:
    """pypdf 偶尔会吐出 UTF-16 代理字符（形式如 \\uDCXX），
    Python utf-8 编码器拒绝它们。先用 surrogatepass 编码再用 replace 解码，
    把孤立代理换成 U+FFFD。"""
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def extract_one(pdf_path: Path) -> dict:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(clean_text(text))
    full = "\n\n".join(pages)
    # 文件名里的 arxiv 编号（去掉版本号 v1/v2...）
    arxiv_id = ARXIV_RE.search(pdf_path.stem).group(1)
    return {
        "arxiv_id": arxiv_id,
        "filename": pdf_path.name,
        "pages": len(pages),
        "first_page": pages[0] if pages else "",
        "full_text": full,
    }


def main():
    pdfs = sorted(ARTICLES_DIR.glob("*.pdf"))
    print(f"[pdf_to_text] found {len(pdfs)} pdfs in {ARTICLES_DIR}")
    meta = []
    for pdf in pdfs:
        print(f"  - {pdf.name} ...", end=" ", flush=True)
        info = extract_one(pdf)
        # 存文本
        out_txt = PAPERS_TEXT_DIR / f"{info['arxiv_id']}.txt"
        out_txt.write_text(info["full_text"], encoding="utf-8")
        # 存 meta（不含全文，避免 json 臃肿）
        meta.append({k: v for k, v in info.items() if k != "full_text"})
        print(f"{info['pages']} pages, {len(info['full_text'])} chars -> {out_txt.name}")
    PAPER_MAP_JSON.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[pdf_to_text] meta -> {PAPER_MAP_JSON}")


if __name__ == "__main__":
    main()
