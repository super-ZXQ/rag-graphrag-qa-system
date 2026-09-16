"""OpenAlex 引用图客户端：限流、429 退避、响应缓存与有界一跳扩展。

安全边界：
- 只访问 api.openalex.org，不把任意外部 URL 当作下载地址，不下载非公开 PDF；
- Key 可选，从环境变量读取；缓存落 data/openalex_cache（已被 git 忽略）；
- 邻居数量有上限，避免无限爬取。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import DATA_DIR, OPENALEX_API_KEY, OPENALEX_MAILTO

CACHE_DIR = DATA_DIR / "openalex_cache"
API_ROOT = "https://api.openalex.org"
_OPENALEX_ID_RE = re.compile(r"W\d+")
DEFAULT_NEIGHBOR_LIMIT = 6


class CitationGraphError(RuntimeError):
    pass


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:180]
    return CACHE_DIR / f"{safe}.json"


def _get(url: str, attempts: int = 4, ttl_cache: bool = True) -> dict:
    if ttl_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = _cache_path(url)
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
    params = {}
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    separator = "&" if "?" in url else "?"
    full = f"{url}{separator}{urlencode(params)}" if params else url
    request = Request(full, headers={"User-Agent": "ScholarGraph/0.3", "Accept": "application/json"})
    last_error = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if ttl_cache:
                _cache_path(url).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < attempts - 1:
                time.sleep(min(2 ** attempt, 20))
                continue
            raise CitationGraphError(f"OpenAlex 请求失败（HTTP {exc.code}）。") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(min(2 ** attempt, 20))
                continue
            raise CitationGraphError("OpenAlex 暂时不可用。") from exc
    raise CitationGraphError(f"OpenAlex 请求失败：{last_error}")


def openalex_id(url_or_id: str) -> str | None:
    match = _OPENALEX_ID_RE.search(url_or_id or "")
    return match.group(0) if match else None


def _title_matches(work: dict, title_hint: str) -> bool:
    """标题启发式：防止把 RAPTOR 匹到 mTOR、CRAG 匹到洞穴论文等同名噪声。"""
    if not title_hint:
        return True
    work_title = (work.get("title") or "").lower()
    hint = title_hint.lower().strip()
    if not work_title:
        return False
    # 短缩写（RAPTOR/CRAG/FLARE）：必须以该缩写作为标题开头独立出现
    if len(hint) <= 10 and re.fullmatch(r"[a-z0-9]+", hint):
        return bool(re.match(rf"^{re.escape(hint)}\b", work_title))
    # 多词标题片段：要求完整短语出现，或按词序连续匹配，禁止“词袋式”宽松匹配
    if hint in work_title:
        return True
    tokens = [t for t in re.findall(r"[a-z0-9]+", hint) if len(t) >= 2]
    if len(tokens) >= 2:
        pattern = r"\b" + r"\W+".join(re.escape(t) for t in tokens) + r"\b"
        return bool(re.search(pattern, work_title))
    return False


def _year_plausible(work: dict, arxiv_id: str) -> bool:
    """arXiv ID 前缀含年份（24 04 → 2024）；OpenAlex 年份相差超过 1 年则拒绝。"""
    try:
        year = 2000 + int(arxiv_id[:2])
    except (TypeError, ValueError):
        return True
    pub = work.get("publication_year")
    if pub is None:
        return True
    return year - 1 <= int(pub) <= year + 1


def _pick_works(results: list[dict], arxiv_id: str, title_hint: str) -> list[dict]:
    return [
        w for w in results
        if _title_matches(w, title_hint) and _year_plausible(w, arxiv_id)
    ]


def find_work_by_arxiv(arxiv_id: str, title_hint: str = "") -> dict | None:
    """定位 arXiv 论文的 OpenAlex 规范记录。

    优先 DOI（10.48550/arXiv.{id}），并用标题 + 年份双重校验，
    避免同名缩写 / 同短语误匹配。
    """
    select = (
        "id,title,doi,publication_year,cited_by_count,referenced_works,"
        "related_works,authorships,primary_location"
    )
    filters = (
        f"doi:10.48550/arxiv.{arxiv_id}",
        f"locations.landing_page_url:https://arxiv.org/abs/{arxiv_id}",
    )
    for filter_expr in filters:
        url = f"{API_ROOT}/works?" + urlencode(
            {"filter": filter_expr, "per-page": 5, "select": select}
        )
        plausible = _pick_works(_get(url).get("results", []), arxiv_id, title_hint)
        if plausible:
            return _pick_richest(plausible)
    if title_hint:
        search_url = f"{API_ROOT}/works?" + urlencode(
            {"search": title_hint, "per-page": 5, "select": select}
        )
        plausible = _pick_works(_get(search_url).get("results", []), arxiv_id, title_hint)
        if plausible:
            return plausible[0]
    return None


def _pick_richest(works: list[dict]) -> dict:
    return max(works, key=lambda w: (len(w.get("referenced_works", [])), w.get("cited_by_count", 0)))


def _compact_work(work: dict, relation: str) -> dict:
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])[:4]
    ]
    return {
        "openalex_id": openalex_id(work.get("id", "")),
        "title": work.get("title", ""),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "doi": (work.get("doi") or ""),
        "relation": relation,
        "authors": authors,
    }


def _batch_works(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    filter_expr = "openalex_id:" + "|".join(ids)
    url = f"{API_ROOT}/works?" + urlencode(
        {"filter": filter_expr, "per-page": len(ids),
         "select": "id,title,doi,publication_year,cited_by_count,authorships,primary_location"}
    )
    try:
        return _get(url).get("results", [])
    except CitationGraphError:
        return []


def citation_neighbors(work_id: str, limit: int = DEFAULT_NEIGHBOR_LIMIT) -> dict:
    """返回种子论文的高相关施引与参考文献（有界一跳）。"""
    seed = _get(f"{API_ROOT}/works/{work_id}")
    referenced_ids = [openalex_id(u) for u in seed.get("referenced_works", []) if openalex_id(u)][:limit]
    citing_url = (
        f"{API_ROOT}/works?" + urlencode(
            {"filter": f"cites:{work_id}", "per-page": limit,
             "sort": "cited_by_count:desc",
             "select": "id,title,doi,publication_year,cited_by_count,authorships,primary_location"}
        )
    )
    citing = _get(citing_url).get("results", [])
    referenced = _batch_works(referenced_ids)
    return {
        "seed": _compact_work(seed, "seed"),
        "citing": [_compact_work(w, "citing") for w in citing[:limit]],
        "referenced": [_compact_work(w, "referenced") for w in referenced[:limit]],
    }
