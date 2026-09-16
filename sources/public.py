"""Narrow arXiv/OpenAlex clients with timeouts and no arbitrary URL fetching."""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import OPENALEX_API_KEY, OPENALEX_MAILTO

ARXIV_ID_RE = re.compile(r"^(?:[a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.I)
USER_AGENT = f"ScholarGraph/0.2 ({OPENALEX_MAILTO or 'local-research-demo'})"


class PublicSourceError(RuntimeError):
    pass


def _request(url: str, attempts: int = 3) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/atom+xml"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=15) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise PublicSourceError(f"公开资料服务请求失败（HTTP {exc.code}）。") from exc
            time.sleep(2 ** attempt)
        except (URLError, TimeoutError) as exc:
            if attempt == attempts - 1:
                raise PublicSourceError("公开资料服务暂时不可用。") from exc
            time.sleep(2 ** attempt)
    raise PublicSourceError("公开资料服务暂时不可用。")


def search_arxiv(query: str, limit: int = 8) -> list[dict]:
    phrase = " ".join(query.replace('"', "").split())
    params = urlencode({"search_query": f'all:"{phrase}"', "start": 0, "max_results": limit})
    root = ET.fromstring(_request(f"https://export.arxiv.org/api/query?{params}"))
    atom = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", atom):
        raw_id = (entry.findtext("a:id", default="", namespaces=atom).rstrip("/").split("/")[-1])
        results.append(
            {
                "provider": "arxiv",
                "external_id": raw_id,
                "title": " ".join(entry.findtext("a:title", default="", namespaces=atom).split()),
                "summary": " ".join(entry.findtext("a:summary", default="", namespaces=atom).split()),
                "published": entry.findtext("a:published", default="", namespaces=atom),
                "authors": [author.findtext("a:name", default="", namespaces=atom) for author in entry.findall("a:author", atom)],
                "source_uri": f"https://arxiv.org/abs/{raw_id}",
                "pdf_uri": f"https://arxiv.org/pdf/{raw_id}",
            }
        )
    return results


def search_openalex(query: str, limit: int = 8) -> list[dict]:
    params = {"search": query, "per-page": limit, "select": "id,doi,title,publication_year,authorships,cited_by_count,primary_location"}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    payload = json.loads(_request(f"https://api.openalex.org/works?{urlencode(params)}"))
    results = []
    for work in payload.get("results", []):
        authors = [item.get("author", {}).get("display_name", "") for item in work.get("authorships", [])]
        location = work.get("primary_location") or {}
        results.append(
            {
                "provider": "openalex",
                "external_id": work.get("id", "").rsplit("/", 1)[-1],
                "title": work.get("title", ""),
                "published": work.get("publication_year"),
                "authors": authors,
                "cited_by_count": work.get("cited_by_count", 0),
                "source_uri": work.get("doi") or work.get("id"),
                "landing_page": location.get("landing_page_url"),
            }
        )
    return results


def download_arxiv_pdf(arxiv_id: str) -> bytes:
    if not ARXIV_ID_RE.fullmatch(arxiv_id):
        raise PublicSourceError("arXiv ID 格式无效。")
    return _request(f"https://arxiv.org/pdf/{arxiv_id}")
