"""Claim 抽取与证据可信度校验。

设计原则：
- 硬校验是确定性的：引用 ID 必须存在、能定位到 paper/chunk/page，禁止虚构 ID；
- 软校验（claim 与证据是否语义匹配）默认用保守的词项重叠启发式，LLM 仅作辅助；
- 语义不确定一律降为 partially_supported / insufficient，绝不默认“支持”；
- 工程推断必须显式归类为 engineering_inference；证据不足时审批门槛不通过。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

EVIDENCE_RE = re.compile(r"\[(E-[A-Za-z0-9_-]+)(?:\s[^\]]*)?\]")

CLAIM_TYPES = ("paper_fact", "internal_fact", "engineering_inference", "insufficient_evidence")
SUPPORT_STATUSES = ("supported", "partially_supported", "conflicted", "insufficient")
KEY_FACT_TYPES = {"paper_fact", "internal_fact"}

INFERENCE_MARKERS = (
    "建议", "推断", "推测", "工程上", "实施成本", "改造成本", "运维", "权衡", "可能",
    "预计", "估算", "poc", "推荐草案", "倾向", "recommend", "estimate", "suggest",
    "trade-off", "overhead", "we recommend",
)
INSUFFICIENT_MARKERS = ("证据不足", "无法确认", "尚无证据", "没有证据", "insufficient evidence")
NEGATION_MARKERS = ("不支持", "没有提升", "无提升", "未提升", "does not", "do not", "no gain", "not improve")
STRUCTURAL_TITLES = {
    "report", "执行摘要", "需求与假设", "候选方案", "加权比较矩阵", "关键证据",
    "工程实施影响", "风险与未知项", "推荐草案", "poc计划", "poc 计划", "参考资料",
}

# 句子切句：中英文标点
_SENT_SPLIT = re.compile(r"[。！？!?\n]+|(?<=[.;])\s+")


def _sentences(markdown: str) -> list[str]:
    parts = []
    for block in re.split(r"```.*?```", markdown, flags=re.S):  # 去掉代码块
        for raw in _SENT_SPLIT.split(block):
            stripped = raw.strip()
            if stripped.startswith("#") or stripped.startswith("|"):
                continue
            sentence = raw.strip().lstrip("-*># ").strip()
            content = EVIDENCE_RE.sub("", sentence)
            content = re.sub(r"^[\d\s.)、:：-]+", "", content).strip()
            normalized = re.sub(r"\s+", " ", content).strip(" :：").lower()
            if sentence and normalized not in STRUCTURAL_TITLES and len(_terms(content)) >= 2:
                parts.append(sentence)
    return parts


def _terms(text: str) -> set[str]:
    """拉丁词 + 中文二元字，供确定性重叠使用。"""
    lowered = text.lower()
    latin = set(re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", lowered))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", lowered))
    bigrams = {cjk[i : i + 2] for i in range(len(cjk) - 1)}
    return latin | bigrams


def _overlap_ratio(claim_text: str, evidence_text: str) -> float:
    claim_terms = _terms(claim_text)
    if not claim_terms:
        return 0.0
    evidence_terms = _terms(evidence_text)
    return len(claim_terms & evidence_terms) / len(claim_terms)


def extract_claims(markdown: str, evidence: list[dict], paper_by_id: dict | None = None) -> list[dict]:
    """抽取事实与推断；未引用的事实也必须进入审批校验。"""
    paper_by_id = paper_by_id or {}
    evidence_index = {item["evidence_id"]: item for item in evidence}
    claims: list[dict] = []

    insufficient_section = "证据不足" in markdown and not evidence
    for sentence in _sentences(markdown):
        cited = EVIDENCE_RE.findall(sentence)
        valid_refs = [ref for ref in cited if ref in evidence_index]
        lowered = sentence.lower()
        if any(marker in lowered for marker in (m.lower() for m in INSUFFICIENT_MARKERS)):
            claim_type = "insufficient_evidence"
        elif any(marker in lowered for marker in (m.lower() for m in INFERENCE_MARKERS)):
            claim_type = "engineering_inference"
        elif valid_refs:
            source_types = {
                (paper_by_id.get(evidence_index[ref].get("paper_id")) or {}).get("source_type", "")
                for ref in valid_refs
            }
            if source_types <= {"public_arxiv", ""} and source_types:
                claim_type = "paper_fact"
            elif "internal_upload" in source_types and "public_arxiv" not in source_types:
                claim_type = "internal_fact"
            elif "public_arxiv" in source_types:
                claim_type = "paper_fact"
            else:
                claim_type = "internal_fact"
        else:
            # 无引用时无法可靠判断公开论文事实或内部事实；作为关键事实阻断审批。
            claim_type = "internal_fact"
        claims.append(
            {
                "claim_id": f"C-{len(claims) + 1:03d}",
                "claim_text": sentence[:500],
                "claim_type": claim_type,
                "evidence_ids": sorted(set(cited)),
                "support_status": "insufficient",
                "confidence": 0.0,
                "risk_note": "",
            }
        )

    if insufficient_section and not claims:
        claims.append(
            {
                "claim_id": "C-000",
                "claim_text": "证据不足，无法形成推荐。",
                "claim_type": "insufficient_evidence",
                "evidence_ids": [],
                "support_status": "insufficient",
                "confidence": 0.0,
                "risk_note": "没有可用证据，系统拒绝给出推荐。",
            }
        )
    return claims


def hard_validate(claims: list[dict], evidence: list[dict], store=None) -> list[dict]:
    """返回硬错误列表；空列表表示通过。校验引用存在性与可定位性。"""
    evidence_index = {item["evidence_id"]: item for item in evidence}
    chunk_cache: dict[str, bool] = {}
    errors: list[dict] = []

    for claim in claims:
        if claim["claim_type"] in KEY_FACT_TYPES and not claim["evidence_ids"]:
            errors.append(
                {"claim_id": claim["claim_id"], "evidence_id": None, "code": "missing_citation",
                 "message": "关键事实没有引用证据。"}
            )
        for ref in claim["evidence_ids"]:
            item = evidence_index.get(ref)
            if item is None:
                errors.append(
                    {"claim_id": claim["claim_id"], "evidence_id": ref, "code": "fabricated_citation",
                     "message": f"引用 {ref} 在证据库中不存在，疑似虚构引用。"}
                )
                continue
            if not item.get("paper_id") or not item.get("chunk_id"):
                errors.append(
                    {"claim_id": claim["claim_id"], "evidence_id": ref, "code": "unlocatable_evidence",
                     "message": f"引用 {ref} 无法定位到 paper/chunk。"}
                )
                continue
            if int(item.get("page_number") or 0) < 1:
                errors.append(
                    {"claim_id": claim["claim_id"], "evidence_id": ref, "code": "missing_page",
                     "message": f"引用 {ref} 缺少页码。"}
                )
            if store is not None:
                key = item["chunk_id"]
                if key not in chunk_cache:
                    paper = store.get_paper(item["paper_id"])
                    chunk_ids = {c["chunk_id"] for c in store.list_chunks(item["paper_id"])} if paper else set()
                    chunk_cache[key] = key in chunk_ids
                if not chunk_cache[key]:
                    errors.append(
                        {"claim_id": claim["claim_id"], "evidence_id": ref, "code": "chunk_missing",
                         "message": f"引用 {ref} 指向的分块在资料库中不存在。"}
                    )
    return errors


def _heuristic_support(claim: dict, evidence_index: dict[str, dict]) -> tuple[str, float, str]:
    refs = [ref for ref in claim["evidence_ids"] if ref in evidence_index]
    if claim["claim_type"] == "insufficient_evidence" or not refs:
        return "insufficient", 0.0, "没有可核验的证据支持该结论。"
    best = 0.0
    conflicted = False
    for ref in refs:
        item = evidence_index[ref]
        ratio = _overlap_ratio(claim["claim_text"], item.get("text", ""))
        best = max(best, ratio)
        lowered = item.get("text", "").lower()
        if any(marker in lowered for marker in NEGATION_MARKERS) and ratio > 0.2:
            conflicted = True
    if conflicted:
        return "conflicted", round(best, 3), "证据中出现与结论相反的表述。"
    if best >= 0.45:
        return "supported", round(min(1.0, best + 0.3), 3), ""
    if best >= 0.2:
        return "partially_supported", round(best, 3), "证据与结论仅部分相关，需人工复核。"
    return "insufficient", round(best, 3), "引用存在但内容不能支持该结论。"


def _llm_support(claims: list[dict], evidence_index: dict[str, dict], model) -> dict[str, dict] | None:
    """让 LLM 对 claim-证据做语义判定；解析失败返回 None 回退启发式。LLM 不能提升硬校验。"""
    compact_evidence = {
        ref: f"{item.get('title', '')} p{item.get('page_number')}: {item.get('text', '')[:600]}"
        for ref, item in evidence_index.items()
    }
    payload = [
        {"claim_id": c["claim_id"], "claim_text": c["claim_text"], "evidence_ids": c["evidence_ids"]}
        for c in claims
    ]
    system = SystemMessage(content=(
        "你是严谨的证据核验员。只根据给出的证据判断每条 claim 的支持程度，"
        "不得使用外部知识。输出 JSON 数组，每个元素形如 "
        '{"claim_id":"C-001","support_status":"supported|partially_supported|conflicted|insufficient",'
        '"confidence":0到1的数字,"risk_note":"简短中文说明"}。'
        "只要证据没有明确支持，就选 partially_supported 或 insufficient。"
    ))
    human = HumanMessage(content=f"证据：{json.dumps(compact_evidence, ensure_ascii=False)}\n\n"
                                 f"待核验结论：{json.dumps(payload, ensure_ascii=False)}")
    try:
        response = model.invoke([system, human])
        content = response.content if isinstance(response.content, str) else str(response.content)
        match = re.search(r"\[.*\]", content, re.S)
        data = json.loads(match.group(0) if match else content)
        result = {}
        for row in data:
            status = row.get("support_status")
            if row.get("claim_id") and status in SUPPORT_STATUSES:
                result[row["claim_id"]] = {
                    "support_status": status,
                    "confidence": max(0.0, min(1.0, float(row.get("confidence", 0.0)))),
                    "risk_note": str(row.get("risk_note", ""))[:300],
                }
        return result or None
    except Exception:
        return None


def evaluate_claims(markdown: str, evidence: list[dict], model=None, store=None,
                    use_llm: bool = True) -> dict:
    evidence_index = {item["evidence_id"]: item for item in evidence}
    cited_ids = set(EVIDENCE_RE.findall(markdown))
    missing = sorted(cited_ids - set(evidence_index))

    paper_by_id = {}
    if store is not None:
        for item in evidence:
            paper_by_id.setdefault(item.get("paper_id"), store.get_paper(item.get("paper_id")))

    claims = extract_claims(markdown, evidence, paper_by_id)
    hard_errors = hard_validate(claims, evidence, store=store)
    if "```" in markdown:
        hard_errors.append(
            {"claim_id": None, "evidence_id": None, "code": "invalid_report_format",
             "message": "报告包含代码围栏；疑似把证据中的格式指令当成了输出要求。"}
        )
    fabricated = missing + [e["evidence_id"] for e in hard_errors if e["code"] == "fabricated_citation"]

    # 软校验：LLM 仅辅助，失败/不确定回退保守启发式。
    llm_results = None
    if use_llm and model is not None and claims:
        llm_results = _llm_support(claims, evidence_index, model)
    for claim in claims:
        status, confidence, note = _heuristic_support(claim, evidence_index)
        if llm_results and claim["claim_id"] in llm_results:
            llm = llm_results[claim["claim_id"]]
            # LLM 不能把硬错误 claim 判成 supported；不能比启发式更激进地“翻案”为 supported
            if llm["support_status"] != "supported" or status == "supported":
                status, confidence, note = llm["support_status"], llm["confidence"], llm["risk_note"]
        claim["support_status"] = status
        claim["confidence"] = confidence
        if note and not claim["risk_note"]:
            claim["risk_note"] = note

    counts = {name: 0 for name in SUPPORT_STATUSES}
    for claim in claims:
        counts[claim["support_status"]] += 1

    key_claims = [c for c in claims if c["claim_type"] in KEY_FACT_TYPES]
    supported_key = [
        c for c in key_claims
        if c["support_status"] == "supported"
        and all(ref in evidence_index for ref in c["evidence_ids"])
    ]
    key_coverage = (len(supported_key) / len(key_claims)) if key_claims else 0.0

    inferences = [c for c in claims if c["claim_type"] == "engineering_inference"]
    unmarked_inferences = [
        c for c in inferences
        if not any(marker.lower() in c["claim_text"].lower() for marker in INFERENCE_MARKERS)
    ]
    insufficient_only = bool(evidence) is False or (
        all(c["claim_type"] == "insufficient_evidence" for c in claims) if claims else True
    )

    citation_rate = 1.0 if not cited_ids else len(cited_ids & set(evidence_index)) / len(cited_ids)
    approval_allowed = (
        not missing
        and not hard_errors
        and citation_rate == 1.0
        and bool(cited_ids)
        and key_coverage == 1.0
        and not insufficient_only
        and all(c["support_status"] != "conflicted" for c in claims)
    )

    return {
        "valid": approval_allowed,
        "approval_allowed": approval_allowed,
        "available_evidence": len(evidence_index),
        "cited_evidence": len(cited_ids & set(evidence_index)),
        "missing_evidence": missing,
        "citation_existence_rate": citation_rate,
        "hard_errors": hard_errors,
        "fabricated_citations": sorted(set(fabricated)),
        "claim_count": len(claims),
        "key_claim_count": len(key_claims),
        "claim_evidence_coverage": round(key_coverage, 3),
        "support_counts": counts,
        "engineering_inference_count": len(inferences),
        "engineering_inferences_marked": not unmarked_inferences,
        "unmarked_engineering_inference_count": len(unmarked_inferences),
        "insufficient_only": insufficient_only,
        "soft_checker": "llm+heuristic" if llm_results else "heuristic",
        "claims": claims,
    }
