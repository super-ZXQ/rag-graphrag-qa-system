"""无答案问题的可答性探针。

检索分数（cosine/RRF/词项重叠）在跨语言语义检索下不能可靠区分“无答案”，
因此拒答在产品中由证据/claim 语义层完成。这里用与报告相同的生成模型做一次
受限可答性判断：只依据 top-k 片段决定 answer/refuse；模型不可用时返回不可用状态，
绝不臆造准确率。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage


def _decide(model, query: str, snippets: list[dict]) -> dict:
    context = "\n\n".join(
        f"[{i + 1}] {s.get('title', '')} p{s.get('page_number', '?')}: {s.get('text', '')[:700]}"
        for i, s in enumerate(snippets)
    )
    system = SystemMessage(content=(
        "你是企业知识库的可答性判断器。只能依据给出的检索片段判断问题是否可回答，"
        "禁止使用外部知识。若片段没有直接包含答案，必须 refuse。"
        "只输出 JSON：{\"decision\": \"answer\" 或 \"refuse\", \"reason\": \"简短中文说明\"}。"
    ))
    human = HumanMessage(content=f"检索片段：\n{context}\n\n问题：{query}")
    response = model.invoke([system, human])
    content = response.content if isinstance(response.content, str) else str(response.content)
    match = re.search(r"\{.*\}", content, re.S)
    data = json.loads(match.group(0) if match else content)
    decision = data.get("decision")
    if decision not in ("answer", "refuse"):
        raise ValueError("无法解析可答性判断结果")
    return {"decision": decision, "reason": str(data.get("reason", ""))[:200]}


def run_answerability_probe(cases: list[dict], search, model, answer_sample_step: int = 4) -> dict:
    refuse_cases = [c for c in cases if c["expected_behavior"] == "refuse"]
    # 确定性抽样部分应答类问题，用于度量“过度拒答”。
    answer_cases = [c for i, c in enumerate(cases) if c["expected_behavior"] == "answer"
                    and i % answer_sample_step == 0]
    rows, refuse_ok, answer_ok = [], 0, 0
    for case in refuse_cases + answer_cases:
        snippets = search(case["query"])
        try:
            verdict = _decide(model, case["query"], snippets[:5])
            error = None
        except Exception as exc:  # noqa: BLE001
            verdict, error = {"decision": "error", "reason": ""}, type(exc).__name__
        expected = case["expected_behavior"]
        correct = verdict["decision"] == expected
        if expected == "refuse":
            refuse_ok += int(correct)
        else:
            answer_ok += int(correct)
        rows.append({
            "id": case["id"], "query": case["query"], "expected": expected,
            "predicted": verdict["decision"], "correct": correct,
            "reason": verdict.get("reason", ""), "error": error,
        })
    return {
        "checker": "llm_answerability",
        "refuse_case_count": len(refuse_cases),
        "sampled_answer_case_count": len(answer_cases),
        "noanswer_accuracy": round(refuse_ok / len(refuse_cases), 4) if refuse_cases else None,
        "sampled_answer_accuracy": round(answer_ok / len(answer_cases), 4) if answer_cases else None,
        "cases": rows,
    }
