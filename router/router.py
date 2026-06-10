"""
智能路由模块（步骤 8）— 自动判断问题类型并路由到 RAG / GraphRAG

路由策略（三级 fallback）：
  1) 关键词路由（最快，0 LLM 调用，靠 GraphRAG 论文别名识别）
  2) 语义路由（基于嵌入相似度，0 LLM 调用，延迟 <0.1s）
  3) LLM 路由（最智能，~1 次 LLM 调用，可能解析失败）

smart_route() 主入口：先 1) → 2) → 3)，结果合并到 dict 返回。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import (
    EMBED_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    ROUTER_MARGIN_THRESHOLD,
)

sys.stdout.reconfigure(encoding="utf-8")

# ============== 1. 路由目的地定义 ==============
DESTINATIONS = {
    "vector_rag": {
        "name": "RAG",
        "description": "论文内容、概念、方法、模型机制的语义查询；返回原文片段 + 引用。",
        "examples": [
            "RAPTOR 论文的核心思想是什么？",
            "CRAG 的检索评估器触发哪三种动作？",
            "FLARE 论文中模型如何决定何时检索？",
            "什么是 GraphRAG？",
        ],
    },
    "graph_rag": {
        "name": "GraphRAG",
        "description": "论文之间的引用关系查询；返回 Neo4j 图遍历结果。",
        "examples": [
            "STAR 论文引用了哪些论文？",
            "HiQA 论文是否引用了 RAPTOR？",
            "从 Lewis 2020 到 STAR 的引用路径？",
            "哪些论文引用了 RAG 奠基论文？",
        ],
    },
}

# ============== 2. 关键词路由（最优先，最快） ==============
# 命中以下规则 → 直接路由，不调 LLM
# 规则：先看"引用"系关键词 + 是否同时检测到 ≥1 篇论文
GRAPH_KEYWORDS = [
    r"引用了哪些",     # 出边
    r"引用了.*?论文",   # 出边 / 是否引用
    r"是否引用",
    r"被谁引用",
    r"哪些论文引用",
    r"被引用(?!量|数|次|率)",  # 排除"被引用量/数/次数"
    r"引用路径",
    r"引用.*?路径",
    r"经过.*?论文",     # 路径
    r"\bpath\b",        # 路径
]


def keyword_route(question: str) -> str | None:
    """
    关键词路由：识别明显的图查询意图
    - 命中"引用路径/是否引用/引用了哪些"等关键词 → graph_rag
    - 未命中以上模式 → 返回 None，交由下一级路由判断
    """
    q = question.strip()
    # 路径 / 列表 / 是否引用 → 一定是图查询
    for pat in GRAPH_KEYWORDS:
        if re.search(pat, q, re.IGNORECASE):
            return "graph_rag"
    return None


# ============== 3. 语义路由（嵌入相似度） ==============
# 预计算每个目的地的"平均例句向量"；运行时算 query 与每个均值的余弦相似度
embeddings_router = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def _build_destination_vectors():
    """预计算每个路由目的地的平均例句向量"""
    print("[router] 预计算目的地向量（首次加载约 10s）...")
    dest_vecs = {}
    for dest, info in DESTINATIONS.items():
        vecs = [embeddings_router.embed_query(ex) for ex in info["examples"]]
        dest_vecs[dest] = np.mean(vecs, axis=0)
    print(f"[router] 已加载 {len(dest_vecs)} 个目的地的语义向量")
    return dest_vecs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# 模块级单例（首次 import 时计算一次）
_DEST_VECTORS: dict[str, np.ndarray] = {}


def _get_dest_vectors() -> dict[str, np.ndarray]:
    global _DEST_VECTORS
    if not _DEST_VECTORS:
        _DEST_VECTORS = _build_destination_vectors()
    return _DEST_VECTORS


def semantic_route(question: str) -> tuple[str, dict[str, float]]:
    """
    语义路由：query 嵌入 vs 每个目的地的例句平均嵌入，取相似度最高的
    返回 (destination, scores_dict)
    """
    dest_vecs = _get_dest_vectors()
    q_vec = np.array(embeddings_router.embed_query(question))
    scores = {dest: _cosine(q_vec, vec) for dest, vec in dest_vecs.items()}
    best = max(scores, key=scores.get)
    return best, scores


# ============== 4. LLM 路由（兜底，最智能但最慢） ==============
llm_router = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
)

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
你是智能路由助手。根据用户问题，选择最合适的知识库。

可选目的地：
{destinations}

用户问题：{input}

只输出 JSON（不要 ``` 包裹，不要任何解释）：
{{"destination": "vector_rag 或 graph_rag", "reason": "一句话说明判断依据"}}
""")

_router_llm_chain = ROUTER_PROMPT | llm_router | StrOutputParser()


def _build_destinations_text() -> str:
    """把 DESTINATIONS 字典格式化成 prompt 里可读的描述"""
    lines = []
    for dest, info in DESTINATIONS.items():
        lines.append(f'- "{dest}"（{info["name"]}）：{info["description"]}')
        for ex in info["examples"][:2]:
            lines.append(f'    例：{ex}')
    return "\n".join(lines)


def parse_router_output(text: str) -> dict:
    """从 LLM 输出中抠 JSON；容错：去掉 ```json 包裹、找第一个 {...}"""
    text = text.strip()
    # 去掉 markdown 包裹
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "").strip()
    # 抠 JSON 对象
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 最后兜底：猜一个
    if "graph" in text.lower():
        return {"destination": "graph_rag", "reason": "LLM 输出异常，按关键词兜底"}
    return {"destination": "vector_rag", "reason": "LLM 输出异常，默认 RAG"}


def llm_route(question: str) -> dict:
    """LLM 路由：返回 {destination, reason, raw, input}"""
    try:
        raw = _router_llm_chain.invoke({
            "destinations": _build_destinations_text(),
            "input": question,
        })
        decision = parse_router_output(raw)
        decision["raw"] = raw
        decision["input"] = question  # 保留原始问题
        return decision
    except Exception as e:
        return {
            "destination": "vector_rag",
            "reason": f"LLM 路由失败: {e}",
            "raw": "",
            "input": question,
        }


# ============== 5. 路由执行 ==============
def route_decision(decision: dict) -> dict:
    """根据 decision 调对应的链；返回标准结果 dict"""
    dest = decision.get("destination", "vector_rag")
    if dest not in ("vector_rag", "graph_rag"):
        dest = "vector_rag"  # 兜底

    # 兜底：保证下游拿到原始问题
    user_input = decision.get("input") or decision.get("question") or ""

    if dest == "graph_rag":
        from graphrag.pipeline import graphrag_query
        result = graphrag_query(user_input)
    else:
        from rag.pipeline import rag_query
        result = rag_query(user_input)

    # 合并路由信息
    result["router_decision"] = {
        "destination": dest,
        "destination_name": DESTINATIONS[dest]["name"],
        "reason": decision.get("reason", ""),
    }
    result["route"] = DESTINATIONS[dest]["name"]
    return result


# ============== 6. 三级 fallback 主入口 ==============
def _keyword_match(question: str) -> str | None:
    """关键词路由 + 实体校验；命中则返回 destination，否则返回 None。

    - 命中"引用路径/是否引用/引用了哪些"等关键词 → 初步判为 graph_rag
    - 若 graph_rag 候选问题中未识别到任何论文实体 → 视为误匹配，降级走下一级
    """
    kw_dest = keyword_route(question)
    if not kw_dest:
        return None
    if kw_dest == "graph_rag":
        from graphrag.pipeline import detect_arxiv_ids
        if not detect_arxiv_ids(question):
            return None  # 误匹配 → 交给下一级路由
    return kw_dest


def _build_keyword_decision(question: str, kw_dest: str) -> dict:
    """把命中信息打包成下游 route_decision() 需要的决策 dict"""
    matched = [p for p in GRAPH_KEYWORDS if re.search(p, question, re.IGNORECASE)]
    return {
        "destination": kw_dest,
        "input": question,
        "reason": f"关键词命中: {matched[0] if matched else '未知模式'}",
    }


def smart_route(question: str) -> dict:
    """
    智能路由主入口：关键词 → 语义 → LLM
    返回 dict，包含 answer / sources / route / router_decision / router_method 等
    """
    method_chain: list[str] = []  # 记录实际命中的路由方法链

    # === 1) 关键词路由（命中即停；误匹配则降级到下一级）===
    kw_dest = _keyword_match(question)
    if kw_dest:
        result = route_decision(_build_keyword_decision(question, kw_dest))
        result["router_method"] = "keyword"
        return result
    method_chain.append("keyword-miss")

    # === 2) 语义路由 ===
    try:
        sem_dest, sem_scores = semantic_route(question)
        # 置信度阈值：差值 < 阈值时判为不确定，让 LLM 兜底
        scores_sorted = sorted(sem_scores.values(), reverse=True)
        margin = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else 1.0
        if margin >= ROUTER_MARGIN_THRESHOLD:
            decision = {
                "destination": sem_dest,
                "input": question,
                "reason": f"语义相似度（margin={margin:.3f}）",
                "scores": sem_scores,
            }
            result = route_decision(decision)
            result["router_method"] = "semantic"
            result["router_scores"] = sem_scores
            return result
        method_chain.append(f"semantic-uncertain(margin={margin:.3f})")
    except Exception as e:
        method_chain.append(f"semantic-failed({e})")

    # === 3) LLM 路由（兜底） ===
    decision = llm_route(question)
    result = route_decision(decision)
    result["router_method"] = "llm" if not method_chain else "+".join(method_chain) + "+llm"
    return result


# ============== 7. 自检入口 ==============
if __name__ == "__main__":
    test_questions = [
        # 期望 → RAG
        ("RAPTOR 论文的核心思想是什么？", "vector_rag"),
        ("CRAG 论文中，检索评估器会触发哪三种动作？", "vector_rag"),
        ("FLARE 论文中，模型如何决定何时进行检索？", "vector_rag"),
        # 期望 → GraphRAG
        ("STAR 论文引用了哪些论文？（列出标题）", "graph_rag"),
        ("HiQA 论文是否引用了 RAPTOR？", "graph_rag"),
        ("从 Lewis et al. 2020 到 STAR 论文的引用路径是什么？", "graph_rag"),
    ]

    out_log = Path(__file__).parent.parent / "data" / "router_test_output.txt"
    log_lines = []
    print("\n" + "=" * 70)
    print("智能路由模块自检（6 道测试题）")
    print("=" * 70)

    correct = 0
    for q, expected in test_questions:
        block = []
        block.append("\n" + "=" * 70)
        block.append(f"[Q] {q}")
        block.append(f"[expected] {expected}")
        block.append("=" * 70)
        result = smart_route(q)
        actual = result.get("route", "?")
        method = result.get("router_method", "?")
        reason = result.get("router_decision", {}).get("reason", "")
        expected_label = "RAG" if expected == "vector_rag" else "GraphRAG"
        match = "✓" if actual == expected_label else "✗"
        if match == "✓":
            correct += 1
        block.append(f"[route] {actual}  [method] {method}  [{match}]")
        block.append(f"[reason] {reason}")
        if "router_scores" in result:
            block.append(f"[scores] {result['router_scores']}")
        block.append(f"\n[A]\n{result['answer'][:600]}{'...' if len(result['answer']) > 600 else ''}")
        log_lines.append("\n".join(block))
        for ln in block:
            print(ln)

    summary = f"\n\n{'=' * 70}\n[summary] {correct}/{len(test_questions)} routes correct\n{'=' * 70}"
    log_lines.append(summary)
    print(summary)
    out_log.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[router] test log -> {out_log}")
