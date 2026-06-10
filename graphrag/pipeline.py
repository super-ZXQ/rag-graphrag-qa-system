"""
GraphRAG 问答模块（步骤 7）— 基于 Neo4j 引用图
流程：
  用户问题 → 关键词/别名识别 → 选择预定义 Cypher 模板
         → 命中则直接执行；不命中则调用 LLM 生成 Cypher
         → 拿到结果 → LLM 整理成自然语言答案
         → 返回 (答案 + 原始 Cypher + 查询结果 + 论文标题)

支持的查询类型：
  1) 出边查询："X 引用了哪些论文？"  → MATCH (X)-[:CITES]->(b)
  2) 入边查询："哪些论文引用了 X？"  → MATCH (a)-[:CITES]->(X)
  3) 路径查询 ："X 到 Y 的引用路径？"  → shortestPath(...)
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TEMPERATURE,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OLLAMA_BASE_URL,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase

from graph.citations import PAPERS

sys.stdout.reconfigure(encoding="utf-8")

# ============== 1. Neo4j 连接 ==============
# 用原生 driver（不依赖 APOC / langchain-neo4j），更轻量
_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _run_cypher(cypher: str) -> list[dict]:
    """执行 Cypher，返回 list[dict]；出错抛异常"""
    with _driver.session(database=NEO4J_DATABASE) as s:
        return [dict(r) for r in s.run(cypher)]


# ============== 2. 论文别名 → arxiv_id ==============
# 用 arxiv_id 做精确匹配（"2605.18765"），也支持短名（"STAR"）
# 优先级：先匹配长关键词（避免 "Lewis" 误命中 "Lewis 2020"）
PAPER_ALIASES: dict[str, str] = {
    # arxiv_id
    "2005.11401": "2005.11401",
    "2305.06983": "2305.06983",
    "2309.15217": "2309.15217",
    "2401.15884": "2401.15884",
    "2401.18059": "2401.18059",
    "2402.01767": "2402.01767",
    "2404.16130": "2404.16130",
    "2409.14924": "2409.14924",
    "2410.12837": "2410.12837",
    "2503.10677": "2503.10677",
    "2601.08773": "2601.08773",
    "2605.18765": "2605.18765",
    # 简称
    "lewis 2020": "2005.11401",
    "lewis et al. 2020": "2005.11401",
    "lewis et al": "2005.11401",
    "rag 奠基": "2005.11401",
    "rag奠基": "2005.11401",
    "rag 奠基论文": "2005.11401",
    "rag奠基论文": "2005.11401",
    "flatre": "2305.06983",  # 偶尔拼错
    "flare": "2305.06983",
    "jiang 2023": "2305.06983",
    "jiang et al. 2023": "2305.06983",
    "ragas": "2309.15217",
    "es 2023": "2309.15217",
    "es et al. 2023": "2309.15217",
    "crag": "2401.15884",
    "yan 2024": "2401.15884",
    "yan et al. 2024": "2401.15884",
    "raptor": "2401.18059",
    "sarthi 2024": "2401.18059",
    "sarthi et al. 2024": "2401.18059",
    "hiqa": "2402.01767",
    "chen 2024": "2402.01767",
    "chen et al. 2024": "2402.01767",
    "graphrag": "2404.16130",
    "microsoft graphrag": "2404.16130",
    "edge 2024": "2404.16130",
    "edge et al. 2024": "2404.16130",
    "rag and beyond": "2409.14924",
    "zhao 2024": "2409.14924",
    "zhao et al. 2024": "2409.14924",
    "rag survey": "2410.12837",
    "gupta 2024": "2410.12837",
    "gupta et al. 2024": "2410.12837",
    "知识导向": "2503.10677",
    "knowledge-oriented": "2503.10677",
    "cheng 2025": "2503.10677",
    "cheng et al. 2025": "2503.10677",
    "code graphrag": "2601.08773",
    "chinthareddy 2026": "2601.08773",
    "chinthareddy": "2601.08773",
    "star": "2605.18765",
    "li 2026": "2605.18765",
    "li et al. 2026": "2605.18765",
}

# 按"长度倒序"匹配（避免短关键词先吃掉长关键词的位置）
_SORTED_ALIASES = sorted(PAPER_ALIASES.items(), key=lambda x: -len(x[0]))

# 中英文括号兼容
ALIAS_PATTERN = re.compile(
    r"(" + "|".join(re.escape(a) for a, _ in _SORTED_ALIASES) + r")",
    re.IGNORECASE,
)


def detect_arxiv_ids(question: str) -> list[str]:
    """从问题中识别提到的论文，返回 arxiv_id 列表（去重，按出现顺序）"""
    seen = set()
    out = []
    for m in ALIAS_PATTERN.finditer(question):
        aid = PAPER_ALIASES[m.group(1)]  # regex 已 IGNORECASE，key 保持小写
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


# ============== 3. 预定义 Cypher 模板 ==============
# 模板覆盖考试 Q4/Q5/Q6；其他问句走 LLM 生成

def cypher_outgoing(arxiv_id: str) -> str:
    """出边：X 引用了哪些论文"""
    return (
        f"MATCH (a:Paper {{arxiv_id:'{arxiv_id}'}})-[:CITES]->(b:Paper) "
        f"RETURN b.arxiv_id AS arxiv_id, b.title AS title, "
        f"b.short_name AS short_name, b.year AS year "
        f"ORDER BY b.year, b.arxiv_id"
    )


def cypher_incoming(arxiv_id: str) -> str:
    """入边：哪些论文引用了 X"""
    return (
        f"MATCH (a:Paper)-[:CITES]->(b:Paper {{arxiv_id:'{arxiv_id}'}}) "
        f"RETURN a.arxiv_id AS arxiv_id, a.title AS title, "
        f"a.short_name AS short_name, a.year AS year "
        f"ORDER BY a.year, a.arxiv_id"
    )


def cypher_path(src: str, dst: str, directed: bool = False) -> str:
    """路径查询：X → Y"""
    if directed:
        rel = "-[:CITES*1..6]->"
    else:
        rel = "-[:CITES*1..6]-"
    return (
        f"MATCH p = shortestPath((a:Paper {{arxiv_id:'{src}'}}){rel}(b:Paper {{arxiv_id:'{dst}'}})) "
        f"RETURN [n IN nodes(p) | n.short_name] AS path, "
        f"length(p) AS hops"
    )


# ============== 4. 模板选择 + 关键词路由 ==============
def template_query(question: str) -> tuple[str, list[dict], str] | None:
    """
    关键词路由 → 模板查询
    返回 (cypher, results, intent) 或 None（未命中）
    """
    q = question
    ids = detect_arxiv_ids(q)

    # --- 路径查询：必须同时提到 2 篇 ---
    if ("路径" in q or "path" in q.lower() or "经过" in q) and len(ids) >= 2:
        cypher = cypher_path(ids[0], ids[1], directed=False)
        rows = _run_cypher(cypher)
        return cypher, rows, "path"

    if "是否引用" in q and len(ids) >= 2:
        cypher = cypher_outgoing(ids[0])  # 看 X 出边里有无 Y
        rows = _run_cypher(cypher)
        cited_ids = {r["arxiv_id"] for r in rows}
        yes = ids[1] in cited_ids
        return cypher, [{"cited": yes, "from": ids[0], "to": ids[1]}], "boolean"

    # --- 出边：X 引用了哪些（"引用" 一词覆盖所有引用相关查询）---
    if ids and "引用" in q:
        cypher = cypher_outgoing(ids[0])
        rows = _run_cypher(cypher)
        return cypher, rows, "outgoing"

    # --- 入边：哪些引用了 X ---
    if ids and ("被谁引用" in q or "哪些论文引用" in q or "被引用" in q):
        cypher = cypher_incoming(ids[0])
        rows = _run_cypher(cypher)
        return cypher, rows, "incoming"

    # 关键词模板均未命中 → 交给 LLM 生成 Cypher
    return None


# ============== 5. LLM 生成 Cypher（兜底） ==============
# 当模板没命中时，调 LLM 把自然语言转 Cypher
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
)

CYPHER_GEN_PROMPT = ChatPromptTemplate.from_template("""
你是 Cypher 专家。根据 Neo4j 图 schema 和用户问题，生成正确的 Cypher 查询。
只输出 Cypher 一行语句（可以多行但不要带任何解释、不要 ``` 包裹）。

Schema:
- 节点 (:Paper) 属性: arxiv_id (String, 主键), short_id, title, first_author, year, short_name
- 关系 -[:CITES]->  (Paper -> Paper)，表示源论文引用目标论文
- 数据共 12 篇论文；CITES 方向为"新论文 → 旧论文"（引用方 → 被引用方）

可用 arxiv_id 列表：
{arxiv_ids}

示例：
问题：STAR 论文引用了哪些论文？
Cypher：MATCH (a:Paper {{arxiv_id:'2605.18765'}})-[:CITES]->(b:Paper) RETURN b.title, b.short_name, b.year ORDER BY b.year

问题：HiQA 论文是否引用了 RAPTOR？
Cypher：MATCH (a:Paper {{arxiv_id:'2402.01767'}})-[:CITES]->(b:Paper {{arxiv_id:'2401.18059'}}) RETURN b.short_name

问题：从 Lewis 2020 到 STAR 的引用路径是什么？
Cypher：MATCH p = shortestPath((a:Paper {{arxiv_id:'2005.11401'}})-[:CITES*1..6]-(b:Paper {{arxiv_id:'2605.18765'}})) RETURN [n IN nodes(p) | n.short_name] AS path

现在请回答：
问题：{question}
Cypher：""")

CYPHER_CLEAN_RE = re.compile(r"```[a-zA-Z]*\n?|```")


def clean_cypher(text: str) -> str:
    """清洗 LLM 输出：去掉 ```cypher 包裹、首尾空白"""
    text = CYPHER_CLEAN_RE.sub("", text).strip()
    return text


def llm_generate_cypher(question: str) -> str:
    """LLM 生成 Cypher；不直接执行（交给调用方跑 + 兜底）"""
    chain = CYPHER_GEN_PROMPT | llm | StrOutputParser()
    arxiv_list = ", ".join(f"'{a}'" for a in sorted(set(PAPER_ALIASES.values())) if a in PAPERS)
    raw = chain.invoke({"question": question, "arxiv_ids": arxiv_list})
    return clean_cypher(raw)


# ============== 6. 答案生成 ==============
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
你是学术论文助手。根据图查询结果，用中文回答用户问题。

用户问题：{question}
查询意图：{intent}
Cypher 查询：{cypher}
查询结果：{result}

要求：
1) 答案用中文，简洁准确
2) 如果结果为空，明确说"图谱中未找到相关信息"
3) 列出论文标题；如有多篇，用编号列出
""")

answer_chain = ANSWER_PROMPT | llm | StrOutputParser()


def _llm_invoke_with_retry(chain, inputs: dict, max_retries: int = LLM_MAX_RETRIES) -> str:
    """带重试的 LLM 调用，防止 Ollama 偶尔断连"""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [LLM retry {attempt+1}/{max_retries}] {e}, waiting {wait}s...")
                time.sleep(wait)
            else:
                # 最后一次仍失败，返回兜底答案
                return f"LLM 生成失败（{e}），原始查询结果：{inputs.get('result', '（无）')}"


def _format_result(intent: str, rows: list[dict]) -> str:
    """把 neo4j 行转成可读文本塞进 LLM prompt"""
    if not rows:
        return "（无结果）"
    if intent == "boolean":
        return str(rows[0])
    if intent == "path":
        return str(rows[0].get("path", rows[0]))
    # outgoing / incoming：每行一篇论文
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i}. {r.get('short_name', '?')} - {r.get('title', '?')} "
            f"({r.get('year', '?')}) [{r.get('arxiv_id', '?')}]"
        )
    return "\n".join(lines)


# ============== 7. 对外主入口 ==============
def graphrag_query(question: str) -> dict:
    """
    GraphRAG 问答主入口
    流程：模板匹配 → 命中则直接执行；未命中则 LLM 生成 Cypher
    返回：{answer, cypher, results, intent, route}
    """
    # 1) 先试模板（快、可靠）
    tmpl = template_query(question)
    if tmpl is not None:
        cypher, rows, intent = tmpl
    else:
        # 2) 兜底：LLM 生成 Cypher
        cypher = ""
        try:
            cypher = llm_generate_cypher(question)
            rows = _run_cypher(cypher)
            intent = "llm_generated"
        except Exception as e:
            return {
                "answer": f"图查询生成失败：{e}",
                "cypher": cypher,
                "results": [],
                "intent": "failed",
                "route": "GraphRAG",
            }

    # 3) 用 LLM 整理成自然语言答案
    # boolean / path 类型答案已确定，不走 LLM 避免幻觉 + Ollama 不稳定
    if intent == "boolean" and rows:
        rec = rows[0]
        if rec.get("cited"):
            from_name = PAPERS.get(rec["from"], {}).get("short_name", rec["from"])
            to_name = PAPERS.get(rec["to"], {}).get("short_name", rec["to"])
            answer = f"是：{from_name} 引用了 {to_name}。"
        else:
            from_name = PAPERS.get(rec["from"], {}).get("short_name", rec["from"])
            to_name = PAPERS.get(rec["to"], {}).get("short_name", rec["to"])
            answer = f"否：{from_name} 没有直接引用 {to_name}（图谱中无对应 CITES 边）。"
    elif intent == "path" and rows:
        path_list = rows[0].get("path", [])
        hops = rows[0].get("hops", "?")
        if path_list:
            path_str = " → ".join(path_list)
            answer = f"引用路径（{hops} 跳）：{path_str}"
        else:
            answer = "图谱中未找到相关引用路径。"
    else:
        result_text = _format_result(intent, rows)
        answer = _llm_invoke_with_retry(answer_chain, {
            "question": question,
            "intent": intent,
            "cypher": cypher,
            "result": result_text,
        })

    # 4) 整理 paper_titles 列表（界面侧用）
    paper_titles = []
    if intent in ("outgoing", "incoming"):
        for r in rows:
            paper_titles.append({
                "arxiv_id": r.get("arxiv_id", ""),
                "title": r.get("title", ""),
                "short_name": r.get("short_name", ""),
                "year": r.get("year", 0),
            })
    elif intent == "path" and rows:
        paper_titles = [{"short_name": n} for n in rows[0].get("path", [])]
    elif intent == "boolean":
        paper_titles = [{"summary": str(rows[0])}]

    return {
        "answer": answer,
        "cypher": cypher,
        "results": rows,
        "intent": intent,
        "paper_titles": paper_titles,
        "route": "GraphRAG",
    }


# ============== 8. 自检入口 ==============
if __name__ == "__main__":
    test_questions = [
        "STAR 论文引用了哪些论文？（列出标题）",
        "HiQA 论文是否引用了 RAPTOR？",
        "从 Lewis et al. 2020 到 STAR 论文的引用路径是什么？（列出路径上的论文标题）",
    ]
    out_log = Path(__file__).parent.parent / "data" / "graphrag_test_output.txt"
    log_lines = []
    for q in test_questions:
        block = []
        block.append("\n" + "=" * 70)
        block.append(f"[Q] {q}")
        block.append("=" * 70)
        out = graphrag_query(q)
        block.append(f"\n[intent] {out['intent']}")
        block.append(f"\n[cypher]\n{out['cypher']}")
        block.append(f"\n[results] ({len(out['results'])} rows)")
        for r in out["results"][:8]:
            block.append(f"  {r}")
        block.append(f"\n[A]\n{out['answer']}")
        log_lines.append("\n".join(block))
        # 实时打印
        for ln in block:
            print(ln)
    out_log.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[graphrag] test log -> {out_log} (utf-8)")
