"""
Streamlit UI — 基于 RAG + GraphRAG 的学术论文问答系统
- 模式 1：RAG 问答（向量检索 + LLM）
- 模式 2：GraphRAG 问答（图查询 + LLM）
- 模式 3：自动路由（智能判断 → 上述两者之一）
- 6 道测试题快捷按钮 + 自由问答
- 完整显示：答案 / 引用 / 检索/查询细节

启动：cd d:\小组作业\rag_qa_system && $env:NEO4J_PASSWORD="neo4j12345"; streamlit run ui/app.py
访问：http://localhost:8501
"""
import sys
from pathlib import Path

# 让 ui/app.py 能 import 顶层 config / rag / graphrag / router
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ============== 页面配置 ==============
st.set_page_config(
    page_title="学术论文问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== 自定义 CSS ==============
# 设计原则：所有自定义卡片都用浅色背景 + 深色文字（!important），
# 这样无论 Streamlit 主题（light/dark）都能清晰阅读。
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555 !important;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 8px;
    }
    .badge-rag { background: #e3f2fd; color: #1565c0 !important; }
    .badge-graphrag { background: #fff3e0; color: #e65100 !important; }
    .badge-auto { background: #f3e5f5; color: #6a1b9a !important; }

    /* === 答案卡：浅绿底 + 深色文字 === */
    .answer-card {
        background: #e8f5e9 !important;
        border-left: 5px solid #2e7d32 !important;
        padding: 16px 20px !important;
        border-radius: 6px !important;
        margin: 12px 0 !important;
        font-size: 1.05rem !important;
        line-height: 1.8 !important;
        color: #1a1a1a !important;
    }
    .answer-card b { color: #1b5e20 !important; }

    /* === 引用来源卡：浅蓝底 + 深色文字 === */
    .source-card {
        background: #e3f2fd !important;
        border-left: 4px solid #1565c0 !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        border-radius: 6px !important;
        color: #1a1a1a !important;
    }
    .source-card b {
        color: #0d47a1 !important;
        font-size: 1.02rem !important;
    }
    .source-card i { color: #333 !important; }
    .source-card code {
        background: #fff !important;
        color: #0d47a1 !important;
        padding: 1px 6px !important;
        border-radius: 3px !important;
        font-size: 0.85rem !important;
    }
    .source-card small { color: #444 !important; }

    /* === 路由决策卡：浅紫底 + 深色文字 === */
    .route-card {
        background: linear-gradient(90deg, #f3e5f5 0%, #fce4ec 100%) !important;
        border: 1px solid #ce93d8 !important;
        padding: 14px 18px !important;
        border-radius: 8px !important;
        margin: 8px 0 !important;
        color: #1a1a1a !important;
    }
    .route-card b { color: #4a148c !important; }
    .route-card small { color: #555 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ============== 懒加载三个 pipeline（首次访问才初始化） ==============
@st.cache_resource(show_spinner="正在加载 RAG 流水线（Qdrant + Embedding）...")
def get_rag():
    from rag.pipeline import rag_query
    return rag_query


@st.cache_resource(show_spinner="正在加载 GraphRAG 流水线（Neo4j）...")
def get_graphrag():
    from graphrag.pipeline import graphrag_query
    return graphrag_query


@st.cache_resource(show_spinner="正在加载智能路由模块...")
def get_router():
    from router.router import smart_route
    return smart_route


# ============== 侧边栏 ==============
with st.sidebar:
    st.markdown("### 📖 系统说明")
    st.markdown(
        """
**基于 RAG + GraphRAG 的学术论文问答系统**

数据集：12 篇 RAG/GraphRAG 相关论文
- **RAG 模式**：基于 Qdrant 向量检索 + Ollama LLM
- **GraphRAG 模式**：基于 Neo4j 引用图 + Cypher 查询
- **自动路由**：智能判断问题类型并路由
"""
    )

    st.divider()
    st.markdown("### 🔧 系统组件")
    st.markdown(
        """
- **嵌入模型**：`qwen3-embedding:4b`（维度 2560）
- **生成模型**：`qwen3:4b`
- **向量库**：Qdrant（1560 chunks）
- **图库**：Neo4j（12 节点 / 31 边）
- **编排**：LangChain 1.x LCEL
"""
    )

    st.divider()
    st.markdown("### 📊 统计")
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.metric("本次会话查询数", st.session_state.query_count)

    if st.button("🗑️ 清空历史", use_container_width=True):
        st.session_state.history = []
        st.session_state.query_count = 0
        st.rerun()

# ============== 主页面 ==============
st.markdown('<div class="main-header">📚 学术论文问答系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">基于 RAG (Qdrant) + GraphRAG (Neo4j) 的智能问答 · 基于 LangChain LCEL 编排</div>',
    unsafe_allow_html=True,
)

# --- 模式选择 ---
mode = st.radio(
    "**选择问答模式**",
    options=["🔍 RAG 问答", "🕸️ GraphRAG 问答", "🤖 自动路由"],
    horizontal=True,
    help="RAG：基于论文内容语义检索 | GraphRAG：基于引用图关系查询 | 自动路由：系统自动判断",
)

# 映射到内部标识
MODE_KEY = {
    "🔍 RAG 问答": "rag",
    "🕸️ GraphRAG 问答": "graphrag",
    "🤖 自动路由": "auto",
}
mode_key = MODE_KEY[mode]

# 模式说明卡片
if mode_key == "rag":
    st.info("📌 **RAG 模式**：用 Qdrant 检索 Top-3 相关文本块 → LLM 生成答案。适合：'XX 论文的核心思想是什么？'")
elif mode_key == "graphrag":
    st.info("📌 **GraphRAG 模式**：用 Neo4j 引用图遍历 → LLM 整理答案。适合：'X 引用了谁？X 到 Y 的引用路径？'")
else:
    st.info("📌 **自动路由模式**：先用关键词 + 嵌入相似度快速判断；都不确定时调 LLM。系统自动选择最合适的处理路径。")

st.divider()

# --- 示例问题 ---
st.markdown("#### 💡 示例问题（点击直接提问）")
EXAMPLE_QUESTIONS = {
    "RAG 类（3 道）": [
        "RAPTOR 论文的核心思想是什么？",
        "CRAG 论文中，检索评估器会触发哪三种动作？",
        "FLARE 论文中，模型如何决定何时进行检索？",
    ],
    "GraphRAG 类（3 道）": [
        "STAR 论文引用了哪些论文？（列出标题）",
        "HiQA 论文是否引用了 RAPTOR？",
        "从 Lewis et al. 2020 到 STAR 论文的引用路径是什么？",
    ],
}

cols = st.columns(2)
with cols[0]:
    st.markdown("**RAG 类**")
    for q in EXAMPLE_QUESTIONS["RAG 类（3 道）"]:
        if st.button(q, key=f"ex_rag_{q}", use_container_width=True):
            st.session_state.pending_question = q
with cols[1]:
    st.markdown("**GraphRAG 类**")
    for q in EXAMPLE_QUESTIONS["GraphRAG 类（3 道）"]:
        if st.button(q, key=f"ex_graphrag_{q}", use_container_width=True):
            st.session_state.pending_question = q

# --- 输入框 ---
st.divider()
st.markdown("#### ✍️ 自由提问")

# 用 pending_question 同步到 question_input key，避免 rerun 丢失
if st.session_state.get("pending_question"):
    st.session_state.question_input = st.session_state.pending_question
    st.session_state.pending_question = ""

question = st.text_input(
    "请输入您的问题：",
    key="question_input",
    placeholder="例如：RAPTOR 论文的核心思想是什么？",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit = st.button("🚀 提交", type="primary", use_container_width=True)
with col2:
    clear = st.button("🔄 清空", use_container_width=True)

if clear:
    st.rerun()

# ============== 处理问答 ==============
def run_query(q: str, mk: str) -> dict:
    """根据模式调用对应 pipeline，返回统一格式的 result"""
    if mk == "rag":
        fn = get_rag()
        return fn(q)
    elif mk == "graphrag":
        fn = get_graphrag()
        return fn(q)
    else:  # auto
        fn = get_router()
        return fn(q)


if submit and question.strip():
    # 计数
    st.session_state.query_count = st.session_state.get("query_count", 0) + 1
    result = None

    with st.spinner("🤔 正在思考中..."):
        try:
            result = run_query(question.strip(), mode_key)
        except Exception as e:
            st.error(f"❌ 查询出错：{e}")

    if result is not None:
        # 存历史
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "question": question.strip(),
            "mode": mode_key,
            "result": result,
        })

# ============== 展示最新结果 ==============
if st.session_state.get("history"):
    latest = st.session_state.history[-1]
    result = latest["result"]
    q = latest["question"]

    st.divider()
    st.markdown(f"#### 📝 答案")
    st.markdown(f"**问题**：{q}")

    # --- 路由信息（仅 auto 模式）---
    if latest["mode"] == "auto":
        rd = result.get("router_decision", {})
        method = result.get("router_method", "?")
        scores = result.get("router_scores", {})
        dest_name = rd.get("destination_name", "?")
        reason = rd.get("reason", "")
        route_color = {
            "RAG": "badge-rag",
            "GraphRAG": "badge-graphrag",
        }.get(dest_name, "badge-auto")

        st.markdown('<div class="route-card">', unsafe_allow_html=True)
        st.markdown(
            f"**🤖 路由决策**：<span class='mode-badge {route_color}'>{dest_name}</span> &nbsp; "
            f"**方法**：`{method}` &nbsp; **理由**：{reason}",
            unsafe_allow_html=True,
        )
        if scores:
            st.markdown(
                f"<small>语义相似度得分：vector_rag = `{scores.get('vector_rag', 0):.3f}` · "
                f"graph_rag = `{scores.get('graph_rag', 0):.3f}`</small>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 答案 ---
    answer = result.get("answer", "（无答案）")
    route_label = result.get("route", "?")
    st.markdown(
        f'<div class="answer-card"><b>【{route_label} 回答】</b><br>{answer}</div>',
        unsafe_allow_html=True,
    )

    # --- RAG 模式：显示引用来源 ---
    if "sources" in result and result["sources"]:
        st.markdown("##### 📑 引用来源")
        for src in result["sources"]:
            arxiv = src.get("arxiv_id", "?")
            short = src.get("short_name", arxiv)
            title = src.get("title", "")
            chunk = src.get("chunk_id", "")
            snippet = src.get("snippet", "")
            st.markdown(
                f'<div class="source-card">'
                f'<b>[{src["n"]}] {short}</b> <small>({arxiv})</small><br>'
                f'<i>{title}</i> · <code>chunk={chunk}</code><br>'
                f'<small>…{snippet[:280]}{"…" if len(snippet) > 280 else ""}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- GraphRAG 模式：显示 Cypher + 论文标题 ---
    if "cypher" in result:
        with st.expander("🔍 GraphRAG 查询细节（点击展开）", expanded=True):
            st.markdown("**Cypher 查询语句：**")
            st.code(result["cypher"], language="cypher")
            if result.get("results"):
                st.markdown(f"**查询结果**（{len(result['results'])} 条）：")
                st.json(result["results"])
            if result.get("paper_titles"):
                st.markdown("**涉及论文：**")
                for p in result["paper_titles"]:
                    if "path" in p or "summary" in p:
                        st.write(f"  - {p}")
                    else:
                        st.write(
                            f"  - **{p.get('short_name', '?')}** "
                            f"({p.get('year', '?')}) — {p.get('title', '?')[:80]}"
                        )

    # --- RAG 模式：显示完整检索 chunk ---
    if "retrieved_chunks" in result and result["retrieved_chunks"]:
        with st.expander("📄 完整检索 chunks（点击展开）", expanded=False):
            for ch in result["retrieved_chunks"]:
                st.markdown(
                    f"**[{ch['n']}] {ch.get('short_name', '?')}** · `{ch.get('chunk_id', '')}`"
                )
                st.text(ch.get("text", "")[:800] + ("..." if len(ch.get("text", "")) > 800 else ""))
                st.divider()

# ============== 历史记录（折叠） ==============
if st.session_state.get("history") and len(st.session_state.history) > 1:
    with st.expander(f"📚 历史记录（{len(st.session_state.history) - 1} 条）", expanded=False):
        for i, h in enumerate(reversed(st.session_state.history[:-1]), 1):
            mode_emoji = {"rag": "🔍", "graphrag": "🕸️", "auto": "🤖"}.get(h["mode"], "?")
            r = h["result"]
            route = r.get("route", "?")
            ans = r.get("answer", "")[:200].replace("\n", " ")
            st.markdown(
                f"**{i}.** {mode_emoji} `{h['mode']}` → **{route}** | {h['question']}<br>"
                f"<small>{ans}{'...' if len(r.get('answer', '')) > 200 else ''}</small>",
                unsafe_allow_html=True,
            )

# ============== 页脚 ==============
st.divider()
st.markdown(
    "<small>💡 提示：自动路由会先用关键词 + 嵌入相似度快速判断；都不确定时调 LLM 出决策。</small>",
    unsafe_allow_html=True,
)
