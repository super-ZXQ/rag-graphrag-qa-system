"""
RAG 问答模块（步骤 6） — 基于 LangChain 1.x LCEL
流程：
  用户问题 → OllamaEmbeddings 嵌入 → 自定义 Retriever 检索 Top-K 文本块
         → 拼接 context → ChatPromptTemplate → ChatOllama → StrOutputParser
         → 返回 (答案 + 引用来源 + 检索细节)

依赖（config.py 已配置）：
  - EMBED_MODEL = "qwen3-embedding:4b"
  - LLM_MODEL   = "qwen3:4b"
  - QDRANT_URL  = "http://localhost:6333"
  - QDRANT_COLLECTION = "papers"
  - RETRIEVER_K = 3
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EMBED_MODEL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RETRIEVER_K,
)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient

from graph.citations import PAPERS

sys.stdout.reconfigure(encoding="utf-8")

# ============== 1. 初始化组件 ==============
# 嵌入模型（与 build_qdrant.py 保持一致，否则向量空间不对齐）
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

# 连接 Qdrant 现有 collection（不重建；只读模式）
client = QdrantClient(url=QDRANT_URL)


# ============== 1.1 自定义 Retriever ==============
# 原因：build_qdrant.py 写入的 payload 是扁平的 {arxiv_id, chunk_id, filename, text}；
# langchain-qdrant 1.x 的 QdrantVectorStore 默认按 content_payload_key='page_content'、
# metadata_payload_key='metadata' 取，找不到 key → page_content 为空、metadata 只有 _id。
# 不重新嵌入，直接读 raw payload 再手工包装成 Document。
def qdrant_flat_search(query: str) -> list[Document]:
    """用 Ollama 嵌入 query，在 Qdrant 中取 Top-K，扁平 payload → Document。"""
    vec = embeddings.embed_query(query)
    # qdrant-client 1.x 取消了 search()，用 query_points() 替代
    resp = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        limit=RETRIEVER_K,
        with_payload=True,
        with_vectors=False,
    )
    docs: list[Document] = []
    for hit in resp.points:
        p = hit.payload or {}
        meta = {
            "arxiv_id": p.get("arxiv_id", ""),
            "chunk_id": p.get("chunk_id", ""),
            "filename": p.get("filename", ""),
            "score": hit.score,
            "_id": hit.id,
        }
        docs.append(Document(page_content=p.get("text", ""), metadata=meta))
    return docs


# 包成 RunnableLambda 才能进 LCEL 管道（用 `|` 串到下游）
retriever = RunnableLambda(qdrant_flat_search)

# 生成 LLM
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
)

# ============== 2. 文档格式化 ==============
# payload 里有 arxiv_id / chunk_id / filename / text
# short_name / title 走 PAPERS 字典查表，方便界面展示


def _doc_label(doc: Document) -> str:
    """把一篇检索结果拼成 '作者.年份 (短名)' 这种人类可读标签"""
    aid = doc.metadata.get("arxiv_id", "?")
    meta = PAPERS.get(aid, {})
    return meta.get("short_name", aid)


def format_docs(docs: list[Document]) -> str:
    """把 Top-K 文档拼成 context 字符串；每段前加 [N] 编号 + 论文短名"""
    parts = []
    for i, d in enumerate(docs, 1):
        label = _doc_label(d)
        parts.append(f"[{i}] 来源：{label}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ============== 3. Prompt 模板 ==============
# 用 ChatPromptTemplate（system + human）让模型明确"只根据检索内容回答"
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个学术论文问答助手。请严格根据【检索内容】回答用户问题。\n"
        "要求：\n"
        "1) 答案用中文，简洁准确。\n"
        "2) 引用检索内容时用 [N] 编号标注，对应下方【检索内容】的序号。\n"
        "3) 如果【检索内容】中确实没有答案，请直接说「检索内容未覆盖此问题」，不要编造。\n\n"
        "【检索内容】\n{context}",
    ),
    ("human", "{question}"),
])

# ============== 4. LCEL 链 ==============
# 同时拿"上下文"和"原问题"喂给 prompt；context 经过 retriever + format_docs 管道
rag_chain = (
    RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough(),
    )
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)


def _llm_invoke_with_retry(chain, inputs, max_retries: int = LLM_MAX_RETRIES):
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
                return f"LLM 生成失败（{e}），请重试。"


# ============== 5. 对外接口 ==============
def rag_query(question: str) -> dict:
    """
    RAG 问答主入口
    返回：
      - answer: str  LLM 生成的答案（带 [N] 引用编号）
      - sources: list[dict]  引用来源列表（含 arxiv_id / short_name / chunk_id / text 摘录）
      - retrieved_chunks: list[dict]  完整检索细节（用于界面"展开"面板）
    """
    # 1) 检索一次（避免双重嵌入浪费）
    docs: list[Document] = retriever.invoke(question)
    context_str = format_docs(docs)

    # 2) 直接用 prompt + LLM 生成答案（跳过链内 retriever，带重试）
    answer: str = _llm_invoke_with_retry(
        RAG_PROMPT | llm | StrOutputParser(),
        {"context": context_str, "question": question},
    )

    # 3) 整理引用来源
    sources = []
    for i, d in enumerate(docs, 1):
        aid = d.metadata.get("arxiv_id", "?")
        meta = PAPERS.get(aid, {})
        sources.append({
            "n": i,
            "arxiv_id": aid,
            "short_name": meta.get("short_name", aid),
            "title": meta.get("title", d.metadata.get("filename", "")),
            "filename": d.metadata.get("filename", ""),
            "chunk_id": d.metadata.get("chunk_id", ""),
            "snippet": d.page_content[:300],  # 截前 300 字符用于界面展示
        })

    # 4) 检索细节（完整文本，供"展开"面板用）
    retrieved_chunks = [
        {
            "n": i,
            "arxiv_id": d.metadata.get("arxiv_id", "?"),
            "short_name": _doc_label(d),
            "chunk_id": d.metadata.get("chunk_id", ""),
            "text": d.page_content,
        }
        for i, d in enumerate(docs, 1)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "route": "RAG",
    }


# ============== 6. 自检入口 ==============
if __name__ == "__main__":
    # 跑 3 道 RAG 测试题
    test_questions = [
        "RAPTOR 论文的核心思想是什么？",
        "CRAG 论文中，检索评估器会触发哪三种动作？",
        "FLARE 论文中，模型如何决定何时进行检索？",
    ]
    # 同时写一份 UTF-8 干净版（PowerShell 的 > 重定向会强制 UTF-16 BOM）
    out_log = Path(__file__).parent.parent / "data" / "rag_test_output.txt"
    log_lines = []
    for q in test_questions:
        block = []
        block.append("\n" + "=" * 70)
        block.append(f"[Q] {q}")
        block.append("=" * 70)
        out = rag_query(q)
        block.append(f"\n[A]\n{out['answer']}\n")
        block.append(f"[sources] {len(out['sources'])} chunks:")
        for s in out["sources"]:
            block.append(
                f"  [{s['n']}] {s['short_name']:<48}  chunk={s['chunk_id']}"
            )
            block.append(f"      {s['snippet'][:150]}...")
        log_lines.append("\n".join(block))
        # 实时输出到 stdout（让终端看得到）
        print(block[0])
        print(block[1])
        print(block[2])
        print(block[3])
        print(block[4])
        for ln in block[5:]:
            print(ln)
    # 写干净 UTF-8 文件
    out_log.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[rag] test log -> {out_log} (utf-8)")
