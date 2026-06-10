# 基于 RAG + GraphRAG 的学术论文问答系统

> 综合运用 **Qdrant**、**Neo4j**、**Ollama**、**LangChain 1.x** 等课程所学知识，构建一个面向 10 篇 RAG / GraphRAG 相关论文的智能问答系统，支持语义检索、关系图查询与自动路由三种模式。

---

## 一、功能特性

- 🔍 **RAG 问答**：根据用户问题从论文正文检索 Top-K 相关段落，调用 LLM 生成带 [N] 引用编号的答案
- 🕸️ **GraphRAG 问答**：基于 Neo4j 中的论文引用图谱，查询"X 引用了哪些"、"X 是否引用 Y"、"X 到 Y 的引用路径"等关系
- 🤖 **智能路由**：三级 fallback 策略（关键词 → 语义嵌入 → LLM）自动判断问题类型并分发到对应模式
- 🌐 **Streamlit Web UI**：模式切换、示例问题、答案卡片、引用来源卡、查询过程可视化
- 📊 **数据准备流水线**：PDF → 分块 → 向量化入库 Qdrant；自动检测引用关系入库 Neo4j

---

## 二、技术栈

| 类别       | 技术 / 版本              | 用途                              |
|------------|--------------------------|-----------------------------------|
| 编程语言   | Python 3.13              | 主开发语言                        |
| 编排框架   | LangChain 1.x（LCEL）    | 链式组合 retriever / prompt / LLM |
| 向量数据库 | Qdrant (Docker)          | 存储 1560 个文本块向量            |
| 图数据库   | Neo4j 5 (Docker)         | 12 节点 + 31 条 CITES 边          |
| 嵌入模型   | Ollama `qwen3-embedding:4b` | 文本向量化（2560 维）             |
| 生成模型   | Ollama `qwen3:4b`        | 答案生成、Cypher 生成             |
| Web UI     | Streamlit 1.51           | 三模式交互界面                    |
| PDF 解析   | pypdf 5.0                | 论文 PDF 文本提取                 |

详细依赖见 [`requirements.txt`](./requirements.txt)。

---

## 三、项目结构

```
rag_qa_system/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖（锁定版本）
├── config.py                    # 统一配置（Qdrant/Neo4j/Ollama URL、模型名等）
├── verify_env.py                # 环境自检脚本
├── verify_citations.py          # 引用边校对脚本
│
├── graph/                       # 论文元数据 + 引用边定义
│   └── citations.py             # PAPERS 字典 + get_citations()
│
├── ingest/                      # 数据准备流水线
│   ├── pdf_to_text.py           # PDF → 纯文本
│   ├── chunking.py              # 文本分块（500 字/块，100 字重叠）
│   ├── build_qdrant.py          # 向量化 + 写入 Qdrant
│   ├── detect_citations.py      # 自动检测引用关系
│   └── build_neo4j.py           # 节点 + CITES 边写入 Neo4j
│
├── rag/                         # RAG 检索 + 生成
│   └── pipeline.py              # rag_query(question)
│
├── graphrag/                    # GraphRAG 关系查询
│   └── pipeline.py              # graphrag_query(question)
│
├── router/                      # 智能路由
│   └── router.py                # smart_route(question)
│
└── ui/                          # Streamlit Web 界面
    └── app.py                   # 启动入口
```

---

## 四、快速开始

### 4.1 环境准备

| 依赖      | 版本 / 镜像                                | 说明                                |
|-----------|--------------------------------------------|-------------------------------------|
| Python    | 3.10+                                      | 推荐 3.13                           |
| Docker    | Desktop / Engine                           | 用于运行 Qdrant / Neo4j 容器        |
| Ollama    | ≥ 0.6                                      | 本地 LLM 与嵌入服务                 |
| 内存      | ≥ 8 GB RAM                                 | 运行 4B 级别模型                    |
| 磁盘      | ≥ 20 GB 可用                               | 镜像 + 模型 + 向量数据              |

### 4.2 安装步骤

```bash
# 1. 解压源代码
unzip rag_qa_system_src.zip
cd rag_qa_system

# 2. 创建虚拟环境（可选但推荐）
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 启动 Docker 容器（Qdrant + Neo4j）
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j12345 \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5

# 5. 拉取 Ollama 模型
ollama pull qwen3:4b
ollama pull qwen3-embedding:4b

# 6. 验证环境
python verify_env.py
```

### 4.3 数据初始化（首次运行）

```bash
# 论文 PDF 文件放到 data/papers/ 目录后执行：
python ingest/pdf_to_text.py      # PDF → TXT
python ingest/chunking.py         # 文本分块
python ingest/build_qdrant.py     # 向量化 + 写入 Qdrant
python ingest/detect_citations.py # 自动检测引用关系
python ingest/build_neo4j.py      # 节点 + 边写入 Neo4j
```

> **注意**：仓库本身不包含 PDF / 缓存数据，需要自行准备 12 篇 RAG/GraphRAG 相关论文。

### 4.4 启动 UI

```bash
# Windows PowerShell
$env:NEO4J_PASSWORD = "neo4j12345"
streamlit run ui/app.py

# macOS / Linux
export NEO4J_PASSWORD=neo4j12345
streamlit run ui/app.py
```

浏览器自动打开 `http://localhost:8501`，可见三种模式选择 + 6 个示例问题按钮。

---

## 五、使用方法

### 5.1 三种模式

| 模式        | 适用问题                                              | 后端       |
|-------------|------------------------------------------------------|------------|
| 🔍 RAG 问答  | "RAPTOR 核心思想是什么？"等开放性内容问题             | Qdrant     |
| 🕸️ GraphRAG  | "STAR 引用了哪些论文？"、"HiQA 是否引用 RAPTOR？"等关系查询 | Neo4j      |
| 🤖 自动路由  | 任意问题，系统自动判断 + 展示路由决策与得分           | RAG + GraphRAG |

### 5.2 6 道测试题（与考试要求对应）

| # | 类型       | 问题                                                                  | 预期答案                |
|---|------------|----------------------------------------------------------------------|-------------------------|
| 1 | RAG        | RAPTOR 论文的核心思想是什么？                                         | 树状结构分层检索        |
| 2 | RAG        | CRAG 论文中，检索评估器会触发哪三种动作？                            | 正确 / 错误 / 模糊      |
| 3 | RAG        | FLARE 论文中，模型如何决定何时进行检索？                              | 生成 [Search(query)] token |
| 4 | GraphRAG   | STAR 论文引用了哪些论文？（列出标题）                                  | HiQA (2024) 等          |
| 5 | GraphRAG   | HiQA 论文是否引用了 RAPTOR？                                          | 否（图谱中无对应边）    |
| 6 | GraphRAG   | 从 Lewis 2020 到 STAR 的引用路径是什么？                              | Lewis → HiQA → STAR     |

---

## 六、关键模块说明

### 6.1 数据准备

- **分块策略**：`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`
- **Embedding**：调用 `OllamaEmbeddings(model="qwen3-embedding:4b")`，输出维度 2560
- **Qdrant 集合**：`papers`，配置 `VectorParams(size=2560, distance=Cosine)`，共 1560 个点
- **Neo4j Schema**：
  - 节点：`(Paper {arxiv_id, short_id, title, first_author, short_name, year})`
  - 关系：`(paperA)-[:CITES]->(paperB)`，共 31 条边

### 6.2 RAG 检索 + 生成

- 用户问题 → 嵌入 → Qdrant 检索 Top-3 → 构造 prompt（context + question）→ `qwen3:4b` 生成答案
- 返回结构：`{answer, sources, retrieved_chunks, route: "RAG"}`

### 6.3 GraphRAG 关系查询

- **论文别名识别**：内置 `ALIASES` 字典（RAPTOR → Sarthi 2024，STAR → Li 2026 等）
- **三种 Cypher 模板**：
  1. 模板 A：`MATCH (a:Paper {arxiv_id})-[:CITES]->(b) RETURN b.title, b.short_name`
  2. 模板 B：`MATCH (b:Paper {arxiv_id})<-[:CITES]-(a) RETURN a.title, a.short_name`
  3. 模板 C：`MATCH p = (a:Paper)-[:CITES*1..6]-(b:Paper) WHERE ... RETURN p`
- **LLM 兜底**：当关键词路由 + 别名识别都失败时，调用 LLM 生成 Cypher

### 6.4 智能路由（三级 fallback）

```
1. 关键词路由（regex 匹配"引用了"/"是否引用"等 → 直接判定为 graph_rag，0 LLM 调用）
   ↓ 不命中
2. 语义路由（计算问题与各目的地平均嵌入向量的余弦相似度，margin ≥ 0.03 即可判定）
   ↓ 差距过小
3. LLM 路由（ChatOllama 生成 JSON 路由决策，regex 解析）
```

测试集 6 道题 **100% 命中前两级**，LLM 兜底未被触发。

---

## 七、常见问题

### Q1: 启动时 "Connection refused" to Qdrant / Neo4j

**A**: Docker 容器未运行。执行：
```bash
docker ps                 # 查看运行中的容器
docker start qdrant neo4j
```

### Q2: Ollama 报 "model not found"

**A**: 未拉取模型。执行：
```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:4b
```

### Q3: Qdrant 集合为空 / 维度不匹配

**A**: 需要先运行 `python ingest/build_qdrant.py` 重新构建。

### Q4: Neo4j 密码错误

**A**: 启动容器时设置了 `NEO4J_AUTH=neo4j/neo4j12345`，运行 UI 前需设环境变量：
```bash
$env:NEO4J_PASSWORD = "neo4j12345"   # Windows PowerShell
export NEO4J_PASSWORD=neo4j12345     # macOS / Linux
```

### Q5: 中文路径 / PowerShell 编码报错

**A**: 建议将项目放在纯英文路径下（如 `D:\projects\rag_qa_system`），或运行脚本前设置：
```powershell
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 八、参考资源

- **相关论文清单**：RAG / GraphRAG 相关论文 12 篇（Lewis 2020, Edge 2024, Gupta 2024 等）
- **LangChain 文档**：https://python.langchain.com/docs/introduction/
- **Qdrant 文档**：https://qdrant.tech/documentation/
- **Neo4j Cypher 参考**：https://neo4j.com/docs/cypher-manual/current/
- **Ollama 模型库**：https://ollama.com/library

---

## 九、版本

| 项目       | 版本 / 日期   |
|------------|---------------|
| LangChain  | 1.3.1         |
| Qdrant     | 1.18.0 client |
| Neo4j      | 6.2.0 client / 5.x server |
| Python     | 3.13          |
| 报告版本   | v1.0（2026-06） |

---

## 十、许可证与声明

本项目为个人学习作品，仅供学习交流使用。论文原文版权归原作者所有。
