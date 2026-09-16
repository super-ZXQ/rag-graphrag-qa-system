# ScholarGraph：企业研发技术选型 Agent

ScholarGraph 将公开论文、引用关系和企业内部约束组织成可核验的技术选型报告。它不是只回答“论文说了什么”，而是回答“在当前业务约束下应该怎么选、依据是什么、还需要验证什么”。

## 为什么做这个项目

研发团队做 RAG 等新技术选型时，常见问题不是资料不足，而是论文结论、内部限制和工程成本彼此割裂。ScholarGraph 建立一条完整决策链：

1. 导入内部 PDF/Markdown，并从 arXiv 搜索公开论文。
2. 在本地解析、增量索引，通过向量检索和 FTS5 召回证据。
3. 受控 Agent 按固定步骤比较候选方案。
4. 根据业务约束和用户权重确定性计算 1–5 分矩阵，硬约束失败的方案不得成为首选。
5. 报告中的关键事实绑定稳定证据 ID、来源和页码。
6. AI 只生成建议草案，人工确认后才成为最终报告。

## 核心能力

- **增量资料库**：上传后经历 `parsed → indexing → ready / index_failed`，内容哈希保证幂等。
- **混合检索**：本地 Ollama embedding + Qdrant 向量检索 + SQLite FTS5，使用 RRF 融合。
- **公开资料连接器**：arXiv 搜索与 PDF 导入；OpenAlex 元数据搜索。
- **受控研究 Agent**：显式状态节点、最多 12 步、单任务执行、失败状态持久化。
- **证据约束报告**：区分论文事实、内部事实、工程推断和证据不足。
- **业务决策矩阵**：约束激活、逐指标评分、加权总分和隐私硬约束均由代码确定性计算；未映射项明确使用中性分并要求人工复核。
- **Claim 双层校验**：硬校验确定性核验引用存在且能定位到 paper/chunk/page；软校验（LLM 仅辅助，默认保守）判断结论是否被证据支持，输出支持/部分支持/冲突/证据不足计数。LLM 无权把硬伤“翻案”为支持。
- **审批门槛**：引用存在率 100%、关键论文/内部事实均有有效证据、无虚构引用、工程推断已显式标记、证据不足时拒绝推荐，才允许人工确认。
- **任务生命周期**：可取消、失败持久化（failed_step/error_type）、从失败处恢复、修订保留历史版本，并记录模型调用、耗时、检索/发送片段数与 Token 预算。
- **隐私边界**：整篇内部文档不发送给生成模型；仅发送本地命中的必要片段，日志不记录片段全文。
- **提示注入防护**：论文中的 Prompt/JSON 输出模板在进入模型上下文前被确定性过滤并计数；异常代码围栏报告直接拒绝审批。
- **人工审批**：报告状态为 `draft`，审批后才进入 `approved/completed`。
- **可复现评测**：仓库包含虚构企业材料、检索样例，以及分别计算文档召回和精确证据召回的运行脚本。

## 架构

```text
PDF / Markdown / arXiv
          │
          ▼
  Parse + stable chunks ───────► SQLite + FTS5
          │                           │
          ▼                           │
  Local Ollama embedding              │
          │                           │
          ▼                           ▼
       Qdrant ───── RRF hybrid retrieval ──► (可选) 引用图弱重排
                              │                      │
                              │               OpenAlex 引用图
                              │            SQLite / Neo4j 只读模板
                              ▼
                    bounded research workflow
                              │
               necessary evidence snippets only
                              │
                     DeepSeek / Ollama
                              │
          deterministic weighted matrix + claim gate
                              │
                        human approval
                              │
              Web report / Markdown + 证据链
```

Neo4j 保留用于论文引用关系和受限图查询；系统不执行模型自由生成的 Cypher。

## 快速开始

### 1. 配置

```powershell
Copy-Item .env.example .env.local
```

在 `.env.local` 填入新的 `DEEPSEEK_API_KEY`。密钥只从环境变量读取；不要提交该文件。若密钥曾出现在聊天、截图或日志中，应先到提供商控制台轮换。

本地嵌入依赖 Ollama：

```powershell
ollama pull qwen3-embedding:4b
```

### 2. 启动（本地模式，推荐开发/演示）

Docker Engine 不可用时直接用本机进程 + Qdrant 本地持久化：

```powershell
$env:QDRANT_PATH = 'data/qdrant_local'
# 无 DeepSeek 余额时用本地生成模型
$env:LLM_PROVIDER = 'ollama'
$env:LLM_MODEL = 'qwen2.5:7b'

uvicorn api.app:app --host 127.0.0.1 --port 8000
# 另开终端
streamlit run ui/app.py --server.port 8501
```

- 产品界面：<http://localhost:8501>
- API 文档：<http://localhost:8000/docs>
- 存活检查：<http://localhost:8000/health/live>

### 3. 启动（Docker）

```powershell
docker compose up --build
```

- UI: <http://localhost:8501>
- API: <http://localhost:8000/docs>
- Qdrant: <http://localhost:6333>
- Neo4j: <http://localhost:7474>

约束与设计：

- `.env.local` **不会**打进镜像（见 `.dockerignore`）；密钥只经 `env_file`/环境变量注入。
- API 容器内强制 `QDRANT_PATH=""`，使用独立 Qdrant 服务；`./data` 挂载 volume。
- Ollama 在宿主机：`OLLAMA_BASE_URL` 默认 `http://host.docker.internal:11434`。
- Engine 未启动时 `docker compose up` 会失败；先启动 Docker Desktop，或退回上面的本地模式。

### 4. 加载黄金演示材料

`demo_data/internal` 是明确标注为虚构的企业需求、架构约束和运行基线：

```powershell
$env:QDRANT_PATH = 'data/qdrant_local'   # 本地模式；Docker 模式请清空该变量
python scripts/seed_demo.py --index
python scripts/import_golden_papers.py --index
python scripts/build_citation_graph.py --limit 6
```

产品内黄金路径：

1. **01 资料库**：确认内部 3 份 + 公开 4 篇为「可检索」。
2. **02 创建选型任务**：打开「黄金场景模板」，权重合计 100，候选 5 条，启动。
3. **03 研究过程与报告**：看时间线与预算 → 展开证据与 claim → 门槛通过后人工确认 → 下载 Markdown。

也可直接运行脚本生成报告：

```powershell
python scripts/run_golden_demo.py --output evaluation/golden_report.md
```

## API 主流程

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/papers` | 上传内部 PDF/Markdown，并启动索引 |
| `GET` | `/sources/search` | 搜索 arXiv 或 OpenAlex |
| `POST` | `/sources/import` | 通过受限 arXiv ID 导入公开 PDF |
| `POST` | `/papers/{id}/retry` | 重试失败的索引 |
| `POST` | `/research-tasks` | 创建受控研究任务 |
| `GET` | `/research-tasks/{id}` | 查询步骤、事件、失败状态与预算 |
| `POST` | `/research-tasks/{id}/cancel` | 取消排队/运行中的任务（节点边界生效，保留证据与草案） |
| `POST` | `/research-tasks/{id}/resume` | 从失败节点恢复，复用已冻结证据 |
| `POST` | `/research-tasks/{id}/revise` | 待审批报告打回修订，版本号递增并保留历史版本 |
| `GET` | `/research-tasks/{id}/report/versions` | 查看报告历史版本 |
| `GET` | `/research-tasks/{id}/report` | 获取最新结构化报告数据 |
| `GET` | `/research-tasks/{id}/report.md` | 下载 Markdown |
| `POST` | `/research-tasks/{id}/approve` | 人工确认最终建议 |

同一时间只允许一个 `queued/running` 研究任务，新的任务返回 `409`。

任务失败会持久化 `failed_step` 与 `error_type`，可在节点边界取消、从失败处恢复，或在审批时打回修订（旧报告标记为 `superseded`）。预算信息记录模型调用次数、各节点耗时、检索条数、发送片段数，以及 Token 用量（优先取 API 返回的 usage）；成本估算默认关闭，只有显式配置 `DEEPSEEK_PRICE_PER_1M_INPUT/OUTPUT` 后才输出，避免臆造数字。

## 评测与测试

```powershell
python -m pytest -q
$env:QDRANT_PATH='data/qdrant_local'
python evaluation/run_retrieval.py --output evaluation/results.json
python scripts/run_golden_demo.py --output evaluation/golden_report.md
python scripts/build_citation_graph.py --limit 6
```

评测脚本只在资料已加载并完成索引后运行，指标由当前索引现场计算，不写死“提升幅度”。

### 真实结果（2026-09-16，本地 Qdrant + qwen3-embedding:4b）

语料：7 篇（3 内部 + 4 黄金论文）、475 块；评测 40 条（35 answer / 5 refuse）。

| 模式 | 文档 R@1 | 文档 R@5 | 文档 MRR | 精确证据 R@1 | 精确证据 R@5 | 精确证据 MRR |
|---|---:|---:|---:|---:|---:|---:|
| dense_only | 0.6286 | 0.8000 | 0.6962 | 0.3429 | 0.6286 | 0.4457 |
| fts5_only | 0.5143 | 0.5429 | 0.5214 | 0.1714 | 0.2571 | 0.2010 |
| hybrid_rrf | 0.6571 | 0.8286 | 0.7262 | 0.3714 | 0.6000 | 0.4581 |
| hybrid_graph | 0.6571 | 0.8286 | 0.7262 | 0.3714 | 0.5714 | 0.4524 |

诚实结论：

- 文档级 R@5：hybrid RRF **0.8286**，略高于 dense **0.8000**，显著高于 FTS5 **0.5429**。
- 精确证据 R@5：hybrid **0.6000**，说明“找到相关文档”不等于“找齐正确证据”，这是下一轮检索优化目标。
- 引用图模式是对已召回结果的弱信号重排，不会把只有元数据的邻居当证据；本语料上未提升 Recall，结果如实保留。
- 可答性探针（Ollama `qwen2.5:7b`）：无答案拒答准确率 **1.0**（5/5）；抽样可答问题准确率 **0.6667**（6/9，3 例保守拒答）。
- Claim 校验包含无引用事实绕过用例；正例引用存在率、负例拒绝率均单独报告。

`evaluation/golden_report.md` 是通过 claim gate 的 fail-closed 展示版；事实均来自本次冻结证据，业务排序来自确定性矩阵。被本地 7B 模型生成但未通过门槛的原始草案保存在 `evaluation/rejected_report.md`，便于演示系统为何拒绝，而不是为了指标放宽规则。

## 引用图与证据链

```powershell
python scripts/build_citation_graph.py --limit 6
```

- 用 OpenAlex DOI 优先定位种子论文，并用 **标题片段 + 年份** 双重校验，避免 RAPTOR→mTOR、CRAG→洞穴论文等同名误匹配。
- 图存储默认 SQLite（`data/citation_graph.sqlite3`，已 gitignore）；可选 Neo4j（固定参数化只读 Cypher）。
- 报告末尾追加确定性引用网络小节：展示种子论文和一跳邻居；只有元数据的邻居明确标为“独立证据未验证”。
- GraphRAG 原文 arXiv:2404.16130 在 OpenAlex 无可靠记录时跳过，不写入错误种子。

## 项目结构

```text
api/              FastAPI 接口
demo_data/        可公开复现的虚构企业资料
docs/             演示脚本 / 简历描述 / 面试问答
evaluation/       检索评测样例与运行器
graph/            领域映射、引用图存储与扩展
ingest/           PDF/Markdown 摄取与增量索引
paper_library/    SQLite schema、FTS5、任务与报告持久化
research/         受控单 Agent 工作流与引用校验
retrieval/        向量 + 关键词 RRF 混合检索
scripts/          黄金论文导入 / 评测 / 引用图构建
sources/          arXiv/OpenAlex 连接器
ui/               技术选型工作台
tests/            单元和 API 测试
```

## 交付验收清单

| 项 | 状态 |
|---|---|
| 上传后自动索引并可检索 | 已实现（parsed→indexing→ready / index_failed + 重试） |
| 任务创建 / 取消 / 失败恢复 / 修订 / 审批 | 已实现并有测试 |
| 本地 Qdrant 模式可复现评测与黄金报告 | 已跑通（`evaluation/results.json`） |
| Docker Compose 配置与密钥隔离 | 配置语法与固定镜像版本已验证；本机 Engine 未启动，端到端构建由 CI/可用 Engine 复验 |
| 引用存在率门槛 + 人工确认 | 已实现；无引用事实、部分支持、提示注入格式均拒绝审批 |
| 真实评测（非手写指标） | 文档级与精确证据级指标均已落盘，见上文 |
| UI 三页产品化 | 已完成并浏览器验证 |
| 作品集材料 | `docs/demo_script.md` / `resume.md` / `interview_qa.md` |

## 安全提醒

- `.env.local` 已被 Git 忽略，且不会打进 Docker 镜像。
- 若 DeepSeek Key 曾出现在聊天、截图或日志中，**请立即在控制台撤销并轮换**；新 Key 只放 `.env.local` 或系统环境变量。
- 日志与 UI 不展示内部文档全文；生成时仅发送命中必要片段。

## 当前边界

- 扫描版 PDF 暂不做 OCR。
- V1 使用本地单任务后台执行，不引入 Redis/Celery。
- API 启动时会把进程中断遗留的 `queued/running` 任务转为可恢复的 `failed`，避免永久卡死。
- OpenAlex 用于发现和元数据；系统不会绕过版权限制下载论文。部分 arXiv 论文（如 GraphRAG 2404.16130）在 OpenAlex 缺可靠记录时跳过图谱种子。
- 报告引用校验包含确定性硬校验与保守软校验；LLM 软校验可能误判，关键结论仍以可定位证据和人工确认为准。
- 本项目不把 LLM 评分包装成真实线上实验，工程推断会在报告中明确标记。
- DeepSeek 余额不足时可用 `LLM_PROVIDER=ollama` + `LLM_MODEL=qwen2.5:7b` 做真实本地生成；小模型引用纪律弱于商业 API，审批门槛会如实拒绝不合格报告。

## 技术栈

Python 3.12+、FastAPI、Streamlit、SQLite/FTS5、Qdrant、Neo4j、LangChain、Ollama Embeddings、DeepSeek OpenAI-compatible API。

本项目用于个人作品集与工程实践；论文版权归原作者所有，演示企业资料均为虚构。
