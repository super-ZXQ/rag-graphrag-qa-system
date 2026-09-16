# 企业知识库 RAG 技术选型报告

## 执行摘要

工程推断：根据业务硬约束和用户权重，建议把 基础混合检索 RAG 作为首期基线，并通过 POC 再决定是否叠加复杂技术。

## 关键事实

- Atlas 初期语料约 180,000 份文档，其中约 35% 是超过 30 页的长文档 [E-local-bcba105fe48f-p1-c0]。
- 现有环境提供 16 核 CPU、64 GB 内存和单张 24 GB 显存 GPU [E-local-f4c9fff07eb3-p1-c0]。
- 首期由 4 人小组在 8 周内交付 [E-local-f4c9fff07eb3-p1-c0]。
- GraphRAG results show stronger performance than vector RAG when GPT-4 is used as the LLM [E-local-730f1a9f38d1-p2-c4]。

## 风险与未知项

- 工程推断：论文结果不能直接等同于 Atlas 线上效果，仍需使用企业问题集验证延迟、准确性与维护成本。
- 工程推断：精确证据召回仍低于文档召回，应优先优化分块、重排与跨文档证据聚合。

## 推荐草案

- 建议首期采用 基础混合检索 RAG；硬约束失败的候选不进入首选。
- 建议用 2 周 POC 测试可答率、精确证据 Recall@5、P95 延迟和增量索引耗时。

## 参考资料

- [E-local-730f1a9f38d1-p2-c4]
- [E-local-bcba105fe48f-p1-c0]
- [E-local-f4c9fff07eb3-p1-c0]

## 系统确定性业务评分

| 候选方案 | 检索质量与可追溯性 | 在线延迟 | 数据更新复杂度 | 实施成本 | 运行成本 | 隐私与部署适配 | 运维与团队适配 | 加权总分/5 | 硬约束 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 基础混合检索 RAG | 3.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 4.5 | 通过 |
| RAPTOR | 5.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.5 | 通过 |
| CRAG | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 1.0 | 3.0 | 2.8 | 不通过 |
| FLARE | 5.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.5 | 通过 |
| GraphRAG | 3.0 | 3.0 | 1.0 | 1.0 | 3.0 | 3.0 | 1.0 | 2.2 | 通过 |

> 工程推断：评分按用户权重确定性计算；1/3/5 分分别表示冲突、部分契合、契合。
> 工程推断：未映射项取中性 3 分，不冒充线上实验结果；硬约束失败的方案不能成为首选。
> 工程推断：规则计算首选为 基础混合检索 RAG，最终仍需人工确认。


## 推荐证据链（引用图）

> 本节由本地 OpenAlex 引用图确定性生成，不是模型编写。
> 邻居论文只有元数据，不能证明其独立支持推荐；全文未入库时不计为证据。

### RAPTOR

- 种子论文数：1
- 引用网络邻居数：6
- 已验证独立证据数：0
- 独立多论文支持：未验证（建议人工复核邻居全文）
- 高相关引用/被引论文（按被引数排序，最多 5 篇）：
  - `W4404534210` · 2024 · cited_by=2046 · A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challe
  - `W4410929991` · 2025 · cited_by=72 · Retrieval-Augmented Generation (RAG)
  - `W4410536095` · 2025 · cited_by=18 · Customized large-scale model for human-AI collaborative operation and maintenanc
  - `W4405810201` · 2024 · cited_by=18 · CRP-RAG: A Retrieval-Augmented Generation Framework for Supporting Complex Logic
  - `W4408360053` · 2025 · cited_by=15 · Agricultural large language model for standardized production of distinctive agr

### CRAG

- 种子论文数：1
- 引用网络邻居数：6
- 已验证独立证据数：0
- 独立多论文支持：未验证（建议人工复核邻居全文）
- 高相关引用/被引论文（按被引数排序，最多 5 篇）：
  - `W4404355908` · 2024 · cited_by=39 · Agent design pattern catalogue: A collection of architectural patterns for found
  - `W4407197060` · 2025 · cited_by=36 · Accelerating Retrieval-Augmented Generation
  - `W4399653983` · 2024 · cited_by=27 · Future of Evidence Synthesis: Automated, Living, and Interactive Systematic Revi
  - `W4408734515` · 2025 · cited_by=23 · RAGVA: Engineering retrieval augmented generation-based virtual assistants in pr
  - `W4408046520` · 2025 · cited_by=22 · Use of Retrieval-Augmented Large Language Model for COVID-19 Fact-Checking: Deve

### FLARE

- 种子论文数：1
- 引用网络邻居数：6
- 已验证独立证据数：0
- 独立多论文支持：未验证（建议人工复核邻居全文）
- 高相关引用/被引论文（按被引数排序，最多 5 篇）：
  - `W4389520468` · 2023 · cited_by=180 · Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Gen
  - `W4409284961` · 2025 · cited_by=22 · Intelligent, Personalized Scientific Assistant via Large Language Models for Sol
  - `W4389519208` · 2023 · cited_by=14 · Knowledge-Augmented Language Model Verification
  - `W4406707540` · 2025 · cited_by=7 · Augmenting LLMs to Securely Retrieve Information for Construction and Facility M
  - `W4405165896` · 2024 · cited_by=6 · Biological Database Mining for LLM-Driven Alzheimer’s Disease Drug Repurposing
