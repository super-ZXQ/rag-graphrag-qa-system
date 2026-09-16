# 企业知识库 RAG 技术路线选型报告（草案，待人工确认）

> 说明：本报告只依据给定证据与业务约束编写；系统确定性业务评分是权威业务排序，不重新计算或改写。文中区分「论文事实」「内部事实」「工程推断」；推荐为草案，必须等待人工确认。

## 执行摘要

- 内部事实：目标知识库初期约 180,000 份文档，其中 35% 是超过 30 页的长文档，每日新增或变更约 2,000 份，文档包含内部敏感资料，中文约 70%、英文约 30%；回答必须展示来源文档和页码，找不到依据时必须明确拒答，普通问题首屏响应目标 10 秒以内，复杂跨文档问题可在 60 秒内生成研究型答案 [E-local-bcba105fe48f-p1-c0]。
- 内部事实：原始内部文档和完整索引不得上传到第三方服务；允许将本地检索选出的必要片段发送给已签约大模型 API，但每次请求必须限制片段数量并记录证据标识；日志不得保存片段全文和 API Key [E-local-f4c9fff07eb3-p1-c0]。
- 内部事实：现有环境可运行 Docker，提供 16 核 CPU、64 GB 内存和单张 24 GB 显存 GPU；团队已有 PostgreSQL 和对象存储运维经验，但没有大规模图数据库生产经验；首期由 4 人小组在 8 周内交付，应优先采用可增量更新、可观测、可回滚的组件，避免复杂分布式调度系统 [E-local-f4c9fff07eb3-p1-c0]。
- 系统确定性业务评分显示：基础混合检索 RAG 加权总分 4.5 且硬约束通过；RAPTOR 与 FLARE 为 3.5 且硬约束通过；CRAG 为 2.8 且硬约束不通过；GraphRAG 为 2.2 且硬约束通过。工程推断：硬约束失败方案不能成为首选。
- 推荐草案：首选基础混合检索 RAG，结合企业隐私、更新、延迟与团队约束落地；可把 RAPTOR 作为长文档离线增强候选、FLARE 作为复杂问题可选策略。最终选择等待人工确认。

## 需求与假设

### 内部事实

- Atlas 面向研发与售后团队，首期服务 600 名员工，日均约 8,000 次查询；初期约 180,000 份文档，35% 为超过 30 页长文档，每日新增或变更约 2,000 份，文档属于内部敏感资料 [E-local-bcba105fe48f-p1-c0]。
- 回答必须展示来源文档和页码；找不到依据时必须明确拒答，不能补全不存在的事实；普通问题首屏 10 秒以内，复杂跨文档问题 60 秒内生成研究型答案 [E-local-bcba105fe48f-p1-c0]。
- 安全边界要求原始内部文档和完整索引不得上传第三方；允许必要检索片段发送给已签约大模型 API，但必须限制片段数量并记录证据标识；日志不得保存片段全文和 API Key [E-local-f4c9fff07eb3-p1-c0]。
- 基础设施为 Docker、16 核 CPU、64 GB 内存、单张 24 GB GPU；已有 PostgreSQL 和对象存储运维经验，没有大规模图数据库生产经验；4 人 8 周交付，优先增量更新、可观测、可回滚，避免复杂分布式调度 [E-local-f4c9fff07eb3-p1-c0]。

### 工程假设与待确认项

- 工程推断：长文档占比高意味着需要兼顾分块检索与跨段落概括能力，但具体中文长文档效果需要 POC 验证；本地证据未提供本系统实测检索质量，证据不足。
- 工程推断：每日约 2,000 份变更要求索引更新幂等、可回滚、可观测；未提供具体增量流水线实测吞吐，证据不足。
- 证据不足：未提供线上查询分布、片段大小限制、模型 API 限流、GPU 离线构建预算、标注预算和中文检索基准；这些不能从论文实验直接外推。

## 候选方案

### 基础混合检索 RAG

- 论文事实：传统 RAG 检索过程返回固定数量与查询语义相似的记录，生成答案只使用这些检索记录；常见做法是使用文本嵌入，在向量空间返回最接近查询的记录 [E-local-730f1a9f38d1-p2-c6]。
- 内部事实：其部署方式与安全边界较契合，因为原始内部文档和完整索引可留在本地，只允许必要片段发送给已签约大模型 API，并限制片段数量、记录证据标识 [E-local-f4c9fff07eb3-p1-c0]。
- 题设人工整理（非实验实测）：基础混合检索 RAG 在隐私、每日增量、普通问题 10 秒、4 人 8 周交付、单张 24 GB GPU 等约束上标记为 fits；但对 35% 超过 30 页长文档的跨段落概括支持有限，标记为 partial。
- 证据不足：本地证据未覆盖 Qdrant upsert、FTS 幂等增量、中文混合检索效果和本系统延迟实测。

### RAPTOR

- 论文事实：RAPTOR 使用多层级树结构；论文报告不同数据集和检索器下，18.5% 到 57% 的检索节点来自非叶节点，表明多层级树结构对检索有贡献 [E-local-01319e2e6e6e-p22-c2]。
- 论文事实：在 NarrativeQA 上，带 RAPTOR 的 SBERT、BM25、DPR 均优于不带 RAPTOR 的对应基线；论文还报告 QuALITY 数据集上准确率为 62.4% [E-local-01319e2e6e6e-p7-c2]。这些是论文实验数字，不是本系统实测。
- 题设人工整理（非实验实测）：RAPTOR 对 35% 长文档的跨段落概括标记为 fits；但单张 24 GB GPU 离线构建预算、每日约 2,000 份变更的局部重建摘要树、普通问题 10 秒首屏均标记为 partial。
- 工程推断：RAPTOR 更适合作为长文档离线增强视图，而不是替代基础增量检索层；每日频繁变更下需要评估局部重建成本和一致性。

### CRAG

- 论文事实：CRAG 是 plug-and-play 的 Corrective Retrieval Augmented Generation，使用检索评估器提升生成鲁棒性 [E-local-efc0de696312-p9-c1]。
- 论文事实：论文报告 Self-RAG 和 Self-CRAG 的生成性能随检索性能下降而下降，但 Self-CRAG 下降更小，表明对检索性能下降的鲁棒性更强 [E-local-efc0de696312-p9-c1][E-local-efc0de696312-p9-c0]。
- 论文事实：论文报告 CRAG 相比 RAG 在 PopQA 上提升 2.1% accuracy、Biography 上提升 2.8% FactScore（基于 LLaMA2-hf-7b）；在 SelfRAG-LLaMA2-7b 基础上，PopQA 提升 19.0% accuracy、Biography 提升 14.9% FactScore、PubHealth 提升 36.6% accuracy、ArcChallenge 提升 8.1% accuracy [E-local-efc0de696312-p6-c4]。这些是论文实验数字，不是本系统实测。
- 硬约束：系统确定性业务评分中 CRAG 硬约束为「不通过」，不得推荐为首选。题设人工整理指出 CRAG 在错误时回退公开网页搜索，与敏感资料不出域冲突；本地证据未覆盖该回退机制细节，证据不足。

### FLARE

- 论文事实：FLARE 迭代生成临时下一句，若包含低概率 token，则将其作为查询检索相关文档，并重新生成下一句，直到结束；该机制适用于任何现有 LM 的推理时，无需额外训练 [E-local-2425bd04de6e-p2-c5]。
- 论文事实：FLARE 在不同数据集上使用不同超参数，包括 θ、β、查询构造方式以及是否结合单次与多次检索 [E-local-2425bd04de6e-p16-c1]。
- 题设人工整理（非实验实测）：FLARE 对长答案与多跳生成标记为 fits；但生成中多次触发检索会增加复杂问题延迟，且默认实现偏外部搜索，使用本地检索器时可满足隐私。
- 工程推断：FLARE 可作为复杂问题的可选生成策略，但必须强制使用本地检索器并控制发送片段数量，否则与内部安全边界冲突 [E-local-f4c9fff07eb3-p1-c0]。

### GraphRAG

- 论文事实：传统 RAG 返回语义相似固定记录，GraphRAG 对比 vector RAG 的能力在于回答需要跨整个数据语料进行全局 sensemaking 的查询 [E-local-730f1a9f38d1-p2-c6]。
- 论文事实：论文报告在使用 GPT-4 作为 LLM 时，GraphRAG 在全局 sensemaking 问题上强于 vector RAG；GraphRAG 已开源，并有 LangChain、LlamaIndex、NebulaGraph、Neo4J 等扩展 [E-local-730f1a9f38d1-p2-c4]。
- 论文事实：GraphRAG 的图索引、富文本注释和层次社区结构支持当前方法，并为后续改进提供可能 [E-local-730f1a9f38d1-p12-c0]。
- 内部事实冲突：团队没有大规模图数据库生产经验 [E-local-f4c9fff07eb3-p1-c0]；4 人小组 8 周内应避免复杂分布式调度系统 [E-local-f4c9fff07eb3-p1-c0]；每日约 2,000 份变更下，实体图与社区摘要的增量更新复杂。工程推断：GraphRAG 不适合作为首期首选。

## 加权比较矩阵

以下直接引用题设的系统确定性业务评分，只解释，不重新计算或改写分数。

| 候选方案 | 检索质量与可追溯性 | 在线延迟 | 数据更新复杂度 | 实施成本 | 运行成本 | 隐私与部署适配 | 运维与团队适配 | 加权总分/5 | 硬约束 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 基础混合检索 RAG | 3.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 4.5 | 通过 |
| RAPTOR | 5.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.5 | 通过 |
| CRAG | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 1.0 | 3.0 | 2.8 | 不通过 |
| FLARE | 5.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.5 | 通过 |
| GraphRAG | 3.0 | 3.0 | 1.0 | 1.0 | 3.0 | 3.0 | 1.0 | 2.2 | 通过 |

> 工程推断：评分按用户权重确定性计算；1/3/5 分分别表示冲突、部分契合、契合。未映射项取中性 3 分，不冒充线上实验结果；硬约束失败的方案不能成为首选。规则计算首选为基础混合检索 RAG，最终仍需人工确认。

## 关键证据

### 内部事实

- Atlas 初期约 180,000 份文档，35% 为超过 30 页长文档，每日新增或变更约 2,000 份，文档为内部敏感资料，中文约 70%、英文约 30%，回答必须展示来源文档和页码，找不到依据必须拒答，普通问题 10 秒以内、复杂问题 60 秒内 [E-local-bcba105fe48f-p1-c0]。
- 原始内部文档和完整索引不得上传第三方；允许必要检索片段发送给已签约大模型 API，但必须限制片段数量并记录证据标识；日志不得保存片段全文和 API Key [E-local-f4c9fff07eb3-p1-c0]。
- 现有环境为 Docker、16 核 CPU、64 GB 内存、单张 24 GB 显存 GPU；已有 PostgreSQL 和对象存储运维经验，但没有大规模图数据库生产经验；4 人小组 8 周交付，优先增量更新、可观测、可回滚，避免复杂分布式调度 [E-local-f4c9fff07eb3-p1-c0]。

### 论文事实

- 传统 RAG 检索固定数量语义相似记录，生成答案只使用这些记录；GraphRAG 面向需要整个语料全局理解的查询 [E-local-730f1a9f38d1-p2-c6]。
- GraphRAG 在 GPT-4 下对全局 sensemaking 问题强于 vector RAG，且已开源并提供多种扩展 [E-local-730f1a9f38d1-p2-c4]。
- GraphRAG 的图索引、富文本注释和层次社区结构支持当前方法 [E-local-730f1a9f38d1-p12-c0]。
- RAPTOR 的检索节点中，18.5% 到 57% 来自非叶节点，说明多层级树结构对检索有贡献 [E-local-01319e2e6e6e-p22-c2]。
- RAPTOR 在 NarrativeQA 上带 RAPTOR 的多种检索器优于对应基线，论文还报告 QuALITY 准确率 62.4% [E-local-01319e2e6e6e-p7-c2]；这些是论文实验，不是本系统实测。
- CRAG 使用检索评估器提升鲁棒性；检索性能下降时 Self-CRAG 比 Self-RAG 下降更小 [E-local-efc0de696312-p9-c1][E-local-efc0de696312-p9-c0]。
- 论文报告 CRAG 在 PopQA、Biography、PubHealth、ArcChallenge 等数据集上有提升 [E-local-efc0de696312-p6-c4]；这些是论文实验，不是本系统实测。
- FLARE 迭代生成临时下一句，低概率 token 时作为查询检索，再生成直到结束，适用于任何现有 LM 推理时且无需额外训练 [E-local-2425bd04de6e-p2-c5]。
- FLARE 在不同数据集使用不同超参数和检索组合策略 [E-local-2425bd04de6e-p16-c1]。

### 工程推断与证据不足

- 工程推断：系统确定性业务评分是权威业务排序；基础混合检索 RAG 加权总分最高且硬约束通过，CRAG 硬约束不通过，硬约束失败方案不能推荐为首选。
- 证据不足：本地证据未覆盖 CRAG 错误回退公开网页搜索的具体机制、Qdrant/FTS 增量实现、中文检索质量、本系统延迟与成本、24 GB GPU 离线摘要树构建预算；这些必须通过 POC 或补充材料确认。

## 工程实施影响

- 隐私与部署：必须本地保存原始文档和完整索引，仅发送必要检索片段；每次请求限制片段数量并记录证据标识，日志不得保存片段全文和 API Key [E-local-f4c9fff07eb3-p1-c0]。工程推断：基础混合检索 RAG 更容易控制发送片段边界。
- 数据更新：每日约 2,000 份变更，要求增量更新、可观测、可回滚，避免复杂分布式调度 [E-local-f4c9fff07eb3-p1-c0]。工程推断：RAPTOR 摘要树和 GraphRAG 社区摘要的增量维护风险更高。
- 延迟：普通问题 10 秒首屏、复杂问题 60 秒 [E-local-bcba105fe48f-p1-c0]。工程推断：CRAG、FLARE 等引入在线评估或多次检索会增加延迟；需实测。
- 团队与交付：4 人小组 8 周内交付，已有 PostgreSQL 和对象存储经验，但没有大规模图数据库生产经验 [E-local-f4c9fff07eb3-p1-c0]。工程推断：GraphRAG 首期落地风险高。
- 可追溯性：回答必须展示来源文档和页码，找不到依据必须拒答 [E-local-bcba105fe48f-p1-c0]。工程推断：基础混合检索 RAG 更容易在检索层保留来源元数据；长文档摘要树需额外维护原文映射。
- 长文档：35% 文档超过 30 页 [E-local-bcba105fe48f-p1-c0]。论文事实：RAPTOR 多层级树对检索有贡献 [E-local-01319e2e6e6e-p22-c2]，并在 NarrativeQA 等数据集上优于基线 [E-local-01319e2e6e6e-p7-c2]；但论文结果不能直接外推本系统。

## 风险与未知项

- CRAG 硬约束不通过，且题设人工整理指出错误时回退公开网页搜索，与敏感资料不出域冲突；本地证据未覆盖该机制，证据不足。
- GraphRAG 在全局问题上论文表现强 [E-local-730f1a9f38d1-p2-c6]，但团队无大规模图数据库生产经验，8 周交付和每日约 2,000 份变更的增量更新复杂 [E-local-f4c9fff07eb3-p1-c0]。
- RAPTOR 与 FLARE 的论文实验数字不能写成“本系统实测”；RAPTOR 的离线构建、局部重建和 FLARE 的多次检索延迟需要 POC 验证 [E-local-01319e2e6e6e-p7-c2][E-local-2425bd04de6e-p2-c5]。
- 本地证据未提供中文 70%、英文 30% 语料下的检索质量、页码级可追溯性和拒答质量；证据不足。
- 单张 24 GB GPU 是否足够支撑离线嵌入、RAPTOR 摘要树或 GraphRAG 图谱抽取，本地证据未提供实测；证据不足。
- 运行成本、API 限流、片段大小、证据标识存储方案未提供实测或约束细节；证据不足。

## 推荐草案

> 以下为草案，必须等待人工确认；不替代系统确定性业务评分，也不把论文榜单当作业务排序。

- 首选：基础混合检索 RAG。系统确定性业务评分中，其加权总分为 4.5 且硬约束通过；结合内部安全约束，原始内部文档和完整索引不得上传第三方，只允许必要检索片段发送给已签约大模型 API，且必须限制片段数量并记录证据标识 [E-local-f4c9fff07eb3-p1-c0]。其组件相对少，更符合 4 人小组 8 周交付、优先增量更新、可观测、可回滚、避免复杂分布式调度的要求 [E-local-f4c9fff07eb3-p1-c0]。同时它更容易控制普通问题 10 秒首屏所需的在线额外调用 [E-local-bcba105fe48f-p1-c0]。
- 备选增强：RAPTOR 仅作为 35% 长文档的离线摘要树或二级检索视图进行 POC。论文事实显示 RAPTOR 多层级树中相当比例检索节点来自非叶节点 [E-local-01319e2e6e6e-p22-c2]，并在 NarrativeQA 上优于对应基线 [E-local-01319e2e6e6e-p7-c2]；但这些是论文实验，不是本系统实测。每日约 2,000 份变更下，摘要树局部重建和一致性需验证 [E-local-f4c9fff07eb3-p1-c0]。
- 复杂问题可选：FLARE 风格主动检索。论文事实显示 FLARE 在生成过程中根据低概率 token 触发检索并重新生成 [E-local-2425bd04de6e-p2-c5]；若采用，必须使用本地检索器并遵守片段最小化与证据标识要求 [E-local-f4c9fff07eb3-p1-c0]。其多次检索可能增加延迟，需在 60 秒复杂问题预算内验证 [E-local-bcba105fe48f-p1-c0]。
- 不推荐 CRAG 为首选：系统确定性业务评分硬约束不通过；题设人工整理指出其错误回退公开网页搜索与敏感资料不出域冲突，本地证据未覆盖该机制，证据不足。
- 不推荐 GraphRAG 为首选：论文事实显示其适合全局 sensemaking [E-local-730f1a9f38d1-p2-c6]，但团队没有大规模图数据库生产经验，且 4 人 8 周内应避免复杂分布式调度系统 [E-local-f4c9fff07eb3-p1-c0]；每日增量更新复杂。
- 结论：推荐草案为首选基础混合检索 RAG，RAPTOR/FLARE 作为可选增强进入 POC；最终方案等待人工确认。

## POC 计划

- 第 1 周：确认安全与合规边界，定义片段数量上限、证据标识格式、日志脱敏规则；原始文档和完整索引不出域 [E-local-f4c9fff07eb3-p1-c0]。
- 第 2-3 周：落地基础混合检索 RAG MVP，验证来源文档和页码展示、找不到依据拒答、普通问题 10 秒首屏 [E-local-bcba105fe48f-p1-c0]。
- 第 4-5 周：验证每日约 2,000 份变更的增量索引、幂等、回滚和可观测性，优先采用可增量更新、可观测、可回滚组件 [E-local-f4c9fff07eb3-p1-c0]。
- 第 6-7 周：对 35% 长文档试点 RAPTOR 离线摘要树或二级检索视图，比较跨段落概括改善；注意论文 RAPTOR 数字不是本系统实测 [E-local-01319e2e6e6e-p7-c2][E-local-01319e2e6e6e-p22-c2]。
- 第 8 周：复杂问题 60 秒预算压力测试、安全评审、人工确认。复杂跨文档问题目标为 60 秒内生成研究型答案 [E-local-bcba105fe48f-p1-c0]。
- 成功标准：硬约束通过；隐私片段最小化；来源可追溯；找不到依据可拒答；增量更新可回滚；普通问题 10 秒、复杂问题 60 秒目标达成。具体指标阈值仍需补充，证据不足。

## 参考资料

- E-local-730f1a9f38d1-p2-c6
- E-local-730f1a9f38d1-p12-c0
- E-local-730f1a9f38d1-p2-c4
- E-local-bcba105fe48f-p1-c0
- E-local-f4c9fff07eb3-p1-c0
- E-local-01319e2e6e6e-p7-c2
- E-local-01319e2e6e6e-p22-c2
- E-local-efc0de696312-p6-c4
- E-local-efc0de696312-p9-c0
- E-local-efc0de696312-p9-c1
- E-local-2425bd04de6e-p2-c5
- E-local-2425bd04de6e-p16-c1

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
