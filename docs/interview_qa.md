# 面试问答

## 为什么不用多 Agent？

V1 的业务是「有证据的技术选型报告」，不是开放对话。受控单 Agent 固定节点（需求归一 → 检索 → 比较 → 起草 → 校验 → 人工审批）更容易保证可取消、可恢复、可审计。多 Agent 会放大协调成本与幻觉面，对实习/作品集阶段收益有限。

## 为什么需要人工审批？

模型只能基于检索片段生成草案。论文实验数字、内部约束、工程成本三者语义不同，自动定案风险高。系统把「引用存在率、claim 支持度、证据不足」作为硬门槛，人工是最后一道责任边界，不是摆设。

## 如何保证不泄露内部资料？

1. 内部文档只在本地解析与索引。
2. 生成时只发送命中的必要片段（上限 `AGENT_MAX_CONTEXT_CHUNKS`），不发整篇。
3. 日志不记录全文/Prompt/Key，只记 task_id、步骤、证据 ID、耗时、Token。
4. 报告引用稳定证据 ID，可定位到 paper/chunk/page。

## 为什么有 GraphRAG 但不默认推荐 GraphRAG？

GraphRAG 论文面向全局 sensemaking；本场景约束是敏感资料、日更、4 人 8 周、无图数据库经验、单卡 24GB。领域映射里 GraphRAG 与「小团队交付 / 日更 / 无图经验」冲突。系统会展示其能力证据，但推荐必须过企业约束，而不是论文榜单。

## 如何评估检索质量？

40 条标注查询（内部约束/论文事实/跨文档/无答案/冲突），对比 dense / FTS5 / hybrid RRF / hybrid+graph。文档级 Recall 衡量是否找到正确来源，精确证据 Recall 衡量是否命中标注 chunk；跨文档题要求找齐全部来源。结果写入 `evaluation/results.json`，不把论文实验当系统实验。

## 如何避免模型编造引用？

- Prompt 要求原样复制可用证据 ID。
- 硬校验：ID 必须存在于证据库并能定位 paper/chunk/page；不存在 → `fabricated_citation`，审批禁止。
- 软校验：词项重叠 + 可选 LLM；不确定降为 partially_supported / insufficient，LLM 不能把硬伤翻案为 supported。

## 如何处理检索文档里的提示注入？

论文附录本身可能包含 “Return output as JSON” 等 Prompt 模板。系统不会只靠一句“这是数据”防护：命中确定性危险模式的 chunk 在进入模型上下文前被过滤并计数，证据使用独立标签包裹；模型若仍输出代码围栏或无引用事实，claim gate 会拒绝审批。仓库同时保留 rejected 与 fail-closed 报告用于演示。

## 为什么 CRAG/FLARE/RAPTOR 不直接全量上线？

各方案有明确代价：CRAG 回退公开搜索与隐私冲突、FLARE 在线多次检索增加延迟、RAPTOR 日更下树维护成本、GraphRAG 构建与团队经验门槛。产品结论是组合落地：混合检索做底座，按约束选择性叠加，而不是「谁论文分高就上谁」。

## Docker 不可用时如何开发？

`QDRANT_PATH=data/qdrant_local` 走 Qdrant 本地持久化；API/UI 本机 uvicorn/streamlit。Docker 恢复后 `docker compose up --build` 验证完整链路。评测与黄金报告在本地模式已真实跑通。

## 如何做任务失败恢复？

节点级重试（`WORKFLOW_MAX_RETRIES`）+ 失败持久化 `failed_step`/`error_type`。`resume` 复用已冻结证据从失败节点继续；`revise` 基于待审批报告出新版本并保留历史；取消在节点边界生效且不删证据。

## hybrid+graph 为什么没有提升 Recall？

如实回答：这里实现的是引用图弱重排和引用网络解释，不是完整 GraphRAG 社区检索。在当前语料上它没有抬高 Recall，精确证据 R@5 还略低于普通 hybrid。只有元数据的邻居不会被算作独立证据，避免为了“有图”夸大价值。
