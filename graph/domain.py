"""确定性的技术/需求领域映射（人工整理，非 LLM 生成）。

- 技术与黄金论文 arXiv ID 绑定；
- FITS 关系表示“技术特性与企业约束的契合度”，stance 取 fits/partial/conflict，
  依据来自论文记载的机制与内部约束文档，rationale 保持简短可核查；
- 图谱查询只使用固定模板，不允许模型自由生成 Cypher。
"""
from __future__ import annotations

TECHNIQUES = {
    "hybrid_rag": {"name": "基础混合检索 RAG", "arxiv_id": None},
    # title_hint 用于 OpenAlex 标题校验，必须是论文真实标题片段，避免同名缩写误匹配
    "raptor": {
        "name": "RAPTOR",
        "arxiv_id": "2401.18059",
        "title_hint": "RAPTOR: Recursive Abstractive",
    },
    "crag": {
        "name": "CRAG",
        "arxiv_id": "2401.15884",
        "title_hint": "Corrective Retrieval",
    },
    "flare": {
        "name": "FLARE",
        "arxiv_id": "2305.06983",
        "title_hint": "Active Retrieval",
    },
    "graphrag": {
        "name": "GraphRAG",
        "arxiv_id": "2404.16130",
        "title_hint": "From Local to Global",
    },
}

# filename -> technique key（与导入的黄金论文文件名对应）
ARXIV_FILENAME_TO_TECHNIQUE = {
    "2401.18059v1.pdf": "raptor",
    "2401.15884v1.pdf": "crag",
    "2305.06983v2.pdf": "flare",
    "2404.16130v2.pdf": "graphrag",
}

REQUIREMENTS = {
    "req_privacy": "敏感资料仅发送必要检索片段，不依赖外部公开网页",
    "req_daily_updates": "每日约 2000 份变更，需要增量索引",
    "req_long_docs": "35% 为超过 30 页的长文档，需要跨段落概括",
    "req_low_latency": "普通问题 10 秒首屏，控制在线额外调用",
    "req_small_team": "4 人小组 8 周交付，避免重型新基础设施",
    "req_no_graph_exp": "团队没有大规模图数据库生产经验",
    "req_single_gpu": "单张 24GB GPU，离线构建预算有限",
    "req_global_questions": "需要回答跨整个语料库的全局性问题",
}

# technique -> requirement -> (stance, rationale)
FITS = {
    "hybrid_rag": {
        "req_privacy": ("fits", "检索与向量均在本地，仅发送命中片段"),
        "req_daily_updates": ("fits", "Qdrant upsert + FTS 支持幂等增量更新"),
        "req_low_latency": ("fits", "单次混合检索，在线额外开销小"),
        "req_small_team": ("fits", "组件少，团队可在 8 周内落地"),
        "req_single_gpu": ("fits", "仅需本地 embedding，无重型离线构建"),
        "req_long_docs": ("partial", "分块检索对长文档全局概括支持有限"),
    },
    "raptor": {
        "req_long_docs": ("fits", "递归摘要树提供多尺度上下文，利于长文档概括"),
        "req_single_gpu": ("partial", "树摘要离线构建有一次性算力成本"),
        "req_daily_updates": ("partial", "文档频繁变更时需要局部重建摘要树"),
        "req_low_latency": ("partial", "树可离线预构建，在线检索开销可控"),
    },
    "crag": {
        "req_daily_updates": ("partial", "检索纠错能缓解索引陈旧，但不替代增量索引"),
        "req_low_latency": ("partial", "评估器与精炼带来额外在线延迟"),
        "req_privacy": ("conflict", "错误时回退公开网页搜索，与敏感资料不出域约束冲突"),
    },
    "flare": {
        "req_long_docs": ("fits", "前瞻式主动检索适合长答案与多跳生成"),
        "req_low_latency": ("partial", "生成中多次触发检索，复杂问题延迟增加"),
        "req_privacy": ("partial", "使用本地检索器时可满足隐私，默认实现偏外部搜索"),
    },
    "graphrag": {
        "req_global_questions": ("fits", "社区摘要专为面向整个语料库的全局问题设计"),
        "req_no_graph_exp": ("conflict", "依赖图数据库与社区构建，团队缺少生产经验"),
        "req_small_team": ("conflict", "图谱构建与调优难以在 8 周内由 4 人小组稳定交付"),
        "req_daily_updates": ("conflict", "实体图与社区摘要的增量更新复杂"),
        "req_single_gpu": ("partial", "全量图谱抽取需要较多离线 LLM 调用与算力"),
    },
}
