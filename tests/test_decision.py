from research.decision import build_decision_matrix, render_decision_matrix


WEIGHTS = {
    "检索质量与可追溯性": 25,
    "在线延迟": 15,
    "数据更新复杂度": 15,
    "实施成本": 15,
    "运行成本": 10,
    "隐私与部署适配": 10,
    "运维与团队适配": 10,
}


def test_business_weights_produce_auditable_totals_and_hard_constraints():
    task = {
        "question": "为敏感、每日更新的长文档知识库选择方案",
        "requirements": {
            "隐私要求": "内部敏感资料，只允许发送必要片段",
            "更新频率": "每日",
            "团队规模": "4 人，8 周交付",
            "基础设施": "单张 24GB GPU",
        },
        "candidates": ["基础混合检索 RAG", "CRAG", "GraphRAG"],
        "weights": WEIGHTS,
    }

    matrix = build_decision_matrix(task)
    rows = {row["candidate"]: row for row in matrix["rows"]}

    assert rows["基础混合检索 RAG"]["weighted_total"] > rows["GraphRAG"]["weighted_total"]
    assert rows["CRAG"]["hard_constraint_pass"] is False
    assert rows["CRAG"]["hard_conflicts"] == ["req_privacy"]
    assert matrix["recommended_candidate"] == "基础混合检索 RAG"
    assert "加权总分/5" in render_decision_matrix(matrix)


def test_unknown_candidate_and_criterion_use_explicit_neutral_score():
    task = {
        "question": "选择方案",
        "requirements": {},
        "candidates": ["方案 A", "方案 B"],
        "weights": {"自定义指标": 100},
    }

    matrix = build_decision_matrix(task)

    assert all(row["weighted_total"] == 3.0 for row in matrix["rows"])
    assert all(
        "人工复核" in row["rationale"]["自定义指标"]
        for row in matrix["rows"]
    )
