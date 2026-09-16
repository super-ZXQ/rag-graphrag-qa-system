"""Deterministic business scoring for the technology-selection workflow."""
from __future__ import annotations

from graph.domain import FITS, REQUIREMENTS, TECHNIQUES


STANCE_SCORE = {"fits": 5, "partial": 3, "conflict": 1}

CRITERION_REQUIREMENTS = {
    "检索质量与可追溯性": ("req_long_docs", "req_global_questions"),
    "在线延迟": ("req_low_latency",),
    "数据更新复杂度": ("req_daily_updates",),
    "实施成本": ("req_small_team", "req_no_graph_exp"),
    "运行成本": ("req_single_gpu", "req_low_latency"),
    "隐私与部署适配": ("req_privacy",),
    "运维与团队适配": ("req_small_team", "req_no_graph_exp", "req_daily_updates"),
}

REQUIREMENT_MARKERS = {
    "req_privacy": ("敏感", "隐私", "内部资料", "必要片段", "不出域"),
    "req_daily_updates": ("每日", "每天", "日更", "daily"),
    "req_long_docs": ("长文档", "30 页", "30页", "跨段落"),
    "req_low_latency": ("延迟", "首屏", "秒以内", "10 秒", "10秒"),
    "req_small_team": ("4 人", "4人", "2–5 人", "2-5 人", "8 周", "8周", "小组"),
    "req_no_graph_exp": ("没有图数据库", "无图数据库", "缺少图数据库", "图数据库经验"),
    "req_single_gpu": ("24gb", "24 gb", "单张", "单卡", "预算有限"),
    "req_global_questions": ("全局问题", "整个语料", "跨整个语料"),
}


def _technique_key(candidate: str) -> str | None:
    normalized = candidate.strip().lower()
    for key, info in TECHNIQUES.items():
        if normalized in {key.lower(), info["name"].lower()}:
            return key
    return None


def detect_active_requirements(question: str, requirements: dict) -> list[str]:
    text = " ".join([question, *[f"{key} {value}" for key, value in requirements.items()]]).lower()
    return [
        requirement_id
        for requirement_id, markers in REQUIREMENT_MARKERS.items()
        if any(marker.lower() in text for marker in markers)
    ]


def build_decision_matrix(task: dict) -> dict:
    """Return an auditable 1–5 engineering score matrix and weighted totals."""
    active = detect_active_requirements(task["question"], task["requirements"])
    rows = []
    for candidate in task["candidates"]:
        technique_key = _technique_key(candidate)
        fits = FITS.get(technique_key or "", {})
        criterion_scores = {}
        rationale = {}
        conflicts = []

        for criterion, weight in task["weights"].items():
            relevant = [req for req in CRITERION_REQUIREMENTS.get(criterion, ()) if req in active]
            assessments = [(req, fits[req]) for req in relevant if req in fits]
            if assessments:
                scores = [STANCE_SCORE[stance] for _, (stance, _) in assessments]
                score = round(sum(scores) / len(scores), 2)
                rationale[criterion] = "；".join(reason for _, (_, reason) in assessments)
                conflicts.extend(req for req, (stance, _) in assessments if stance == "conflict")
            else:
                score = 3.0
                rationale[criterion] = "当前约束与人工规则库没有直接映射，采用中性分并要求人工复核。"
            criterion_scores[criterion] = {"score": score, "weight": weight}

        weighted_total = round(
            sum(item["score"] * item["weight"] for item in criterion_scores.values()) / 100,
            2,
        )
        # 明确的敏感数据边界属于硬约束；其他 conflict 保留为高风险权衡。
        hard_conflicts = sorted(set(conflicts) & {"req_privacy"})
        rows.append(
            {
                "candidate": candidate,
                "technique_key": technique_key,
                "criterion_scores": criterion_scores,
                "weighted_total": weighted_total,
                "hard_constraint_pass": not hard_conflicts,
                "hard_conflicts": hard_conflicts,
                "conflicts": sorted(set(conflicts)),
                "rationale": rationale,
            }
        )

    eligible = [row for row in rows if row["hard_constraint_pass"]]
    ranked = sorted(eligible or rows, key=lambda row: row["weighted_total"], reverse=True)
    return {
        "scale": "1–5 engineering prior; not an online benchmark",
        "active_requirements": [
            {"id": requirement_id, "name": REQUIREMENTS[requirement_id]}
            for requirement_id in active
        ],
        "rows": rows,
        "recommended_candidate": ranked[0]["candidate"] if ranked else None,
    }


def render_decision_matrix(matrix: dict) -> str:
    if not matrix["rows"]:
        return ""
    criteria = list(matrix["rows"][0]["criterion_scores"])
    header = "| 候选方案 | " + " | ".join(criteria) + " | 加权总分/5 | 硬约束 |"
    separator = "|---|" + "---:|" * (len(criteria) + 1) + "---|"
    lines = ["## 系统确定性业务评分", "", header, separator]
    for row in matrix["rows"]:
        scores = [str(row["criterion_scores"][criterion]["score"]) for criterion in criteria]
        hard = "通过" if row["hard_constraint_pass"] else "不通过"
        lines.append(
            f"| {row['candidate']} | " + " | ".join(scores) +
            f" | {row['weighted_total']} | {hard} |"
        )
    lines.extend(
        [
            "",
            "> 工程推断：评分按用户权重确定性计算；1/3/5 分分别表示冲突、部分契合、契合。",
            "> 工程推断：未映射项取中性 3 分，不冒充线上实验结果；硬约束失败的方案不能成为首选。",
            f"> 工程推断：规则计算首选为 {matrix['recommended_candidate'] or '无'}，最终仍需人工确认。",
        ]
    )
    return "\n".join(lines)
