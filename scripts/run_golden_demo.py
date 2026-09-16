"""Run the portfolio's golden technology-selection task against indexed demo data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_library.store import PaperStore
from research.claims import evaluate_claims
from research.showcase import build_validated_showcase
from research.workflow import ResearchWorkflow

WEIGHTS = {
    "检索质量与可追溯性": 25,
    "在线延迟": 15,
    "数据更新复杂度": 15,
    "实施成本": 15,
    "运行成本": 10,
    "隐私与部署适配": 10,
    "运维与团队适配": 10,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation/golden_report.md"))
    args = parser.parse_args()
    store = PaperStore()
    if store.active_task_exists():
        print("an active research task already exists", file=sys.stderr)
        return 2
    task = store.create_task(
        {
            "task_id": str(uuid4()),
            "question": "为包含敏感长文档、每日更新的企业知识库选择可落地的 RAG 技术路线",
            "requirements": {
                "语料规模": "18 万份文档，35% 为长文档",
                "隐私要求": "内部敏感资料，只允许发送必要检索片段",
                "更新频率": "每日约 2,000 份文档变更",
                "延迟目标": "普通问题 10 秒，复杂问题 60 秒",
                "团队规模": "4 人，8 周交付",
                "基础设施": "单张 24GB GPU，可运行 Docker",
            },
            "candidates": ["基础混合检索 RAG", "RAPTOR", "CRAG", "FLARE", "GraphRAG"],
            "weights": WEIGHTS,
        }
    )
    report = ResearchWorkflow(store=store).run(task["task_id"])
    if not report["validation"]["valid"]:
        rejected_path = args.output.with_name("rejected_report.md")
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.write_text(report["markdown"], encoding="utf-8")
        evidence = store.list_evidence(task["task_id"])
        decision_matrix = report["validation"]["decision_matrix"]
        showcase = build_validated_showcase(task, evidence, decision_matrix)
        # 与主工作流一致：模型正文/确定性正文先过 claim gate，附录不冒充证据。
        core = showcase.split("## 系统确定性业务评分", 1)[0].rstrip()
        validation = evaluate_claims(core, evidence, store=store, use_llm=False)
        validation["decision_matrix"] = decision_matrix
        if validation["valid"]:
            report = store.save_report(
                {
                    "report_id": str(uuid4()),
                    "task_id": task["task_id"],
                    "markdown": showcase,
                    "validation": validation,
                },
                evidence_ids=[item["evidence_id"] for item in evidence],
                reviewer_note="模型草案被门槛拒绝；生成仅含已验证事实的 fail-closed 展示版。",
                claims=validation["claims"],
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report["markdown"], encoding="utf-8")
    print(f"task_id={task['task_id']}")
    print(f"validation={report['validation']}")
    print(f"report={args.output.resolve()}")
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
