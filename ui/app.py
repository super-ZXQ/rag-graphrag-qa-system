"""Streamlit client for the evidence-backed technology selection workflow."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_CANDIDATES = ["基础混合检索 RAG", "RAPTOR", "CRAG", "FLARE", "GraphRAG"]
DEFAULT_WEIGHTS = {
    "检索质量与可追溯性": 25,
    "在线延迟": 15,
    "数据更新复杂度": 15,
    "实施成本": 15,
    "运行成本": 10,
    "隐私与部署适配": 10,
    "运维与团队适配": 10,
}
GOLDEN_REQUIREMENTS = {
    "语料规模": "18 万份文档，35% 为长文档",
    "隐私要求": "内部敏感资料，只允许发送必要检索片段",
    "更新频率": "每日约 2,000 份文档变更",
    "延迟目标": "普通问题 10 秒，复杂问题 60 秒",
    "团队规模": "4 人，8 周交付",
    "基础设施": "单张 24GB GPU，可运行 Docker",
}
STATUS_META = {
    "parsed": ("已解析", "#64748b"),
    "indexing": ("索引中", "#0369a1"),
    "ready": ("可检索", "#15803d"),
    "index_failed": ("索引失败", "#b91c1c"),
    "queued": ("排队中", "#64748b"),
    "running": ("运行中", "#0369a1"),
    "awaiting_approval": ("待人工确认", "#b45309"),
    "completed": ("已确认", "#15803d"),
    "failed": ("失败", "#b91c1c"),
    "cancelled": ("已取消", "#64748b"),
    "superseded": ("已替代", "#64748b"),
}
STEP_LABELS = {
    "normalize_requirements": "整理需求",
    "plan_research": "规划检索",
    "retrieve_evidence": "检索证据",
    "build_comparison": "构建比较",
    "draft_report": "起草报告",
    "validate_claims": "校验引用",
    "await_human_approval": "等待审批",
}

st.set_page_config(page_title="ScholarGraph 技术选型 Agent", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.5rem;}
    .eyebrow {color:#2563eb; font-size:.8rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;}
    .hero {font-size:2.1rem; font-weight:760; letter-spacing:-.04em; margin:.25rem 0;}
    .muted {color:#64748b; max-width:780px; line-height:1.65;}
    .privacy {background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:.75rem 1rem; color:#334155; font-size:.9rem;}
    .fact-tag {display:inline-block; padding:.1rem .45rem; border-radius:6px; font-size:.75rem; font-weight:650; margin-right:.35rem;}
    .fact-paper {background:#dbeafe; color:#1d4ed8;}
    .fact-internal {background:#dcfce7; color:#166534;}
    .fact-infer {background:#fef3c7; color:#92400e;}
    .fact-unknown {background:#f1f5f9; color:#475569;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=90, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"API 不可达：{exc}") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text or f"HTTP {response.status_code}"
        raise RuntimeError(str(detail))
    if not response.content:
        return None
    return response.json()


def status_chip(status: str) -> str:
    label, _ = STATUS_META.get(status, (status or "—", "#64748b"))
    return label


def status_badge(status: str) -> None:
    label, color = STATUS_META.get(status, (status or "—", "#64748b"))
    st.markdown(
        f'<span style="display:inline-block;padding:.2rem .65rem;border-radius:999px;'
        f'background:{color}18;color:{color};font-weight:700;font-size:.85rem;">{label}</span>',
        unsafe_allow_html=True,
    )


def format_seconds(value) -> str:
    try:
        seconds = float(value or 0)
    except (TypeError, ValueError):
        return "—"
    if seconds >= 60:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds:.1f} 秒"


def paper_public_link(paper: dict) -> str | None:
    uri = paper.get("source_uri") or ""
    if paper.get("source_type") == "public_arxiv" and uri.startswith("http"):
        return uri
    return None


st.markdown('<div class="eyebrow">Enterprise R&D Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">技术选型，不只给答案，还给证据。</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">融合公开论文与企业内部约束，由受控 Agent 生成可追溯的技术选型报告。关键结论绑定原文证据，最终建议必须由人确认。</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 运行状态")
    api_ok = False
    try:
        health = api("GET", "/health/live")
        st.success(f"API 正常 · {health.get('model', '')}")
        api_ok = True
    except Exception as exc:
        st.error(f"API 不可用：{exc}")
    st.markdown(
        '<div class="privacy">隐私边界：本地解析与检索；只向生成模型发送命中的必要片段，不发送内部全文。</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("task_id"):
        st.divider()
        st.caption("当前研究任务")
        st.code(st.session_state.task_id, language=None)
        if st.button("清除任务选择", use_container_width=True):
            st.session_state.pop("task_id", None)
            st.rerun()

library_tab, task_tab, report_tab = st.tabs(["01 资料库", "02 创建选型任务", "03 研究过程与报告"])

# ───────────────────────── 01 资料库 ─────────────────────────
with library_tab:
    st.subheader("构建可信证据库")
    st.caption("内部资料与公开论文进入同一本地索引。内部全文不会展示在界面，也不会整篇发给外部模型。")

    upload_col, public_col = st.columns(2)
    with upload_col:
        st.markdown("#### 内部资料上传")
        uploaded = st.file_uploader("上传 PDF 或 Markdown", type=["pdf", "md", "markdown"])
        if uploaded and st.button("解析并索引", type="primary", use_container_width=True):
            try:
                result = api(
                    "POST",
                    "/papers",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
                paper = result.get("paper") or {}
                st.success(f"已接收：{paper.get('title', uploaded.name)} · 状态 {status_chip(paper.get('status', ''))}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with public_col:
        st.markdown("#### 公开论文（arXiv）")
        query = st.text_input("搜索 arXiv", placeholder="例如 corrective retrieval augmented generation")
        search_clicked = st.button("搜索", use_container_width=True, disabled=not query.strip() or not api_ok)
        if search_clicked:
            try:
                st.session_state.source_results = api(
                    "GET", "/sources/search", params={"q": query, "provider": "arxiv"}
                )
            except Exception as exc:
                st.session_state.source_results = []
                st.error(str(exc))

        results = st.session_state.get("source_results") or []
        if not results:
            st.caption("搜索后可先预览摘要，再决定是否导入。")
        for item in results:
            with st.container(border=True):
                st.markdown(f"**{item.get('title', '')}**")
                authors = ", ".join((item.get("authors") or [])[:3])
                st.caption(f"{item.get('external_id', '')} · {(item.get('published') or '')[:10]} · {authors}")
                summary = (item.get("summary") or "").strip()
                with st.expander("摘要预览", expanded=False):
                    st.write(summary[:1200] + ("…" if len(summary) > 1200 else "") if summary else "（无摘要）")
                if st.button("导入该论文", key=f"import-{item['external_id']}", use_container_width=True):
                    try:
                        api(
                            "POST",
                            "/sources/import",
                            json={"provider": "arxiv", "external_id": item["external_id"]},
                        )
                        st.success("已进入解析与索引队列")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    st.divider()
    st.markdown("#### 资料库清单")
    if st.button("刷新列表", use_container_width=False):
        st.rerun()
    try:
        papers = api("GET", "/papers") if api_ok else []
    except Exception as exc:
        papers = []
        st.warning(str(exc))

    if not papers:
        st.info("先导入内部资料或公开论文，再创建研究任务。")
    else:
        ready_n = sum(1 for p in papers if p.get("status") == "ready")
        failed_n = sum(1 for p in papers if p.get("status") == "index_failed")
        m1, m2, m3 = st.columns(3)
        m1.metric("资料总数", len(papers))
        m2.metric("可检索", ready_n)
        m3.metric("索引失败", failed_n)

        for paper in papers:
            with st.container(border=True):
                head = st.columns([4, 1])
                with head[0]:
                    st.markdown(f"**{paper.get('title') or paper.get('filename')}**")
                    link = paper_public_link(paper)
                    meta_bits = [
                        f"来源：{paper.get('source_type', '—')}",
                        f"可见性：{paper.get('visibility', '—')}",
                        f"页数：{paper.get('page_count') or '—'}",
                        f"索引版本：{paper.get('index_version') or '—'}",
                    ]
                    st.caption(" · ".join(meta_bits))
                    if link:
                        st.link_button("打开 arXiv 页面", link, use_container_width=False)
                    if paper.get("visibility") == "internal":
                        st.caption("内部资料：界面不展示全文，检索时仅发送命中片段。")
                with head[1]:
                    status_badge(paper.get("status") or "")

                if paper.get("status") == "index_failed":
                    st.error(paper.get("error_message") or "索引失败，原因未知。")
                    if st.button("重试索引", key=f"retry-{paper['paper_id']}"):
                        try:
                            api("POST", f"/papers/{paper['paper_id']}/retry")
                            st.success("已重新排队索引")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                if paper.get("error_message") and paper.get("status") != "index_failed":
                    st.warning(paper.get("error_message"))

# ───────────────────────── 02 创建任务 ─────────────────────────
with task_tab:
    st.subheader("定义业务问题与评价边界")
    st.markdown(
        """
        <div class="privacy">
        <b>黄金演示模板</b>：敏感长文档 + 每日更新 + 4 人 8 周交付的企业知识库 RAG 选型。<br>
        <b>隐私提示</b>：系统只把本地命中的必要片段发送给生成模型，不发送内部全文。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="margin:.75rem 0 .25rem;">
        <span class="fact-tag fact-paper">论文事实</span>可在公开论文中定位；
        <span class="fact-tag fact-internal">内部事实</span>来自企业约束文档；
        <span class="fact-tag fact-infer">工程推断</span>是综合权衡，需标注；
        <span class="fact-tag fact-unknown">证据不足</span>时系统拒绝编造推荐。
        </div>
        """,
        unsafe_allow_html=True,
    )

    use_golden = st.toggle("使用黄金场景模板", value=True)
    with st.form("research-task"):
        if use_golden:
            question = st.text_area(
                "选型问题",
                value="为包含敏感长文档、每日更新的企业知识库选择可落地的 RAG 技术路线",
                height=90,
            )
            candidates = st.multiselect("候选方案（至少 2 个）", DEFAULT_CANDIDATES, default=DEFAULT_CANDIDATES)
            req_cols = st.columns(2)
            requirements = {}
            for index, (key, value) in enumerate(GOLDEN_REQUIREMENTS.items()):
                with req_cols[index % 2]:
                    requirements[key] = st.text_input(key, value=value)
        else:
            question = st.text_area("选型问题", value="", height=90, placeholder="用一句话描述业务目标")
            candidates = st.multiselect("候选方案（至少 2 个）", DEFAULT_CANDIDATES, default=DEFAULT_CANDIDATES[:3])
            c1, c2, c3 = st.columns(3)
            with c1:
                corpus_scale = st.selectbox("语料规模", ["小于 10 万文档", "10–100 万文档", "大于 100 万文档"])
                privacy = st.selectbox("隐私要求", ["内部敏感资料", "公开资料", "允许全量云端处理"])
            with c2:
                freshness = st.selectbox("更新频率", ["每日", "每周", "低频"])
                latency = st.selectbox("在线延迟目标", ["3 秒以内", "10 秒以内", "可接受分钟级"])
            with c3:
                team = st.selectbox("实施团队", ["2–5 人", "6–15 人", "15 人以上"])
                budget = st.selectbox("基础设施预算", ["有限", "中等", "充足"])
            requirements = {
                "语料规模": corpus_scale,
                "隐私要求": privacy,
                "更新频率": freshness,
                "延迟目标": latency,
                "团队规模": team,
                "预算": budget,
            }

        with st.expander("评分权重（合计必须 = 100）", expanded=True):
            edited_weights = {}
            weight_cols = st.columns(2)
            for index, (name, value) in enumerate(DEFAULT_WEIGHTS.items()):
                with weight_cols[index % 2]:
                    edited_weights[name] = st.number_input(name, 0, 100, value, 5, key=f"w-{name}")
            weight_sum = sum(edited_weights.values())
            if weight_sum == 100:
                st.success(f"权重合计 {weight_sum}% · 校验通过")
            else:
                st.error(f"权重合计 {weight_sum}% · 必须等于 100")

        form_ok = bool(question.strip()) and len(candidates) >= 2 and weight_sum == 100
        if not question.strip():
            st.caption("请填写选型问题。")
        if len(candidates) < 2:
            st.caption("候选方案至少 2 个。")
        submitted = st.form_submit_button(
            "启动证据研究",
            type="primary",
            use_container_width=True,
            disabled=not form_ok,
        )

    if submitted:
        try:
            result = api(
                "POST",
                "/research-tasks",
                json={
                    "question": question.strip(),
                    "candidates": candidates,
                    "requirements": requirements,
                    "weights": edited_weights,
                },
            )
            st.session_state.task_id = result["task_id"]
            st.success("研究任务已创建。切换到「03 研究过程与报告」查看进度。")
        except Exception as exc:
            st.error(str(exc))

# ───────────────────────── 03 研究过程与报告 ─────────────────────────
with report_tab:
    st.subheader("研究过程、证据校验与人工确认")
    task_id = st.session_state.get("task_id")
    if not task_id:
        st.info("创建研究任务后，这里会展示步骤、预算、证据校验和报告草案。也可以手动粘贴已有 task_id 继续查看。")
        manual_id = st.text_input("已有任务 ID（可选）", placeholder="粘贴 task_id 以查看历史任务")
        if manual_id.strip() and st.button("加载该任务", use_container_width=True):
            st.session_state.task_id = manual_id.strip()
            st.rerun()
    else:
        action_cols = st.columns([1, 1, 2])
        with action_cols[0]:
            if st.button("刷新状态", use_container_width=True):
                st.rerun()
        try:
            task = api("GET", f"/research-tasks/{task_id}")
        except Exception as exc:
            st.error(str(exc))
            task = None

        if task:
            status_badge(task.get("status") or "")
            st.caption(f"当前节点：`{STEP_LABELS.get(task.get('current_step') or '', task.get('current_step') or '—')}`")

            budget = task.get("budget") or {}
            bcols = st.columns(5)
            bcols[0].metric("模型调用", budget.get("model_calls", 0))
            bcols[1].metric("检索命中", budget.get("retrieved_hits", 0))
            bcols[2].metric("发送片段", budget.get("sent_snippets", 0))
            total_seconds = sum((budget.get("step_durations_seconds") or {}).values())
            bcols[3].metric("累计耗时", format_seconds(total_seconds))
            bcols[4].metric("Token 估算", budget.get("input_tokens", 0) + budget.get("output_tokens", 0))
            if budget.get("model"):
                st.caption(f"生成模型：{budget['model']} · Token 来源：{budget.get('token_source', 'none')}")

            if task.get("status") in {"queued", "running"}:
                with action_cols[1]:
                    if st.button("取消任务", type="secondary", use_container_width=True):
                        try:
                            api("POST", f"/research-tasks/{task_id}/cancel")
                            st.warning("已请求取消，将在节点边界生效。")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                st.caption("任务运行中，每 3 秒自动刷新…")
                time.sleep(3)
                st.rerun()

            if task.get("status") == "failed":
                st.error(f"失败节点：{task.get('failed_step') or '—'} · {task.get('error_type') or ''}")
                if task.get("error_message"):
                    st.caption(task["error_message"])
                if st.button("从失败处恢复", type="primary"):
                    try:
                        api("POST", f"/research-tasks/{task_id}/resume")
                        st.success("已排队恢复")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if task.get("status") == "cancelled":
                st.info("任务已取消。已保存的证据与事件仍可查看。")

            st.markdown("#### 事件时间线")
            events = task.get("events") or []
            if not events:
                st.caption("暂无事件。")
            for event in events:
                etype = event.get("event_type") or ""
                icon = {"completed": "✓", "retry": "↻", "state_changed": "→", "failed": "!"}.get(etype, "·")
                ts = (event.get("created_at") or "")[11:19]
                st.caption(f"{ts} `{event.get('step')}` {icon} {event.get('summary')}")

            evidence = []
            try:
                evidence = api("GET", f"/research-tasks/{task_id}/evidence") or []
            except Exception:
                evidence = []
            if evidence:
                with st.expander(f"已冻结证据 · {len(evidence)} 条", expanded=False):
                    st.caption("仅展示 ID / 来源 / 页码，不展示内部全文。")
                    for item in evidence:
                        st.markdown(
                            f"- `{item.get('evidence_id')}` · {item.get('title') or item.get('filename')} · 第 {item.get('page_number') or '—'} 页"
                        )

            if task.get("status") in {"awaiting_approval", "completed"}:
                try:
                    report = api("GET", f"/research-tasks/{task_id}/report")
                except Exception as exc:
                    report = None
                    st.error(str(exc))

                if report:
                    validation = report.get("validation") or {}
                    st.markdown("#### 引用质量")
                    qcols = st.columns(4)
                    qcols[0].metric("可用证据", validation.get("available_evidence", 0))
                    qcols[1].metric("已引用", validation.get("cited_evidence", 0))
                    rate = validation.get("citation_existence_rate")
                    qcols[2].metric("引用存在率", f"{rate:.0%}" if isinstance(rate, (int, float)) else "—")
                    coverage = validation.get("claim_evidence_coverage")
                    qcols[3].metric("关键结论覆盖率", f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "—")

                    support = validation.get("support_counts") or {}
                    if support:
                        sc = st.columns(4)
                        sc[0].metric("支持", support.get("supported", 0))
                        sc[1].metric("部分支持", support.get("partially_supported", 0))
                        sc[2].metric("冲突", support.get("conflicted", 0))
                        sc[3].metric("证据不足", support.get("insufficient", 0))

                    if validation.get("fabricated_citations"):
                        st.error("检测到虚构引用：" + ", ".join(validation["fabricated_citations"][:8]))
                    if validation.get("hard_errors"):
                        with st.expander("硬校验错误", expanded=False):
                            for err in validation["hard_errors"][:12]:
                                st.caption(f"{err.get('code')} · {err.get('message')}")

                    claims = report.get("claims") or validation.get("claims") or []
                    if claims:
                        st.markdown("#### 关键结论与证据")
                        evidence_index = {item.get("evidence_id"): item for item in evidence}
                        for claim in claims:
                            status = claim.get("support_status") or "insufficient"
                            status_label = {
                                "supported": "支持",
                                "partially_supported": "部分支持",
                                "conflicted": "冲突",
                                "insufficient": "证据不足",
                            }.get(status, status)
                            with st.container(border=True):
                                st.markdown(
                                    f"**{claim.get('claim_id')}** · `{claim.get('claim_type')}` · **{status_label}**"
                                    f"　置信度 {claim.get('confidence', 0)}"
                                )
                                st.write(claim.get("claim_text") or "")
                                if claim.get("risk_note"):
                                    st.caption(f"风险：{claim['risk_note']}")
                                for eid in claim.get("evidence_ids") or []:
                                    item = evidence_index.get(eid) or {}
                                    with st.expander(f"证据 {eid}", expanded=False):
                                        st.write((item.get("text") or "")[:600] or "（摘录不可用）")
                                        st.caption(
                                            f"来源：{item.get('title') or item.get('filename') or '—'} · 页码 {item.get('page_number') or '—'}"
                                        )
                                        # 公开链接从资料库匹配
                                        paper_uri = item.get("source_uri")
                                        if item.get("filename", "").endswith(".pdf") and paper_uri:
                                            st.link_button("公开链接", paper_uri)

                    if validation.get("approval_allowed"):
                        st.success("审批门槛：通过（引用可定位且关键结论有证据）")
                    else:
                        st.warning("审批门槛：未通过。建议先修订或补齐证据，再人工确认。")

                    if task.get("status") == "awaiting_approval":
                        st.info("当前为**草案，等待人工确认**。确认前不是最终结论。")
                        note = st.text_area("评审备注（可选）", key="approve-note")
                        ac1, ac2 = st.columns(2)
                        with ac1:
                            if st.button("确认最终建议", type="primary", use_container_width=True,
                                          disabled=not validation.get("approval_allowed")):
                                try:
                                    api(
                                        "POST",
                                        f"/research-tasks/{task_id}/approve",
                                        json={"reviewer_note": note or None},
                                    )
                                    st.success("报告已由人工确认。")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(str(exc))
                        with ac2:
                            revise_note = st.text_area("修订意见（打回重写）", key="revise-note")
                            if st.button("提交修订", use_container_width=True, disabled=not revise_note.strip()):
                                try:
                                    api(
                                        "POST",
                                        f"/research-tasks/{task_id}/revise",
                                        json={"reviewer_note": revise_note.strip()},
                                    )
                                    st.success("已提交修订，将生成新版本报告。")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(str(exc))
                    elif task.get("status") == "completed":
                        st.success("报告已由人工确认，可下载分发。")

                    st.markdown("#### 报告正文")
                    st.markdown(report.get("markdown") or "（无内容）")
                    st.download_button(
                        "下载 Markdown",
                        report.get("markdown") or "",
                        file_name=f"selection-report-{task_id}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                    try:
                        versions = api("GET", f"/research-tasks/{task_id}/report/versions") or []
                        if len(versions) > 1:
                            with st.expander(f"历史版本 · {len(versions)}", expanded=False):
                                for ver in versions:
                                    st.caption(
                                        f"v{ver.get('version')} · {ver.get('status')} · {ver.get('created_at', '')[:19]}"
                                    )
                    except Exception:
                        pass
