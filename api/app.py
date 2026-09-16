"""面向 Streamlit 和后续 Agent 的最小 HTTP 服务。"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, QDRANT_PATH, QDRANT_URL
from llm import model_label
from ingest.indexer import IndexingError, index_paper
from ingest.library import PdfIngestionError, ingest_document_bytes
from paper_library.store import PaperStore
from research.workflow import ResearchWorkflow
from sources.public import PublicSourceError, download_arxiv_pdf, search_arxiv, search_openalex

@asynccontextmanager
async def lifespan(_app: FastAPI):
    PaperStore().recover_interrupted_tasks()
    yield


app = FastAPI(title="ScholarGraph API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: str = Field(default="auto", pattern="^(rag|graphrag|auto)$")


DEFAULT_WEIGHTS = {
    "检索质量与可追溯性": 25,
    "在线延迟": 15,
    "数据更新复杂度": 15,
    "实施成本": 15,
    "运行成本": 10,
    "隐私与部署适配": 10,
    "运维与团队适配": 10,
}


class ResearchTaskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=4000)
    requirements: dict[str, str | int | float | bool] = Field(default_factory=dict)
    candidates: list[str] = Field(min_length=2, max_length=8)
    weights: dict[str, int] = Field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    @field_validator("candidates")
    @classmethod
    def validate_candidates(cls, candidates: list[str]) -> list[str]:
        cleaned = [candidate.strip() for candidate in candidates]
        if any(not candidate for candidate in cleaned):
            raise ValueError("候选方案不能为空。")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("候选方案不能重复。")
        return cleaned


class ApprovalRequest(BaseModel):
    reviewer_note: str | None = Field(default=None, max_length=2000)


class RevisionRequest(BaseModel):
    reviewer_note: str = Field(min_length=1, max_length=2000)


class SourceImportRequest(BaseModel):
    provider: str = Field(pattern="^arxiv$")
    external_id: str = Field(min_length=5, max_length=40)


def _run_index(paper_id: str) -> None:
    try:
        index_paper(paper_id)
    except IndexingError:
        pass


def _run_research(task_id: str, mode: str = "run") -> None:
    try:
        ResearchWorkflow().run(task_id, mode=mode)
    except Exception:
        # 失败节点、错误类型与消息已由工作流持久化，后台任务不再向上抛。
        pass


def execute_query(question: str, mode: str) -> dict:
    """延迟导入流水线，避免健康检查受模型客户端初始化影响。"""
    if mode == "rag":
        from rag.pipeline import rag_query
        return rag_query(question)
    if mode == "graphrag":
        from graphrag.pipeline import graphrag_query
        return graphrag_query(question)

    from router.router import smart_route
    return smart_route(question)


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok", "service": "scholargraph", "model": model_label()}


@app.get("/health/ready")
def ready() -> dict:
    checks: dict[str, str] = {}
    if QDRANT_PATH:
        parent = Path(QDRANT_PATH).resolve().parent
        checks["qdrant"] = "ok_local" if parent.exists() and os.access(parent, os.W_OK) else "unavailable"
    else:
        try:
            with urlopen(f"{QDRANT_URL}/healthz", timeout=2) as response:
                checks["qdrant"] = "ok" if response.status == 200 else "unavailable"
        except (URLError, TimeoutError, OSError):
            checks["qdrant"] = "unavailable"

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=2,
        )
        driver.verify_connectivity()
        driver.close()
        checks["neo4j"] = "ok"
    except Exception:
        checks["neo4j"] = "optional_unavailable"

    if checks["qdrant"] in {"ok", "ok_local"}:
        return {"status": "ok", "checks": checks}
    raise HTTPException(status_code=503, detail={"status": "unavailable", "checks": checks})


@app.post("/queries")
def query(request: QueryRequest) -> dict:
    try:
        return execute_query(request.question.strip(), request.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="查询执行失败，请检查服务状态后重试。") from exc


@app.get("/papers")
def list_papers() -> list[dict]:
    return PaperStore().list_papers()


@app.get("/sources/search")
def search_sources(q: str, provider: str = "arxiv", limit: int = 8) -> list[dict]:
    if not q.strip() or len(q) > 300:
        raise HTTPException(status_code=422, detail="搜索词长度必须为 1–300。")
    if provider not in {"arxiv", "openalex"}:
        raise HTTPException(status_code=422, detail="provider 仅支持 arxiv 或 openalex。")
    limit = min(max(limit, 1), 8)
    try:
        return search_arxiv(q.strip(), limit) if provider == "arxiv" else search_openalex(q.strip(), limit)
    except PublicSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/sources/import", status_code=status.HTTP_202_ACCEPTED)
def import_source(request: SourceImportRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        content = download_arxiv_pdf(request.external_id)
        paper, created = ingest_document_bytes(
            content,
            f"{request.external_id}.pdf",
            source_type="public_arxiv",
            source_uri=f"https://arxiv.org/abs/{request.external_id}",
            visibility="public",
        )
    except (PublicSourceError, PdfIngestionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if created:
        PaperStore().update_paper_status(paper["paper_id"], "indexing")
        background_tasks.add_task(_run_index, paper["paper_id"])
    return {"paper_id": paper["paper_id"], "created": created, "status": "indexing" if created else paper["status"]}


@app.post("/papers", status_code=status.HTTP_201_CREATED)
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    allowed_types = {
        "application/pdf",
        "application/x-pdf",
        "text/markdown",
        "text/plain",
    }
    filename = file.filename or "paper.pdf"
    if file.content_type not in allowed_types or not filename.lower().endswith((".pdf", ".md", ".markdown")):
        raise HTTPException(status_code=415, detail="仅支持 PDF 或 Markdown 文件。")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单个文件不能超过 50MB。")

    try:
        paper, created = ingest_document_bytes(content, filename)
    except PdfIngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if created:
        PaperStore().update_paper_status(paper["paper_id"], "indexing")
        background_tasks.add_task(_run_index, paper["paper_id"])
        paper = PaperStore().get_paper(paper["paper_id"]) or paper
    return {"paper": paper, "created": created}


@app.post("/papers/{paper_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_index(paper_id: str, background_tasks: BackgroundTasks) -> dict:
    store = PaperStore()
    paper = store.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="资料不存在。")
    if not store.list_chunks(paper_id):
        raise HTTPException(status_code=409, detail="资料没有已解析分块，无法重试索引。")
    store.update_paper_status(paper_id, "indexing")
    background_tasks.add_task(_run_index, paper_id)
    return {"paper_id": paper_id, "status": "indexing"}


@app.post("/research-tasks", status_code=status.HTTP_202_ACCEPTED)
def create_research_task(request: ResearchTaskRequest, background_tasks: BackgroundTasks) -> dict:
    if sum(request.weights.values()) != 100 or any(value < 0 for value in request.weights.values()):
        raise HTTPException(status_code=422, detail="评分权重必须为非负整数且合计 100。")
    store = PaperStore()
    if store.active_task_exists():
        raise HTTPException(status_code=409, detail="当前已有研究任务运行，请等待其完成。")
    task = store.create_task(
        {
            "task_id": str(uuid4()),
            "question": request.question.strip(),
            "requirements": request.requirements,
            "candidates": request.candidates,
            "weights": request.weights,
        }
    )
    background_tasks.add_task(_run_research, task["task_id"])
    return task


@app.get("/research-tasks/{task_id}")
def get_research_task(task_id: str) -> dict:
    task = PaperStore().get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="研究任务不存在。")
    return task


@app.get("/research-tasks/{task_id}/report")
def get_report(task_id: str) -> dict:
    report = PaperStore().get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="报告尚未生成。")
    return report


@app.get("/research-tasks/{task_id}/report.md")
def download_report(task_id: str) -> Response:
    report = PaperStore().get_report(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="报告尚未生成。")
    headers = {"Content-Disposition": f'attachment; filename="report-{task_id}.md"'}
    return Response(report["markdown"], media_type="text/markdown; charset=utf-8", headers=headers)


@app.post("/research-tasks/{task_id}/approve")
def approve_report(task_id: str, request: ApprovalRequest) -> dict:
    store = PaperStore()
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="研究任务不存在。")
    if task["status"] != "awaiting_approval":
        raise HTTPException(status_code=409, detail="只有等待审批的报告可以确认。")
    report = store.get_report(task_id)
    if not report or not report["validation"].get("valid"):
        raise HTTPException(status_code=409, detail="报告引用校验未通过，不能确认为最终版本。")
    return store.approve_report(task_id, request.reviewer_note) or {}


@app.post("/research-tasks/{task_id}/cancel")
def cancel_research_task(task_id: str) -> dict:
    store = PaperStore()
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="研究任务不存在。")
    if task["status"] not in ("queued", "running"):
        raise HTTPException(status_code=409, detail="只有排队或运行中的任务可以取消。")
    return store.request_cancel(task_id) or {}


@app.post("/research-tasks/{task_id}/resume", status_code=status.HTTP_202_ACCEPTED)
def resume_research_task(task_id: str, background_tasks: BackgroundTasks) -> dict:
    store = PaperStore()
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="研究任务不存在。")
    if task["status"] != "failed":
        raise HTTPException(status_code=409, detail="只有失败的任务可以恢复。")
    if store.active_task_exists():
        raise HTTPException(status_code=409, detail="当前已有研究任务运行，请等待其完成。")
    entry = task.get("failed_step") or "normalize_requirements"
    store.reset_for_retry(task_id, entry)
    store.add_event(task_id, entry, "resume_requested", "从失败节点恢复任务，优先复用已冻结证据")
    background_tasks.add_task(_run_research, task_id, "resume")
    return store.get_task(task_id)


@app.post("/research-tasks/{task_id}/revise", status_code=status.HTTP_202_ACCEPTED)
def revise_research_task(task_id: str, request: RevisionRequest, background_tasks: BackgroundTasks) -> dict:
    store = PaperStore()
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="研究任务不存在。")
    if task["status"] != "awaiting_approval":
        raise HTTPException(status_code=409, detail="只有等待审批的报告可以修订。")
    if not store.list_evidence(task_id):
        raise HTTPException(status_code=409, detail="没有可复用的证据，无法修订；请重新创建任务。")
    if store.active_task_exists():
        raise HTTPException(status_code=409, detail="当前已有研究任务运行，请等待其完成。")
    revised = store.begin_revision(task_id, request.reviewer_note)
    background_tasks.add_task(_run_research, task_id, "revise")
    return revised or {}


@app.get("/research-tasks/{task_id}/report/versions")
def list_report_versions(task_id: str) -> list[dict]:
    store = PaperStore()
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="研究任务不存在。")
    return store.list_report_versions(task_id)


@app.get("/research-tasks/{task_id}/evidence")
def list_task_evidence(task_id: str) -> list[dict]:
    store = PaperStore()
    task = store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="研究任务不存在。")
    return store.list_evidence(task_id)
