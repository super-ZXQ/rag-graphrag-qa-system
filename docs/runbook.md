# 运行手册（本地 / Docker）

## A. 本地 Qdrant 模式（当前已验证）

前置：Python 3.12、Ollama（`qwen3-embedding:4b`）、可选 `qwen2.5:7b`。

```powershell
Set-Location 'D:\学术论文问答系统'
$env:QDRANT_PATH = 'data/qdrant_local'
$env:LLM_PROVIDER = 'ollama'          # DeepSeek 有余额时可改 deepseek
$env:LLM_MODEL = 'qwen2.5:7b'

# 终端 1
uvicorn api.app:app --host 127.0.0.1 --port 8000
# 终端 2
streamlit run ui/app.py --server.port 8501 --server.headless true
```

初始化语料（幂等）：

```powershell
python scripts/seed_demo.py --index
python scripts/import_golden_papers.py --index
python scripts/build_citation_graph.py --limit 6
```

验证：

```powershell
python -m pytest -q
Invoke-RestMethod http://127.0.0.1:8000/health/live
```

## B. Docker 模式（Engine 恢复后）

```powershell
# 确认 Engine
docker version
docker compose config

# 启动
docker compose up --build
```

预期端口：

| 服务 | URL |
|---|---|
| UI | http://localhost:8501 |
| API | http://localhost:8000/docs |
| Qdrant | http://localhost:6333 |
| Neo4j | http://localhost:7474 |

完整链路验收：

1. 启动后 `GET /health/ready` 与 UI 侧边栏「API 正常」。
2. 上传/导入资料 → 状态到 ready。
3. 创建黄金选型任务 → 等待 `awaiting_approval`。
4. 查看证据与 claim → 门槛通过则审批 → 下载 Markdown。
5. 确认镜像内无 `.env.local`：`docker compose exec api ls -la /app | grep env` 应无 `.env.local`。

失败排查：

- `failed to connect to the docker API`：启动 Docker Desktop Linux Engine。
- Ollama 连不上：宿主机 `ollama serve`，必要时设 `OLLAMA_BASE_URL`。
- 端口占用：改 compose 端口映射，UI 的 `API_URL` 同步改。
- 误设 `QDRANT_PATH`：容器内必须为空，走 `QDRANT_URL=http://qdrant:6333`。

## C. 常用故障恢复

| 现象 | 处理 |
|---|---|
| 服务重启前任务是 running | API 启动时自动转为 failed，再用 UI 或 `POST .../resume` 恢复 |
| 任务 failed | UI「从失败处恢复」或 `POST .../resume`（复用已冻结证据） |
| 资料 index_failed | 看 error_message，`POST /papers/{id}/retry` |
| 报告审批被拒 | 看 fabricated/coverage 指标，`POST .../revise` |
| 旧 hybrid_graph 差指标 | 以最新 `evaluation/results.json` 为准，勿引用历史坏数字 |
