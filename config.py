"""
全局配置：所有连接参数、模型名称、路径都从这里读。
路径会根据实际工作目录自动解析，无需硬编码绝对路径。
"""
import os
import sys
from pathlib import Path

# ============ 路径 ============
PROJECT_ROOT = Path(__file__).parent.resolve()
try:
    from dotenv import load_dotenv

    # 开发环境只读取本地、被 Git 忽略的配置；系统环境变量优先级更高。
    load_dotenv(PROJECT_ROOT / ".env.local")
except ImportError:
    pass

# 数据目录与本地论文库。
DATA_DIR = PROJECT_ROOT / "data"
LIBRARY_DB = DATA_DIR / "scholargraph.sqlite3"
UPLOADS_DIR = DATA_DIR / "uploads"

# 12 篇 PDF 所在目录（可通过环境变量覆盖，默认 ./papers/）
ARTICLES_DIR = Path(os.environ.get("ARTICLES_DIR", PROJECT_ROOT / "papers")).resolve()
PAPERS_TEXT_DIR = DATA_DIR / "papers_text"
PAPER_MAP_JSON = DATA_DIR / "papers_meta.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"

# ============ Qdrant ============
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers_v1")
QDRANT_COLLECTION_VERSION = os.getenv("QDRANT_COLLECTION_VERSION", "v1")

# ============ Neo4j ============
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# 密码走环境变量，避免硬编码 + 不小心提交到 git
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j12345")
NEO4J_DATABASE = "neo4j"  # 默认 DB

# ============ 模型服务 ============
# 嵌入仍在本地运行；生成模型可通过环境变量切换 DeepSeek API 或 Ollama。
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()

# DeepSeek 使用 OpenAI 兼容接口。密钥只从环境变量读取，绝不写入源码。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# Ollama 是离线备用方案。
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ============ 嵌入维度 ============
# qwen3-embedding:4b 实际输出维度（qwen3-embedding 系列默认 2560）；
# 首次运行时若发现不一致，请用以下命令确认后改这里：
#   from langchain_ollama import OllamaEmbeddings
#   print(len(OllamaEmbeddings(model=EMBED_MODEL).embed_query("test")))
VECTOR_SIZE = 2560

# ============ 分块参数 ============
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ============ 检索参数 ============
RETRIEVER_K = 3  # Top-K
LEXICAL_RETRIEVER_K = int(os.getenv("LEXICAL_RETRIEVER_K", "8"))
VECTOR_RETRIEVER_K = int(os.getenv("VECTOR_RETRIEVER_K", "8"))
HYBRID_RETRIEVER_K = int(os.getenv("HYBRID_RETRIEVER_K", "5"))

# ============ 研究 Agent ============
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
AGENT_MAX_CONTEXT_CHUNKS = int(os.getenv("AGENT_MAX_CONTEXT_CHUNKS", "12"))
# 单个节点（检索/模型调用）失败后的最大重试次数；失败会持久化 failed_step 与 error_type。
WORKFLOW_MAX_RETRIES = int(os.getenv("WORKFLOW_MAX_RETRIES", "2"))
# 可选的成本估算单价（人民币元 / 百万 token）。默认留空，不臆造成本；
# 只有显式配置后才输出 estimated_cost，Token 用量优先取 API 返回的 usage。
def _optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


DEEPSEEK_PRICE_PER_1M_INPUT = _optional_float("DEEPSEEK_PRICE_PER_1M_INPUT")
DEEPSEEK_PRICE_PER_1M_OUTPUT = _optional_float("DEEPSEEK_PRICE_PER_1M_OUTPUT")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")

# ============ 路由参数 ============
# 语义路由：Top1 与 Top2 相似度差值阈值，低于此值判为不确定 → 走 LLM
ROUTER_MARGIN_THRESHOLD = 0.03
# 路由 LLM 与问答 LLM 共用 qwen3:4b；如想区分可在调 router 时单独传

# ============ 嵌入 / 构建批量 ============
EMBED_BATCH_SIZE = 16        # build_qdrant.py 单批嵌入条数
QDRANT_UPSERT_BATCH = 100    # build_qdrant.py 单批 upsert 条数

# ============ LLM 容错 ============
LLM_MAX_RETRIES = 3
# LLM 生成的 Cypher 常带 ```cypher ``` 包裹，需要清洗
CYPHER_CLEAN_REGEX = r"```[a-zA-Z]*\n?|```"

# ============ LangChain 版本信息 ============
# 实际安装的是 langchain 1.x，部分 API 与 0.3.x 不同：
# - `from langchain_community.xxx` 已经并入 `from langchain.xxx` 或对应子包
# - 不再使用 `langchain-experimental`（部分功能并入主包）
# - LCEL 链式语法（`|` 操作符）仍然有效


def self_check():
    """启动时自检（任意一项失败则抛错）"""
    assert ARTICLES_DIR.exists(), f"PDF 目录不存在: {ARTICLES_DIR}"
    pdfs = list(ARTICLES_DIR.glob("*.pdf"))
    assert len(pdfs) == 12, (
        f"应恰好 12 篇 PDF，实际找到 {len(pdfs)} 篇: "
        f"{[p.name for p in pdfs]}"
    )
    assert NEO4J_PASSWORD, "NEO4J_PASSWORD 不能为空（请设置环境变量或使用默认值）"
    print(f"[config] self check passed (Python {sys.version.split()[0]})")
    print(f"[config] ARTICLES_DIR = {ARTICLES_DIR}")
    print(f"[config] PROJECT_ROOT = {PROJECT_ROOT}")


if __name__ == "__main__":
    self_check()
