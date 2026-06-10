"""
全局配置：所有连接参数、模型名称、路径都从这里读。
路径会根据实际工作目录自动解析，无需硬编码绝对路径。
"""
import os
import sys
from pathlib import Path

# ============ 路径 ============
PROJECT_ROOT = Path(__file__).parent.resolve()
# 12 篇 PDF 所在目录（可通过环境变量覆盖，默认 ./papers/）
ARTICLES_DIR = Path(os.environ.get("ARTICLES_DIR", PROJECT_ROOT / "papers")).resolve()
PAPERS_TEXT_DIR = PROJECT_ROOT / "data" / "papers_text"
PAPER_MAP_JSON = PROJECT_ROOT / "data" / "papers_meta.json"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.json"

# ============ Qdrant ============
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "papers"

# ============ Neo4j ============
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
# 密码走环境变量，避免硬编码 + 不小心提交到 git
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j12345")
NEO4J_DATABASE = "neo4j"  # 默认 DB

# ============ Ollama ============
# 嵌入模型（计划指定 4b 版本）
EMBED_MODEL = "qwen3-embedding:4b"
# 生成模型：默认 4b（推荐），若跑不动再降到更小；不要用 0.6b，Cypher 生成会挂
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = "http://localhost:11434"

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

# ============ LLM 容错 ============
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
