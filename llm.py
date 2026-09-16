"""生成模型工厂：DeepSeek API 为默认方案，Ollama 可作为本地备用。"""
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SECONDS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)


def get_chat_model():
    """按配置创建聊天模型；只在首次实际请求时调用，避免导入模块时读取密钥。"""
    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "未设置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env.local 后设置密钥，"
                "或临时设置环境变量。"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            temperature=LLM_TEMPERATURE,
            timeout=DEEPSEEK_TIMEOUT_SECONDS,
        )

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )

    raise RuntimeError(
        f"不支持的 LLM_PROVIDER={LLM_PROVIDER!r}；可选值为 deepseek 或 ollama。"
    )


def model_label() -> str:
    """返回可安全展示和记录的模型标识，不包含任何凭据。"""
    return DEEPSEEK_MODEL if LLM_PROVIDER == "deepseek" else LLM_MODEL
