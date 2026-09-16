FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --index-url "$PIP_INDEX_URL" -r requirements.txt

# .dockerignore 已排除 .env.local / data / .git，密钥不会进入镜像
COPY . .

RUN mkdir -p /app/data

EXPOSE 8000 8501

# 默认命令由 compose 覆盖；镜像本身不写死 API Key
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
