FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# 复制项目
COPY . /app/

# 安装
RUN pip install --no-cache-dir -e ".[all]"

# 创建必要目录
RUN mkdir -p logs state/agents config

# 初始化配置
RUN python3 -m src.cli init-config || true

EXPOSE 8000

CMD ["agent-loop", "start"]
