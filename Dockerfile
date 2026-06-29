FROM python:3.12-slim

WORKDIR /app

# Install SurrealDB
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSf https://install.surrealdb.com | sh && \
    mv /root/.surrealdb/surreal /usr/local/bin/surreal && \
    apt-get remove -y curl && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/

EXPOSE 8080

CMD ["agent-loop", "serve"]
