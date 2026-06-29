# Agent-Loop

> **Loop Engine AI Agent System**
>
> 以 Loop Engine 为心脏，共享记忆为大脑，Agent 生命循环为四肢，
> 构建深度推理、自我进化、自动编排的 AI Agent 系统。

## 核心理念

| 理念 | 实现 |
|------|------|
| **Loop-first** | 三级循环引擎：MainLoop → AgentLoop → ToolLoop |
| **共享记忆** | M-FLOW 四层倒锥拓扑 + 图路由检索 + SurrealDB 统一存储 |
| **自动 Agent 管理** | 按需创建/销毁、MoE 路由、质量评估、异常遗弃 |
| **任务自动拆分** | LLM 驱动分解 → TaskTree → 依赖管理 → 并行派遣 |
| **深度推理** | 模型潜空间推理(RDT) + Agent 级显式循环 |
| **平台化部署** | Docker Compose 多机 + pip install 单机 |

## 快速开始

```bash
# 单机模式
pip install agent-loop
agent-loop init          # 初始化 SurrealDB Schema
agent-loop run "your task"

# Docker 模式
docker-compose up -d
```

## 架构

参见 [ARCHITECTURE.md](ARCHITECTURE.md)

## 项目状态

🚧 早期开发阶段。核心框架已搭建，LLM Provider 和工具生态待接入。
