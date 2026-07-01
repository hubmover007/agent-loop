# Agent-Loop

> Loop Engine for AI Agent orchestration — shared memory, auto task decomposition, agent lifecycle management.

## Quick Start

### 一键安装

```bash
git clone https://github.com/hubmover007/agent-loop.git
cd agent-loop
bash install.sh
```

### Setup

```bash
agent-loop setup
```

交互式配置 LLM provider、SurrealDB、Embedding。

### Run

```bash
agent-loop start       # 启动系统
agent-loop chat "你好"  # 单次对话
agent-loop serve       # API 服务
agent-loop doctor      # 健康检查
agent-loop test        # 跑测试
agent-loop status      # 查看状态
```

### Docker 部署

```bash
docker-compose up -d
```

### 手动安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
agent-loop setup
agent-loop start
```

---

## 核心理念

以 **Loop Engine** 为心脏，**共享记忆** 为大脑，**Agent 生命循环** 为四肢，
构建一个能深度推理、自我进化、自动编排的 AI Agent 系统。

## 架构

```
用户输入
  ↓
MainLoop (7阶段循环)
  ├── INPUT      → 接收用户输入
  ├── RETRIEVE   → M-FLOW 图路由检索记忆
  ├── REASON     → DeepReason (RDT式迭代推理 + ACT自适应深度)
  ├── DECOMPOSE  → TaskManagerAgent LLM拆分任务
  ├── DISPATCH   → 依赖调度 → MoE路由 → Worker Agent分配
  ├── COLLECT    → 收集结果 + 质量评估 + 重规划
  └── OUTPUT     → 合成最终输出

TaskManagerAgent (中心编排)
  ├── decompose()     LLM拆分 → TaskRegistry注册 → 依赖解析
  ├── dispatch_ready() 依赖排序 → AgentRouter MoE匹配 → Worker分配
  ├── _execute_worker() BranchSpace init → agent.run → 评估 → accept/retry/fail
  ├── collect_all()   等待完成 → 自动派遣新ready → 结果收集
  └── replan()        LLM重新规划失败任务

MemoryPool (共享大脑, SurrealDB)
  ├── Fact Layer      事实知识 (稳定)
  ├── Facet Layer     观点面相 (多视角)
  ├── Episode Layer   情节记忆 (时序)
  └── Project Layer   项目知识 (工作区)
```

## 快速开始

### 安装

```bash
pip install -e ".[all]"
```

### 配置

```bash
agent-loop init-config
# 编辑 agent-loop.yaml
```

### 启动 API 服务

```bash
agent-loop serve --port 8000
# 或
python -m src.cli serve --port 8000
```

### 单次对话

```bash
agent-loop chat "帮我检查服务器状态"
```

### Docker 部署

```bash
docker-compose up -d
```

## 系统状态

```bash
agent-loop status
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 记忆/图库 | SurrealDB | 单二进制，图+向量+文档合一 |
| LLM | DeepSeek / Anthropic Bedrock / OpenAI兼容 | 多Provider支持 |
| Web API | FastAPI + uvicorn | REST + WebSocket |
| Agent 隔离 | asyncio subprocess | AgentWorker 进程隔离 |
| 部署 | Docker Compose | 单机+平台双模式 |

## 工具

| 工具 | 说明 |
|------|------|
| `ssh` | 异步 SSH 命令执行 + 文件上传/下载 |
| `web` | Brave Search + 网页内容提取 |
| `code` | Python 执行 + Shell 命令 + 文件读写 |

## 模块清单

| 模块 | 行数 | 说明 |
|------|------|------|
| `src/core.py` | 145 | 核心类型定义 |
| `src/llm.py` | 280 | LLM Provider 实现 |
| `src/memory/__init__.py` | 275 | MemoryPool (SurrealDB) |
| `src/memory/graph_route.py` | 360 | M-FLOW 图路由检索 |
| `src/loop_engine/__init__.py` | 335 | LoopConfig + AgentLoop + ToolLoop |
| `src/loop_engine/main_loop.py` | 320 | MainLoop 7阶段 |
| `src/loop_engine/deep_reason.py` | 290 | RDT式迭代推理 |
| `src/agent/__init__.py` | 380 | Agent + AgentPool + AgentRouter + AgentEvaluator |
| `src/task/__init__.py` | 345 | TaskTree + TaskScheduler + BranchSpace |
| `src/task_manager.py` | 630 | TaskManagerAgent (中心编排) |
| `src/pipeline.py` | 467 | TaskPipeline (旧版, 被TaskManagerAgent取代) |
| `src/worker.py` | 350 | AgentWorker 进程隔离 |
| `src/tools/base.py` | 85 | ToolInterface + ToolRegistry |
| `src/tools/ssh.py` | 300 | SSH Tool |
| `src/tools/web.py` | 370 | Web + Code Tools |
| `src/web/__init__.py` | 290 | FastAPI Web API |
| `src/cli.py` | 150 | CLI 入口 |
| **总计** | **~5400** | 20个源文件 |

## 研究参考

| 项目 | 启发点 |
|------|--------|
| AgentOrchestra (arxiv 2506.12508) | Planning Agent 分解+委派+自适应 |
| AutoGen Orchestrator-Worker | 中心编排器 dispatch+collect |
| CrewAI Hierarchical | Manager Agent 动态创建任务+委派 |
| M-FLOW | 四层倒锥知识图谱 + 图路由检索 |
| OpenMythos | RDT 循环深度推理 |
| LangGraph Supervisor | 依赖感知任务路由 |

## License

MIT
