# Agent-Loop Architecture

> **核心理念**: 以 Loop Engine 为心脏，共享记忆为大脑，Agent 生命循环为四肢，
> 构建一个能深度推理、自我进化、自动编排的 AI Agent 系统。

## 设计目标

| 目标 | 描述 |
|------|------|
| Loop-first | 一切围绕循环设计：推理循环、Agent 生命周期循环、任务编排循环 |
| 共享记忆 | 所有 Agent 读写同一个记忆池，无拷贝无同步 |
| 自动 Agent 管理 | 按需创建/销毁 Agent，自动评估质量，异常自动遗弃 |
| 任务自动拆分 | 复杂任务自动分解为子任务树，Agent 自治子任务空间 |
| 深度推理 | 模型内潜空间推理(类 Mythos) + Agent 级显式循环推理 混合 |
| 平台化部署 | 多用户、多租户、可水平扩展；同时支持单机 `pip install` |

---

## 技术选型

### 选型原则
1. 优先嵌入式/单二进制方案（支持单机部署）
2. 一个组件只做一件事，选最适合的而非最流行的
3. 所有依赖需支持 Python 3.12+

### 核心技术栈

| 层 | 技术 | 理由 |
|----|------|------|
| 记忆/图库 | **SurrealDB** | 单二进制、图关系+向量搜索+全文搜索+文档模型合一、MIT 许可、活跃维护 |
| 向量索引 | SurrealDB 内置 MTREE/HNSW 索引 | 避免引入额外向量库（Qdrant/Milvus），减少运维复杂度 |
| 任务编排 | **自定义 async Loop Engine** | Temporal/Prefect 太重、LangGraph 绑定 LangChain 生态、CrewAI 不够灵活 |
| Agent 运行时 | **Python asyncio + ProcessPoolExecutor** | Agent 进程隔离（一个 Agent 崩不影响其他）、async 高并发 |
| Agent 通信 | **SurrealDB 共享表 + asyncio.Queue** | 共享记忆 = 共享 DB；实时事件 = Queue |
| 嵌入模型 | 可配置（OpenAI/本地 bge-large/自定义） | 不绑定任何厂商 |
| LLM 后端 | 可配置（DeepSeek/Claude/GPT/本地） | 多 Provider 支持 |
| 部署 | Docker Compose + pip install 双模式 | 兼顾平台化和单机 |

### 为什么不选...

| 候选 | 不选原因 |
|------|---------|
| KuzuDB | 2025.10 被 Apple 收购后归档，停止维护 |
| Neo4j | 需要 Java、商业许可限制、不是嵌入式 |
| FalkorDB | 依赖 Redis 做后端，多一个服务 |
| Qdrant/Milvus | 多一个服务，SurrealDB 已内建向量搜索 |
| LangGraph | 绑定 LangChain、不适合脱离其生态 |
| Temporal | 太重、需要独立服务、不适合单机 pip install |
| Prefect | 面向数据流水线、不适合 Agent 动态生命周期 |

---

## 系统架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT-LOOP SYSTEM                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                    LOOP ENGINE (心脏)                         ││
│  │                                                               ││
│  │  MainLoop: 接收 → 推理 → 分解 → 派遣 → 收集 → 输出           ││
│  │  AgentLoop: 收任务 → 规划 → 执行 → 自评 → 提交 → 销毁        ││
│  │  ToolLoop:  调用 → 验证 → 重试(3次) → 返回/失败               ││
│  └──────────────────────────────────────────────────────────────┘│
│                              │                                     │
│  ┌───────────────────────────┼───────────────────────────────────┐│
│  │                    MEMORY POOL (大脑)                          ││
│  │                                                               ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         ││
│  │  │ Facts   │  │ Facets  │  │Episodes │  │Projects │         ││
│  │  │ (锥尖)  │→│ (维度)  │→│ (事件)  │→│ (全景)  │         ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘         ││
│  │       ↑ 语义边过滤 + 路径代价传播 → 图路由检索                ││
│  ├───────────────────────────────────────────────────────────────┤│
│  │  SurrealDB: Graph Relations + Vector Search + Doc Store       ││
│  └───────────────────────────────────────────────────────────────┘│
│                              │                                     │
│  ┌───────────────────────────┼───────────────────────────────────┐│
│  │                    AGENT ORCHESTRATOR                          ││
│  │                                                               ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                    ││
│  │  │ Factory  │  │Scheduler │  │Evaluator │                    ││
│  │  │ 创建Agent │  │ 派遣任务  │  │ 质量评估  │                    ││
│  │  └──────────┘  └──────────┘  └──────────┘                    ││
│  │                                                               ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                    ││
│  │  │ TaskTree │  │BranchMgr│  │Discard   │                    ││
│  │  │ 任务树    │  │ 分支空间  │  │ 遗弃池    │                    ││
│  │  └──────────┘  └──────────┘  └──────────┘                    ││
│  └───────────────────────────────────────────────────────────────┘│
│                              │                                     │
│  ┌───────────────────────────┼───────────────────────────────────┐│
│  │                    TOOL ECOSYSTEM                              ││
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           ││
│  │  │ SSH │ │Web  │ │Code │ │File │ │API  │ │MCP  │           ││
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘           ││
│  │  统一 ToolInterface: execute() → ToolResult                   ││
│  └───────────────────────────────────────────────────────────────┘│
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 一、Loop Engine 设计（心脏）

### 1.1 MainLoop（系统主循环）

```
MainLoop:
  while True:
    1. INPUT  → 接收用户输入，解析意图
    2. RETRIEVE → 从 Memory Pool 检索相关上下文
                    (M-FLOW 图路由: 锥尖锚点 → 路径代价传播 → 最优 Episode)
    3. REASON  → 深度推理循环
                  ┌─ 内部循环 (模型级): continuous latent reasoning
                  │   每轮迭代等价于一步 CoT，但发生在潜空间
                  │   ACT 自适应停止: 简单问题 N=1, 复杂问题 N=5+
                  │
                  └─ 外部循环 (Agent 级): 显式多轮思考
                      每轮: 审视上轮 → 查漏补缺 → 优化方案 → 决定继续/输出
    4. DECOMPOSE → 将推理结果分解为 TaskTree
                   RootTask → SubTask₁, SubTask₂, ...
                   每个 SubTask 包含: scope, context, tools, deadline
    5. DISPATCH → AgentScheduler 派遣:
                   ┌─ 匹配 Agent 专家路由(MoE 式)
                   │  向量化任务 → top-K 专家匹配
                   │  无匹配 → AgentFactory 创建新 Agent
                   └─ 并行/串行调度(按依赖关系)
    6. COLLECT → AgentEvaluator 收集结果:
                 ┌─ 质量评分(自动+人工标准)
                 │  合格 → 合并到主任务
                 │  不合格 → DiscardPool(遗弃池，可审计)
                 │  异常 → 销毁 Agent + 记录
                 └─ 超时 → 强制终止
    7. OUTPUT → 合成最终回复 → 写入 Memory Pool → 返回用户
```

### 1.2 AgentLoop（Agent 内部循环）

```
AgentLoop(agent_id, task):
  1. INIT → 从 Memory Pool 拉取共享上下文
            + Agent 专属 Branch Space 初始化
            + 工具权限白名单

  2. PLAN → 分析子任务 → 规划执行步骤
            如果子任务仍复杂 → 递归分解(SubSubTask)

  3. EXECUTE → for step in plan:
                ┌─ 调用工具(ToolLoop)
                │  成功 → 继续
                │  失败 → 最多重试3次 → 标记失败
                └─ 每步产出记录到 Branch Space

  4. SELF_EVAL → 自评结果质量:
                  ┌─ 完整性: 是否覆盖所有要求
                  │  正确性: 工具输出是否合理
                  │  相关性: 是否偏离任务目标
                  └─ 评分 0-1 → <0.5 自毁，不提交

  5. SUBMIT → 将结果提交到 TaskTree 对应节点
              Branch Space 标记为 done
              清理 Agent 资源

  6. DESTROY → Agent 从 AgentPool 移除
               资源回收
```

### 1.3 ToolLoop（工具调用循环）

```
ToolLoop(tool_name, params):
  attempts = 0
  while attempts < 3:
    result = tool.execute(params)
    if result.status == "success":
      return result
    if result.status == "transient_error":
      attempts += 1
      backoff(2^attempts)
      continue
    if result.status == "fatal_error":
      return result  # 不可重试，直接返回
  return ToolResult(status="failed", reason="max_retries")
```

### 1.4 推理深度设计（混合模式）

```
深度推理 = 模型内循环 × Agent 级循环

模型内循环(类 RDT):
  - 同一推理上下文，迭代 N 次
  - 每次迭代: h_{t+1} = f(h_t, input, memory_context)
  - ACT 机制: confidence_score > threshold → 早停
  - N 由任务复杂度动态决定:
    简单问答: 1-2 轮
    代码分析: 3-5 轮
    架构设计: 5-8 轮
    安全审计: 8-16 轮

Agent 级循环:
  - 每轮产出可见文本思考
  - 包含: 方案审视 → 工具调用 → 结果验证 → 调整
  - 可被用户中断/干预
```

---

## 二、Memory Pool 设计（大脑）

### 2.1 四层倒锥拓扑（M-FLOW 风格）

```
        Layer 0: FACTS (锥尖 ← 检索入口)
        ┌────────────────────────────────┐
        │ Entity (实体)                   │
        │ - 用户: user:{id}               │
        │ - 服务器: server:{ip}           │
        │ - 工具: tool:{name}             │
        │                                 │
        │ FacetPoint (精确断言)            │
        │ - "nginx 监听端口 8000"         │
        │ - "部署时间 2026-06-15"         │
        └────────────────────────────────┘
                    ↓
        Layer 1: FACETS (语义维度)
        ┌────────────────────────────────┐
        │ 运维操作: ops:ssh, ops:deploy   │
        │ 飞书API: feishu:im, feishu:doc  │
        │ AWS 操作: aws:ec2, aws:iam      │
        └────────────────────────────────┘
                    ↓
        Layer 2: EPISODES (事件上下文)
        ┌────────────────────────────────┐
        │ episode:deploy_glm52_20260615  │
        │ episode:fix_bot_i_xxx_20260629 │
        │ episode:research_mflow_20260629│
        └────────────────────────────────┘
                    ↓
        Layer 3: PROJECTS (锥底 ← 返回目标)
        ┌────────────────────────────────┐
        │ project:easyclaw_monitor       │
        │ project:agent_loop             │
        │ project:billing_platform       │
        └────────────────────────────────┘
```

### 2.2 语义边设计

```surreal
-- SurrealDB Schema

-- Layer 0: Facts
DEFINE TABLE fact SCHEMAFULL;
DEFINE FIELD type ON fact TYPE string ASSERT $value IN ["entity", "facetpoint"];
DEFINE FIELD name ON fact TYPE string;
DEFINE FIELD value ON fact;
DEFINE FIELD embedding ON fact TYPE array<float>;
DEFINE FIELD created_at ON fact TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_fact_embedding ON fact FIELDS embedding MTREE DIMENSION 1024;

-- Layer 1: Facets
DEFINE TABLE facet SCHEMAFULL;
DEFINE FIELD name ON facet TYPE string;
DEFINE FIELD description ON facet TYPE string;
DEFINE FIELD embedding ON facet TYPE array<float>;
DEFINE INDEX idx_facet_embedding ON facet FIELDS embedding MTREE DIMENSION 1024;

-- Layer 2: Episodes
DEFINE TABLE episode SCHEMAFULL;
DEFINE FIELD title ON episode TYPE string;
DEFINE FIELD summary ON episode TYPE string;
DEFINE FIELD embedding ON episode TYPE array<float>;
DEFINE FIELD created_at ON episode TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_episode_embedding ON episode FIELDS embedding MTREE DIMENSION 1024;

-- Layer 3: Projects
DEFINE TABLE project SCHEMAFULL;
DEFINE FIELD name ON project TYPE string;
DEFINE FIELD description ON project TYPE string;
DEFINE FIELD embedding ON project TYPE array<float>;
DEFINE INDEX idx_project_embedding ON project FIELDS embedding MTREE DIMENSION 1024;

-- 语义边: 每条边有自然语言描述 + 向量化
DEFINE TABLE edge SCHEMAFULL;
DEFINE FIELD source ON edge TYPE record<TABLE>;
DEFINE FIELD target ON edge TYPE record<TABLE>;
DEFINE FIELD relation ON edge TYPE string;        -- 自然语言描述关系
DEFINE FIELD embedding ON edge TYPE array<float>; -- 关系语义向量
DEFINE INDEX idx_edge_embedding ON edge FIELDS embedding MTREE DIMENSION 1024;
```

### 2.3 图路由检索算法

```python
async def graph_route_retrieve(query: str, top_k: int = 5) -> list[Episode]:
    """
    M-FLOW 风格图路由检索:
    1. 在锥尖锚点搜索 (向量 + 全文)
    2. 沿语义边向下传播代价
    3. 返回最小代价路径的 Episode
    """
    query_embedding = await embed(query)

    # Phase 1: 锥尖广撒网 (在所有层同时搜索)
    anchors = await search_all_layers(query_embedding, top_n=100)

    # Phase 2: 投影到图 (提取子图 + 一跳邻居)
    subgraph = await extract_subgraph(anchors, hop=1)

    # Phase 3: 代价传播 (锥尖 → 锥底)
    episode_scores = {}
    for episode in subgraph.episodes:
        min_cost = float('inf')
        for anchor in anchors:
            for path in find_all_paths(anchor, episode, max_hops=3):
                cost = compute_path_cost(path, query_embedding)
                min_cost = min(min_cost, cost)
        episode_scores[episode] = min_cost

    # 返回最小代价的 top_k Episode
    return sorted(episode_scores, key=episode_scores.get)[:top_k]

def compute_path_cost(path, query_embedding):
    """路径代价 = 起始代价 + Σ边代价 + 跳跃惩罚 + 直接命中惩罚"""
    start_cost = cosine_distance(path[0].embedding, query_embedding)
    edge_cost = sum(
        cosine_distance(edge.embedding, query_embedding)
        for edge in path.edges
    )
    hop_penalty = len(path.edges) * 0.1  # 每跳0.1惩罚
    direct_hit_penalty = 0.3 if path[0].layer == "episode" else 0

    return start_cost + edge_cost + hop_penalty + direct_hit_penalty
```

### 2.4 模型参数检索兼容

Mythos 式模型参数检索的理解：
- Mythos 在训练时将知识内化到权重中，检索即推理
- 我们的系统不可能做到这个（无法训练模型），但可以借鉴其本质：
  **检索不应仅是"找文本"，而应该是"理解后重建"**

实现方式：
```
传统 RAG:       Query → Vector Match → Return Text
模型增强检索:    Query → Vector Match → Graph Route → LLM 理解重建 → Return
                                                    ↑
                                        这里的"理解"是对检索结果的
                                        深度加工，不是简单拼接
```

具体做法：
1. 检索到的记忆不直接拼接进 prompt
2. 而是先经过一个**推理子循环**：LLM 理解检索到的记忆 → 提取关键信息 → 建立与当前查询的关联 → 生成"理解后的上下文"
3. 这个"理解后的上下文"才是真正注入 prompt 的内容

---

## 三、Agent 管理系统

### 3.1 Agent 生命周期

```
                    ┌──────────┐
                    │  CREATED │ ← AgentFactory.create(task, tools)
                    └────┬─────┘
                         ↓
                    ┌──────────┐
                    │  IDLE    │ ← 在 AgentPool 等待任务
                    └────┬─────┘
                         ↓
                    ┌──────────┐
                    │ RUNNING  │ ← 执行 AgentLoop
                    └────┬─────┘
                    ┌─────┼─────┐
                    ↓     ↓     ↓
               ┌────────┐┌────────┐┌──────────┐
               │  DONE  ││FAILED ││DESTROYED │
               │ 提交结果││ 自评   ││ 被 Evaluator│
               │ 回主任务││ 不达标 ││ 强制销毁   │
               └────────┘└───┬────┘└──────────┘
                            ↓
                      ┌──────────┐
                      │ DISCARD  │ ← 结果进遗弃池
                      └──────────┘
```

### 3.2 AgentPool

```python
class AgentPool:
    """Agent 池管理器"""
    agents: dict[str, Agent]          # agent_id → Agent
    max_concurrent: int = 50          # 最大并发 Agent 数
    idle_timeout: int = 300           # 闲置超时(秒)

    async def acquire(self, task: Task) -> Agent:
        """获取或创建 Agent"""
        # 1. 尝试复用闲置 Agent (相似任务)
        # 2. 无可用 → 创建新 Agent
        # 3. 超限 → 等待或拒绝

    async def release(self, agent: Agent):
        """回收 Agent"""
        # 清理 Branch Space
        # 重置状态
        # 放回池中(或销毁)

    async def destroy(self, agent_id: str):
        """强制销毁 Agent"""
        # 终止进程
        # 清理资源
        # 结果进 DiscardPool
```

### 3.3 Agent 专家路由（MoE 风格）

```python
class AgentRouter:
    """Agent 专家路由器: 类似 MoE 的路由选择"""
    experts: dict[str, ExpertProfile]  # 专家画像

    async def route(self, task: Task, k: int = 3) -> list[Agent]:
        """为任务选择 top-K 专家 Agent"""
        task_embedding = await embed(task.description)

        scores = {}
        for expert_id, profile in self.experts.items():
            # 相似度 + 成功率加权 + 负载因子
            sim = cosine_similarity(task_embedding, profile.embedding)
            success_rate = profile.success_count / max(profile.total_count, 1)
            load_factor = 1.0 / (profile.current_load + 1)
            scores[expert_id] = sim * 0.5 + success_rate * 0.3 + load_factor * 0.2

        top_k = sorted(scores, key=scores.get, reverse=True)[:k]

        # 如果最高分仍低于阈值 → 创建新 Agent
        if scores[top_k[0]] < 0.3:
            return [await self.factory.create(task)]

        return [self.pool.agents[eid] for eid in top_k if eid in self.pool.agents]
```

### 3.4 AgentEvaluator（质量评估）

```python
class AgentEvaluator:
    """Agent 输出质量评估器"""

    async def evaluate(self, agent: Agent, result: TaskResult) -> EvaluationResult:
        """多维度评估 Agent 产出"""
        scores = {}

        # 1. 完整性: 是否覆盖了任务的所有要求
        scores["completeness"] = await self._check_completeness(agent.task, result)

        # 2. 正确性: 工具输出是否合理
        scores["correctness"] = await self._check_correctness(result)

        # 3. 相关性: 是否偏离主题
        scores["relevance"] = await self._check_relevance(agent.task, result)

        # 4. 效率: 使用的步数/token 是否合理
        scores["efficiency"] = self._check_efficiency(agent.step_count, agent.task)

        overall = weighted_average(scores, weights=[0.3, 0.3, 0.25, 0.15])

        return EvaluationResult(
            scores=scores,
            overall=overall,
            action="accept" if overall >= 0.7 else "discard",
            reason=self._generate_reason(scores)
        )
```

### 3.5 DiscardPool（遗弃池）

```python
class DiscardPool:
    """遗弃池: 存不合格结果的审计日志"""

    async def discard(self, agent: Agent, result: TaskResult, reason: str):
        """将不合格结果放入遗弃池"""
        record = DiscardRecord(
            agent_id=agent.id,
            task_id=agent.task.id,
            result=result,
            reason=reason,
            timestamp=time.now(),
            agent_log=agent.logs,
        )
        await self.db.insert("discard_pool", record)

    async def query(self, agent_id: str = None, task_id: str = None) -> list[DiscardRecord]:
        """查询遗弃记录（支持按Agent或任务过滤）"""
        # 用于审计追踪
```

---

## 四、Task 管理系统

### 4.0 架构变更：TaskManagerAgent（中心编排 Agent）

> **关键设计**: Task 管理由一个专门的 **TaskManagerAgent** 负责，
> 它本身是一个 LLM-driven Agent，拥有完整的任务生命周期管理能力。

```
MainLoop (用户输入 → 深度推理)
  → TaskManagerAgent (拆分 → 注册 → 派遣 → 跟踪 → 收集)
    → Worker Agent 1 (执行子任务, BranchSpace 隔离)
    → Worker Agent 2 (执行子任务, BranchSpace 隔离)
    → ...
  → MainLoop 合成最终输出
```

**TaskManagerAgent 核心能力（参照 AgentOrchestra / AutoGen / CrewAI）**:

| 能力 | 说明 | 灵感来源 |
|------|------|----------|
| LLM 拆分 | 用 LLM 分析推理输出，生成子任务+依赖图 | AgentOrchestra Planning Agent |
| 任务注册 | 所有任务持久化到 TaskRegistry + MemoryPool | AutoGen Orchestrator |
| 依赖调度 | 只派遣依赖已完成的任务 | LangGraph Supervisor |
| Worker 分配 | MoE 路由匹配最佳 Worker Agent | AutoGen Orchestrator-Worker |
| 质量评估 | 四维评分，接受/丢弃/重试 | CrewAI Manager |
| 分支空间 | 每个任务独立 BranchSpace，完成后提交主线 | Git Branching |
| 自适应重规划 | 失败任务用 LLM 重新规划替代方案 | AgentOrchestra Adaptive |

### 4.1 TaskRegistry（任务注册表）

```python
class TaskRegistry:
    """所有任务的单一来源，持久化到 MemoryPool"""
    
    async def register(scope, priority, dependencies, ...) -> ManagedTask
    def get(task_id) -> ManagedTask | None
    def get_ready() -> list[ManagedTask]       # 依赖已完成的任务
    def all_tasks() -> list[ManagedTask]
    def stats() -> dict                         # pending/running/done/failed
```

### 4.2 ManagedTask（任务数据结构）

```python
@dataclass
class ManagedTask:
    task_id: str
    parent_id: str | None
    scope: str                    # 任务范围描述
    priority: int                 # 1-5
    dependencies: list[str]       # 依赖的任务 ID 列表
    required_tools: list[str]     # 需要的工具白名单
    
    # 生命周期
    status: TaskStatus            # pending → running → done/failed/cancelled
    assigned_agent_id: str | None
    result: TaskResult | None
    evaluation: EvaluationResult | None
    
    # 重试 + 分支
    retry_count: int
    max_retries: int
    branch_id: str | None         # BranchSpace 标识
    error: str | None
```

### 4.3 TaskManagerAgent 方法

```python
class TaskManagerAgent:
    """中心编排 Agent，管理完整任务生命周期"""
    
    async def decompose(reasoning, original) -> list[ManagedTask]
        # LLM 拆分 → 注册 TaskRegistry → 解析依赖边
    
    async def dispatch_ready() -> list[str]
        # 获取 ready 任务 → MoE 路由匹配 Worker → 异步执行
    
    async def _execute_worker(task, agent) -> None
        # BranchSpace init → agent.run → evaluate → accept/retry/fail → commit
    
    async def collect_all(timeout) -> dict[str, TaskResult]
        # 等待所有 in-flight → 自动派遣新 ready → 收集结果
    
    async def replan(failed_task) -> list[ManagedTask] | None
        # LLM 重新规划失败任务 → 生成替代子任务
    
    async def cancel(task_id) / cancel_all()
        # 取消任务 + 销毁关联 Agent
```

### 4.4 BranchSpace（Agent 分支空间）

```python
class BranchSpace:
    """每个任务执行的隔离工作空间"""
    task_id: str
    agent_id: str
    base_dir: Path               # 临时文件目录
    memory_snapshot: dict         # 任务开始时的记忆快照(只读)
    execution_log: list           # 执行步骤日志
    artifacts: dict               # 产出物(代码、报告等)

    async def init(memory)
        # 创建临时目录 + 拉取记忆快照

    async def commit(memory)
        # 接受结果时: 将产出物写入主记忆 episode

    async def cleanup()
        # 清理: 删除临时文件
```

**流程**: `init()` → agent 执行 → 评估通过则 `commit()` → `cleanup()`

---

## 五、系统总体流程示例

```
用户: "修复 bot i-xxx 不回复的问题"

MainLoop:
  1. INPUT → 解析: bot修复请求, 实例 i-xxx
  2. RETRIEVE → Memory Pool 检索:
     - 找到 server config: IP, SSH key
     - 找到 playbook: fix-bot.md
     - 找到最近相关 episode
  3. REASON → 深度推理(N=3 轮内循环):
     内循环1: 分析问题类型 → bot 不回复的 5 种常见原因
     内循环2: 缩小范围 → 结合上下文, 最可能是 Gateway 或飞书插件
     内循环3: 生成方案 → SSH 诊断 → 根据诊断结果修复
  4. DECOMPOSE → TaskTree:
     T0: 修复 bot i-xxx ← Agent α(主)
       T1: SSH 诊断 ← Agent β
       T2: 如果诊断结果=Foo → 修复方案 A ← Agent γ
       T3: 如果诊断结果=Bar → 修复方案 B ← Agent δ
  5. DISPATCH → AgentRouter.route():
     - T1 匹配到 ops-diagnose 专家 → Agent β
     - T2/T3 等待 T1 结果
  6. EXECUTE:
     Agent β 循环:
       执行 SSH 诊断 → 发现 Gateway 进程停止
       自评: 完整性✓ 正确性✓ → 提交
     Agent β 销毁
     → T1 完成 → T2 触发
     Agent γ 循环:
       执行 Gateway 重启 → 验证 bot 回复正常
       自评: 完整性✓ 正确性✓ → 提交
     Agent γ 销毁
  7. COLLECT → Evaluator 评估:
     Agent β 结果: 0.95 → 接受
     Agent γ 结果: 0.90 → 接受
  8. OUTPUT → 合成回复 → 写入 Memory Pool(新 Episode) → 返回用户
```

---

## 六、部署架构

### 6.1 单机模式 (pip install)

```bash
pip install agent-loop
agent-loop init           # 初始化配置 + 启动 SurrealDB
agent-loop start          # 启动 Loop Engine
agent-loop web            # 启动 Web UI (可选)
```

### 6.2 平台模式 (Docker Compose)

```yaml
services:
  surrealdb:
    image: surrealdb/surrealdb:latest
    volumes:
      - ./data:/data
    command: start --user root --pass root file:/data/agent_loop.db

  loop-engine:
    build: .
    environment:
      - SURREAL_URL=ws://surrealdb:8000
      - LLM_PROVIDER=deepseek
      - LLM_API_KEY=${LLM_API_KEY}
    ports:
      - "8080:8080"
    depends_on:
      - surrealdb

  agent-worker:
    build: .
    command: agent-loop worker
    environment:
      - SURREAL_URL=ws://surrealdb:8000
    deploy:
      replicas: 5  # 水平扩展 Agent Worker
```

### 6.3 目录结构

```
agent-loop/
├── ARCHITECTURE.md          # 本文档
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── loop_engine/         # Loop Engine 核心
│   │   ├── main_loop.py     # MainLoop 实现
│   │   ├── agent_loop.py    # AgentLoop 实现
│   │   ├── tool_loop.py     # ToolLoop 实现
│   │   └── deep_reason.py   # 深度推理 (模型潜空间 + Agent级)
│   │
│   ├── memory/              # Memory Pool
│   │   ├── pool.py          # 统一记忆池接口
│   │   ├── graph_store.py   # 图存储 (SurrealDB)
│   │   ├── graph_route.py   # 图路由检索算法
│   │   ├── vector_store.py  # 向量索引
│   │   └── schema.surql     # SurrealDB Schema
│   │
│   ├── agent/               # Agent 管理系统
│   │   ├── lifecycle.py     # Agent 生命周期
│   │   ├── pool.py          # AgentPool
│   │   ├── router.py        # Agent 专家路由 (MoE)
│   │   ├── evaluator.py     # 质量评估
│   │   ├── factory.py       # Agent 工厂
│   │   └── discard.py       # 遗弃池
│   │
│   ├── task/                # Task 管理系统
│   │   ├── task_tree.py     # TaskTree 数据结构
│   │   ├── scheduler.py     # TaskScheduler
│   │   ├── decomposer.py    # 任务分解
│   │   └── branch_space.py  # 分支空间
│   │
│   ├── tools/               # 工具生态
│   │   ├── base.py          # ToolInterface 基类
│   │   ├── registry.py      # 工具注册表
│   │   ├── ssh.py           # SSH 工具
│   │   ├── web.py           # Web 工具
│   │   ├── code.py          # 代码工具
│   │   ├── file.py          # 文件工具
│   │   └── mcp.py           # MCP 协议适配
│   │
│   ├── web/                 # Web UI + API
│   │   ├── api.py           # REST API
│   │   ├── ws.py            # WebSocket 实时推送
│   │   └── ui/              # 前端 (可选)
│   │
│   └── cli.py               # 命令行入口
│
├── tests/                   # 测试
├── examples/                # 示例
└── docs/                    # 文档
```

---

## 七、开发计划

| 阶段 | 内容 | 优先级 | 状态 |
|------|------|--------|------|
| Phase 0 | SurrealDB Schema + Memory Pool 基础实现 | P0 | ✅ Complete |
| Phase 1 | Loop Engine (MainLoop + AgentLoop + ToolLoop) | P0 | ✅ Complete |
| Phase 2 | Agent 生命周期管理 (Factory + Pool + Router + Evaluator) | P0 | ✅ Complete |
| Phase 3 | TaskTree + TaskScheduler + BranchSpace | P1 | ✅ Complete |
| Phase 4 | 图路由检索算法 (graph_route.py) | P1 | ✅ Complete |
| Phase 5 | 深度推理混合模式 (deep_reason.py) | P1 | ✅ Complete |
| Phase 6 | 工具生态 (SSH + Web + Code + File + MCP) | P2 | ✅ Complete |
| Phase 7 | Web UI + API | P2 | ✅ Complete |
| Phase 8 | Docker 部署 + 文档 | P2 | ✅ Complete |
| Phase 9 | pip install 单机部署 | P2 | ✅ Complete |
| Phase 10 | PermissionChecker 权限系统 | P0 | ✅ Complete |
| Phase 11 | SandboxManager 沙盒执行 | P0 | ✅ Complete |
| Phase 12 | Agent 自主修改 + 用户切换 | P1 | ✅ Complete |

---

## 八、新增模块详解

### 8.1 PermissionChecker — 权限系统

**文件**: `src/permissions.py` + `config/permissions.json`

**设计**: 方案 C — 角色模板 + Agent 进化可申请提权 + 高危永远需要确认

**角色模板**:
| 模板 | trust_level | shell | 网络 | agent_ops |
|------|-------------|-------|------|-----------|
| coder | restricted | git/python3/pytest/pip | pypi.org/api.github.com | 可改 own_soul |
| researcher | untrusted | 禁止 | 允许所有 | 可改 own_soul |
| ops | trusted | systemctl/docker/kubectl/ssh | 允许所有 | 默认权限 |
| admin | admin | 全部允许 | 允许所有 | 创建+销毁 agent |

**API**:
```python
class AgentPermissions:
    can_read(path) → bool
    can_write(path) → bool
    can_execute(command) → bool
    can_access_host(host) → bool
    can_modify_file(file_type) → bool  # identity/role/journal/knowledge/profile
    request_elevation(reason) → bool

class PermissionChecker:
    get_permissions(agent_id, template) → AgentPermissions
    check_operation(agent_id, operation, **kwargs) → bool
```

**提权机制**:
- `request_elevation(reason)` — Agent 可申请提权
- 需 InteractionHub 人类确认
- 条件: min_tasks_for_elevation=10, min_success_rate=0.7

### 8.2 SandboxManager — 沙盒执行

**文件**: `src/sandbox.py`

**分级**:
| Level | trust_level | 隔离方式 | shell | 网络 |
|-------|-------------|----------|-------|------|
| 0 | untrusted | RestrictedPython / subprocess | 禁止 | 禁止 |
| 1 | restricted | subprocess + cwd 限制 | 白名单 | 有限 |
| 2 | trusted | subprocess + timeout | 白名单 | 有限 |
| 3 | admin | 直接执行 + 审批 | 全部 | 全部 |

**API**:
```python
class SandboxManager:
    async execute_code(code, language, permissions, interaction_hub) → dict
    async execute_command(command, permissions, interaction_hub) → dict
```

不依赖 firejail/nsjail，使用 subprocess + cwd + timeout 做基础隔离。

### 8.3 Agent 自主修改 (AgentSoul.self_modify)

**文件**: `src/agent_soul.py` (扩展)

Agent 可以修改自己的文件（需权限检查）：
- `self_modify(file_type, new_content, permissions)` — 修改 IDENTITY/ROLE/JOURNAL/KNOWLEDGE/profile
- `request_safety_change(suggestion)` — 建议 SAFETY.md 修改（不能直接改）
- 全部操作有 audit log 写入 JOURNAL.md

**可修改类型**:
| file_type | 文件 | 默认权限 |
|-----------|------|----------|
| identity | IDENTITY.md | ✅ coder/researcher/admin |
| role | ROLE.md | ✅ coder/researcher/admin |
| journal | JOURNAL.md | ✅ 全部模板 |
| knowledge | KNOWLEDGE.md | ✅ coder/researcher/admin |
| profile | profile.json | ❌ 默认禁止 |

### 8.4 Agent 切换 (AgentManagerAgent)

**文件**: `src/system_agents.py` (扩展)

新增方法:
- `list_agents()` — 列出所有 persistent agents
- `get_active_agent()` — 获取当前 active agent
- `switch_agent(agent_id)` — 切换 active agent 上下文

切换流程:
1. 保存当前 agent 的对话上下文
2. 加载目标 agent 的上下文
3. 更新 `state/session.json`

---

*本文档随开发持续更新。每个模块实现前需回顾此文档，确保与整体设计一致。*
