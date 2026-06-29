# Design: Borrowing from OpenClaw + Claude Code

> **状态**: 调研完成，待实现
> **日期**: 2026-06-30
> **参考**: Claude Code v2.1.88 (~512K行), OpenClaw (生产级)

## 一、Claude Code 架构精华（可借鉴）

### 1.1 核心发现："98.4% Infrastructure, 1.6% AI"

Claude Code 的 agent loop 是一个简单的 while-loop。真正的工程复杂度在周围的系统：
- 权限门控（7层安全）
- 上下文管理（5层压缩）
- 工具路由（54个工具 + MCP）
- 恢复逻辑

**→ Agent-Loop 借鉴**：我们的 loop engine 也不应该过度复杂化 AI 决策，而应该投资周围的基础设施。

### 1.2 五层上下文压缩（Before every model call）

| 阶段 | 策略 | 触发条件 |
|------|------|---------|
| Budget Reduction | 每条消息大小上限 | 始终活跃 |
| Snip | 裁剪旧历史 | 特性开关 |
| Microcompact | 缓存感知细粒度压缩 | 始终（基于时间） |
| Context Collapse | 读时虚拟投影（非破坏性） | 特性开关 |
| Auto-Compact | 完整模型生成摘要（最后手段） | 所有其他方法失败时 |

**→ Agent-Loop 借鉴**：在 MainLoop 的 RETRIEVE 阶段后加入压缩管线。

### 1.3 SkillTool vs AgentTool（关键区分）

| 机制 | 代价 | 适用场景 |
|------|------|---------|
| **SkillTool** | 低（注入当前 context） | 需要在当前会话中增加能力 |
| **AgentTool** | 高（新开隔离 context，~7x tokens） | 需要隔离执行，防止 context 爆炸 |

**→ Agent-Loop 借鉴**：内部 Worker Agent 用 SkillTool 模式（注入 skill），外部 Agent 用 AgentTool 模式（隔离执行）。

### 1.4 六种内置 Agent 类型

| 类型 | 职责 | Agent-Loop 对应 |
|------|------|----------------|
| Explore | 探索代码库，返回摘要 | 研究型 Worker |
| Plan | 制定计划，不执行 | TaskAgent.decompose() |
| General-purpose | 通用执行 | 默认 Worker |
| Guide | 指导用户使用工具 | 不需要 |
| Verification | 验证其他 agent 的结果 | AgentEvaluator |
| Statusline | 状态展示 | 不需要 |

**→ Agent-Loop 借鉴**：内置 4 种 Worker Agent 角色（Explore, Code, Verify, Research），对应不同任务类型。

### 1.5 Sidechain 机制

- 每个 subagent 写自己的 JSONL 文件
- **只有摘要返回给 parent**，完整历史不进入 parent context
- 多实例协调用 POSIX flock()（零外部依赖）

**→ Agent-Loop 借鉴**：BranchSpace 的 commit 机制已经实现了类似功能，但可以借鉴 sidechain 的 JSONL 审计日志。

### 1.6 权限不恢复

- 权限在 session resume 时**不恢复**——每次 session 重新建立信任
- Deny-first: deny > ask > allow，严格规则优先

**→ Agent-Loop 借鉴**：外部 Agent 调度时，权限按 session 隔离，不继承。

---

## 二、OpenClaw 架构精华（可借鉴）

### 2.1 Sub-agent 模式

```
Orchestrator (main agent)
├─ sessions_spawn → subagent A1 (isolated session)
│   └─ sessions_spawn → subagent A1-1 (nested)
├─ sessions_spawn → subagent A2
└─ sessions_send → Specialist B (sibling main agent)
```

- **isolated 模式**（默认）：全新 transcript，低 token
- **fork 模式**：分支当前 transcript，适合需要上下文的任务

**→ Agent-Loop 借鉴**：BranchSpace 支持 isolated 和 fork 两种模式。

### 2.2 Push-based 完成（非轮询）

- `sessions_spawn` 非阻塞，立即返回 run id
- 子 agent 完成后**自动推送结果**回 parent session
- **禁止轮询** `subagents list`——只在调试时按需检查

**→ Agent-Loop 借鉴**：AgentManagerAgent 的 _execute 已经是 async task，但需要实现 push-based 完成通知。

### 2.3 Agent Harness（执行器抽象）

OpenClaw 的 harness 是底层执行器抽象：
- PI harness（默认内置）
- Codex harness（原生 Codex app-server）
- Claude harness（Claude Code CLI）
- 自定义 harness

```
OpenClaw core (channels, sessions, tools, approvals)
    ↓
AgentHarness (执行器)
    ↓
PI / Codex / Claude / Custom
```

**→ Agent-Loop 借鉴**：我们的 ExternalAgentBridge 就是类似的抽象层，但需要更清晰地分离"执行器"和"调度器"。

### 2.4 Agent Binding（确定性路由）

OpenClaw 用 `openclaw.json` 中的 bindings 做确定性路由：
```
最具体优先: peer > parentPeer > guildId+roles > guildId > teamId > accountId > channel > default
```

**→ Agent-Loop 借鉴**：任务路由也可以用确定性 binding（按 task type / required_tools / priority 路由到特定 agent）。

---

## 三、整合方案

### 3.1 内置 Agent 角色设计

```python
class BuiltinAgentRole(Enum):
    """Built-in agent roles (inspired by Claude Code's 6 types)."""
    EXPLORE = "explore"      # 探索型：读代码/文件，返回摘要
    CODE = "code"            # 编码型：写/改代码（内部 Worker 或外部 Codex/Claude）
    VERIFY = "verify"        # 验证型：检查其他 agent 的结果
    RESEARCH = "research"    # 研究型：搜索/web fetch，返回分析
    PLAN = "plan"            # 计划型：制定执行计划（TaskAgent 的 decompose）
    OPS = "ops"              # 运维型：SSH/exec/系统操作
```

### 3.2 上下文压缩管线

```python
class CompactionPipeline:
    """5-stage context compression (inspired by Claude Code)."""

    async def compact(self, context: str, budget: int) -> str:
        # Stage 1: Budget reduction (per-message caps)
        context = self._budget_reduction(context, budget)

        # Stage 2: Snip (trim old history)
        context = self._snip(context)

        # Stage 3: Microcompact (fine-grained compression)
        context = self._microcompact(context)

        # Stage 4: Context collapse (read-time projection)
        context = self._context_collapse(context)

        # Stage 5: Auto-compact (model summary, last resort)
        if self._needs_full_compaction(context, budget):
            context = await self._auto_compact(context)

        return context
```

### 3.3 SkillTool vs AgentTool 分离

```python
class SkillInjector:
    """Inject skill into current agent context (cheap, SkillTool pattern)."""

    def inject(self, agent: Agent, skill: Skill) -> str:
        """Add skill instructions to agent's system prompt."""
        return f"{agent.system_prompt}\n\n## Skill: {skill.name}\n{skill.compact_desc}"


class AgentSpawner:
    """Spawn isolated agent for expensive tasks (AgentTool pattern)."""

    async def spawn(self, task: Task, skill: Skill) -> Agent:
        """Create new isolated agent with skill pre-loaded."""
        agent = await self.pool.acquire()
        agent.system_prompt = self._build_prompt(task, skill)
        return agent
```

### 3.4 Push-based 完成通知

```python
class AgentManagerAgent:
    """Enhanced with push-based completion (inspired by OpenClaw)."""

    async def assign(self, task: ManagedTask) -> bool:
        # ... spawn task ...
        # DON'T poll — completion will push to us
        return True

    async def _on_agent_complete(self, task_id: str, result: TaskResult):
        """Called automatically when an agent finishes (push, not poll)."""
        task = self.registry.get(task_id)
        task.result = result
        task.status = TaskStatus.DONE
        # Notify MainLoop that this task is ready
        await self._notify_main_loop(task)
```

### 3.5 Sidechain 审计日志

```python
class BranchSpace:
    """Enhanced with sidechain JSONL audit (inspired by Claude Code)."""

    def __init__(self, task_id: str, agent_id: str):
        self.audit_log = f"branches/{task_id}_{agent_id}.jsonl"

    def log_step(self, step: StepLog):
        """Append-only JSONL audit log for each branch."""
        with open(self.audit_log, "a") as f:
            f.write(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "agent": self.agent_id,
                "step": step.action,
                "result": step.result,
            }) + "\n")
```

---

## 四、实现 TODO

### Phase A: 内置 Agent 角色（2天）
- [ ] 定义 6 种 BuiltinAgentRole
- [ ] 每种角色有专属 prompt 模板
- [ ] AgentManagerAgent 根据任务类型自动选择角色
- [ ] 测试：编码任务 → CODE 角色，研究任务 → RESEARCH 角色

### Phase B: 上下文压缩管线（2天）
- [ ] 实现 CompactionPipeline（5阶段）
- [ ] 集成到 AgentLoop（model call 之前）
- [ ] 测试：长对话不爆 context

### Phase C: SkillTool / AgentTool 分离（1天）
- [ ] SkillInjector：注入 skill 到当前 context
- [ ] AgentSpawner：隔离 agent 执行
- [ ] AgentManagerAgent 决策：何时注入 vs 何时 spawn

### Phase D: Push-based 完成（1天）
- [ ] asyncio.Event 或 callback 机制
- [ ] 移除所有 poll 循环
- [ ] 测试：多 agent 并行 → 完成后自动通知

### Phase E: Sidechain 审计（1天）
- [ ] BranchSpace JSONL 日志
- [ ] 只返回摘要给 parent（保护 parent context）
- [ ] 审计日志可查询

### Phase F: 确定性路由 Binding（未来）
- [ ] 任务类型 → agent binding（类似 OpenClaw 的 channel binding）
- [ ] 优先级路由：task.required_tools > task.type > task.priority > default
- [ ] 可配置的路由规则

---

## 五、不借鉴的部分

| Claude Code 特性 | 不借鉴原因 |
|-----------------|-----------|
| 7层安全权限 | 过度复杂，Agent-Loop 面向可信环境 |
| 27个 hook 事件 | 太重，需要时再加 |
| 4种扩展机制 | 插件系统先不做，专注核心 |
| CLAUDE.md 4级层级 | 我们的 Memory Pool 已覆盖 |
| Auto-mode ML 分类器 | 需要 LLM 调用，太贵 |

| OpenClaw 特性 | 不借鉴原因 |
|-------------|-----------|
| Channel binding | 我们不是消息平台 |
| Plugin manifest 10种组件 | 太复杂 |
| Thread binding | 飞书场景才需要 |
