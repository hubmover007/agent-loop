# Design: PromptSkill Manager Agent + Learning Loop + MCP/Plugin

> **状态**: 设计阶段（不急于实现）
> **日期**: 2026-06-30
> **作者**: 主人理念 → 小白整理

## 一、核心理念（主人原话解读）

### 1.1 Prompt 和 Skill 的本质

| 常见误解 | 主人的理解 |
|---------|----------|
| Prompt 是指令文本 | Prompt 是**行为模式**，agent 以人类视角学习审美、规范、判断 |
| Skill 是工具说明 | Skill 是**经验结晶**——有工作流、有代码、有场景经验 |
| 核心是"怎么组合" | 核心是**agent 如何学习使用这些武器** |
| 用代码路由就够了 | 需要 **LLM Agent 来管理和分发** prompt/skill，它理解任务是否需要 |

### 1.2 为什么代码路由不够

```
代码路由（呆板）:
  if task.type == "code": skill = ["code-debug", "web-search"]
  elif task.type == "ops": skill = ["ssh", "exec"]

Agent 路由（智能）:
  PromptSkillManager Agent 收到任务描述
  → 用 LLM 理解任务意图、场景、约束
  → 从记忆中回忆"上次类似任务用了什么，效果如何"
  → 判断这个任务需要哪些 prompt 模式 + skill 组合
  → 分发给执行 agent
  → 执行后收集结果，反思，沉淀到记忆
```

代码路由是确定性的、静态的；Agent 路由是**非确定性的、学习型的**。

### 1.3 Loop Engine 思维

Prompt+Skill 的使用本身就是一个 Loop Engine：
```
理解任务 → 选择 prompt+skill → 分发执行 → 记录过程 → 反思效果 → 沉淀到记忆 → 下次更好
   ↑                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

---

## 二、论文调研支撑

### 2.1 Loop Engineering（2026年6月，新概念）

来源：Addy Osmani / Peter Steinberger / Boris Cherny

> "Prompt phrasing stopped being the bottleneck in early 2026.
> What replaced it is loop design: **the trigger, the topology, the verifier,
> and the stop rules** that decide what an agent does next and when it quits."

核心要素：
- **Trigger** — 什么唤醒 agent（cron/webhook/另一个agent）
- **Topology** — agent 如何连接（线性/并行/层级）
- **Verifier** — 如何验证结果（不只是"完成了"，而是"做好了"）
- **Stop rules** — 何时退出（避免无限循环）

→ 我们的 Loop Engine 设计完全符合这个框架。

### 2.2 SkillRouter 论文（阿里，2026年4月）

关键发现：
- 80K skill 库中，**隐藏 skill body（只看 name+description）导致 31-44% 路由准确率下降**
- 全文检索 > 元数据检索
- 1.2B 参数的 retrieve-and-rerank 管线达到 74% Hit@1
- 路由准确率提升 → 端到端任务成功率提升

→ 启示：PromptSkillManager 需要看 skill 全文（不只是描述）来做路由决策。

### 2.3 MemRL（2026年1月，自我进化）

> "Apply reinforcement learning directly to episodic memory at runtime,
> without touching model weights."

核心：
- 稳定认知推理 + 动态情节记忆 **解耦**
- 运行时通过 RL 更新记忆，不改模型权重
- 稳定性 vs 可塑性平衡

→ 启示：Prompt/Skill 的进化不改模型，改记忆。

### 2.4 MCP + A2A 协议融合（2026年趋势）

```
MCP (Model Context Protocol) — Anthropic — 工具/数据访问
A2A (Agent-to-Agent) — Google — 多agent协作
ACP (Agent Communication Protocol) — IBM — agent通信
→ 2026 Q3 预期联合互操作规范
```

两层架构成为参考模型：
- **下层 MCP**: agent ↔ tools/data
- **上层 A2A**: agent ↔ agent

### 2.5 Skill Scaling Laws（2026年5月）

> "Routing accuracy decays logarithmically with library size"
> "errors progress from local skill competition to cross-family drift
> to capture by overly general 'black-hole skills'"

→ Skill 库越大，路由越难，需要智能路由而非全量注入。

---

## 三、架构设计

### 3.1 三个固定系统 Agent（更新）

| Agent | 职责 | 灵感来源 |
|-------|------|---------|
| **TaskAgent** | 管任务：拆分、注册、调度、状态 | 已有 |
| **AgentManagerAgent** | 管 Agent：创建、分配、评估、销毁 | 已有 |
| **PromptSkillManager**（新） | 管 Prompt+Skill：路由、分发、学习、进化 | 主人理念 + MemRL + SkillRouter |

### 3.2 PromptSkillManager Agent 设计

```
┌─────────────────────────────────────────────────────┐
│           PromptSkillManager Agent                   │
│           (第三个固定系统 Agent)                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌───────────┐ │
│  │  Understand  │   │   Route     │   │  Learn    │ │
│  │  理解任务    │──→│  路由分发   │──→│  学习沉淀 │ │
│  │              │   │             │   │           │ │
│  │ • LLM分析   │   │ • 选prompt  │   │• 记录效果 │ │
│  │   任务意图  │   │ • 选skill   │   │• 反思改进 │ │
│  │ • 识别场景  │   │ • 选agent   │   │• 进化prompt│ │
│  │ • 判断约束  │   │   类型      │   │• 更新skill │ │
│  └─────────────┘   │             │   │  有效性   │ │
│                    │  非代码路由  │   │           │ │
│                    │  LLM理解匹配 │   │           │ │
│                    └─────────────┘   └───────────┘ │
│                                                      │
│  记忆：                                              │
│  • Prompt 版本历史（v1→v2→v3，带性能指标）            │
│  • Skill 有效性分数（per scenario）                  │
│  • 经验记录（每次使用的完整遥测）                     │
│  • 反思日志（什么有效，什么没用）                     │
└─────────────────────────────────────────────────────┘
```

### 3.3 核心工作流

```python
# 伪代码——不急于实现，先理解流程

class PromptSkillManager:
    """第三个固定系统 Agent：管理 Prompt + Skill 的全生命周期。"""

    async def route(self, task: Task) -> RoutingDecision:
        """理解任务 → 路由 prompt+skill → 分发给 agent"""

        # 1. 理解（LLM 分析，非代码匹配）
        understanding = await self._understand_task(task)
        # → {intent: "fix_bug", scenario: "auth_module", constraints: [...]}

        # 2. 回忆（从记忆中找类似经验）
        past = await self._recall_similar(understanding)
        # → [上次修 auth bug 用了 code-debug + memory-search, 效果 0.85]

        # 3. 判断（LLM 决策，非 if-else）
        decision = await self._llm_route(understanding, past)
        # → RoutingDecision(
        #     agent_role="code",
        #     prompt_version="code_v3",  # 进化后的版本
        #     skills=["code-debug", "memory-search"],  # 只选有效的
        #     rationale="auth模块的bug，code-debug历史效果0.85"  # 决策理由
        #   )

        return decision

    async def learn(self, experience: ExperienceRecord):
        """执行后 → 学习 → 沉淀到记忆"""

        # 1. 记录到情节记忆
        await self._record_episode(experience)

        # 2. 反思（LLM 分析什么有效什么没用）
        reflection = await self._reflect(experience)

        # 3. 更新 skill 有效性
        await self._update_skill_effectiveness(experience, reflection)

        # 4. 判断 prompt 是否需要进化
        if self._should_evolve_prompt(experience.agent_role):
            new_version = await self._evolve_prompt(
                experience.agent_role,
                reflection
            )
            # 新 prompt 版本取代旧版本

        # 5. 沉淀到长期记忆（Memory Pool 的 project 层）
        await self._sediment_to_memory(experience, reflection)
```

### 3.4 Prompt 进化机制

Prompt 不是静态的，它通过经验进化：

```
code_v1 (初始版本)
  │ 用了10次, 成功率 40%, 平均质量 0.55
  │ → 触发进化
  ▼
code_v2 (反思后改进)
  │ "改进点：增加'先理解上下文再修改'的约束"
  │ 用了15次, 成功率 75%, 平均质量 0.78
  │ → 触发进化（微调）
  ▼
code_v3 (再次改进)
  │ "改进点：增加错误恢复步骤"
  │ 用了20次, 成功率 88%, 平均质量 0.85
  │ → 稳定，不进化
  ▼
code_v3 (活跃版本)
```

进化触发条件：
- 成功率 < 50% over 10+ uses
- 平均质量 < 0.6 over 10+ uses
- Agent 反思建议改进
- 用户手动触发

进化方式：
- LLM 分析失败案例 → 生成改进版 prompt
- A/B 测试新旧版本
- 胜出版本成为活跃版本，旧版本退役（但保留历史）

### 3.5 Skill 有效性学习

每个 skill 在每个场景下有独立的有效性分数：

```
code-debug:
  scenario: code/fix_bug → effectiveness 0.85 (用了10次, 9次成功)
  scenario: code/refactor → effectiveness 0.60 (用了5次, 3次成功)
  scenario: research → effectiveness 0.10 (用了2次, 0次成功)
  → PromptSkillManager 学到：code-debug 适合 fix_bug，不适合 research

web-search:
  scenario: research → effectiveness 0.90
  scenario: code/fix_bug → effectiveness 0.20
  → PromptSkillManager 学到：web-search 适合 research
```

关键：这不是代码规则，是 **PromptSkillManager Agent 通过 LLM 理解**这些数据，做出智能决策。

---

## 四、MCP + Plugin 支持

### 4.1 协议分层

```
┌─────────────────────────────────────┐
│        A2A (Agent ↔ Agent)          │  ← 多 agent 协作
├─────────────────────────────────────┤
│   PromptSkillManager                │  ← 路由层
├─────────────────────────────────────┤
│        MCP (Agent ↔ Tools/Data)     │  ← 工具访问
├─────────────────────────────────────┤
│   Plugin System (扩展机制)           │  ← 自定义集成
├─────────────────────────────────────┤
│   Agent-Loop Core (Loop Engine)     │  ← 执行引擎
└─────────────────────────────────────┘
```

### 4.2 MCP 集成设计

```python
class MCPIntegration:
    """MCP 集成：动态工具发现 + 运行时调用。

    MCP 核心价值：
    1. 运行时发现工具（不预注入）
    2. 标准化工具接口
    3. 多传输协议（stdio, SSE, HTTP, WebSocket）
    4. 身份/策略/安全控制
    """

    async def discover_tools(self, context: dict) -> list[Tool]:
        """运行时从 MCP server 发现可用工具。

        与 PromptSkillManager 配合：
        1. PromptSkillManager 理解任务
        2. 判断需要什么能力
        3. 从 MCP 发现匹配的工具
        4. 注入给执行 agent
        """
        # MCP Gateway 根据 identity/environment/policy 返回工具
        tools = await self.mcp_gateway.list_tools(context)
        return tools

    async def invoke_tool(self, tool_id: str, params: dict) -> Any:
        """调用 MCP 工具"""
        return await self.mcp_gateway.call_tool(tool_id, params)
```

### 4.3 Plugin 系统

```python
class PluginSystem:
    """插件系统：扩展 Agent-Loop 能力。

    灵感来源：
    - Claude Code: 4种扩展（Hooks/Skills/Plugins/MCP）, 10种组件类型
    - OpenClaw: Plugin manifest, SDK
    - VSCode: Extension API

    Agent-Loop 插件类型：
    1. Tool Plugin — 新增工具（如 feishu-api, aliyun-sdk）
    2. Skill Plugin — 新增 skill 包（如 ops-playbook, code-review-guide）
    3. Prompt Plugin — 新增 prompt 模板（如 api-design-prompt, security-audit-prompt）
    4. Memory Plugin — 记忆扩展（如 vector-index, graph-index）
    5. Agent Plugin — 新增 agent 角色（如 data-analyst, devops-engineer）
    6. Hook Plugin — 生命周期钩子（before_task, after_task, on_error）
    """

    def register_plugin(self, plugin: Plugin) -> None:
        """注册插件"""
        for component in plugin.components:
            if component.type == "tool":
                self.tool_registry.register(component)
            elif component.type == "skill":
                self.skill_registry.register(component)
            elif component.type == "prompt":
                self.prompt_registry.register(component)
            # ...
```

### 4.4 协议使用策略

| 场景 | 协议 | 理由 |
|------|------|------|
| Agent ↔ 内部 Worker | asyncio Task | 零开销 |
| Agent ↔ 外部 CLI（Codex/Claude） | ACP | OpenClaw 已支持 |
| Agent ↔ MCP Server | MCP | 标准化工具访问 |
| Agent ↔ Agent（跨系统） | A2A | 2026 Q3 联合规范 |
| Plugin 扩展 | 自定义 manifest | 灵活 |

---

## 五、实现 TODO（不急于敲代码）

### Phase 1: PromptSkillManager Agent（核心）
- [ ] 理解模块：LLM 分析任务意图、场景、约束
- [ ] 路由模块：LLM 决策选 prompt+skill（非代码路由）
- [ ] 学习模块：记录经验 → 反思 → 更新有效性
- [ ] 进化模块：prompt 版本管理 + A/B 测试 + 自动进化
- [ ] 记忆模块：情节记忆 + 长期沉淀
- [ ] 集成到 MainLoop（作为第三个固定 Agent）

### Phase 2: Skill 全文检索
- [ ] Skill 不只存描述，存全文（SkillRouter 论文启示）
- [ ] 检索时用全文匹配，注入时用摘要（lazy-load）
- [ ] Skill 版本管理（skill 也能进化）

### Phase 3: MCP 集成
- [ ] MCP Client（连接外部 MCP Server）
- [ ] MCP Gateway（身份/策略/安全）
- [ ] 运行时工具发现（不预注入）
- [ ] 工具调用遥测（记录到学习系统）

### Phase 4: Plugin 系统
- [ ] Plugin manifest 格式（YAML）
- [ ] 6 种插件类型（Tool/Skill/Prompt/Memory/Agent/Hook）
- [ ] 生命周期钩子（before_task/after_task/on_error）
- [ ] 插件市场（未来）

### Phase 5: A2A 协议
- [ ] Agent 间通信（跨进程/跨机器）
- [ ] Agent 能力广播
- [ ] 任务委托协议

### Phase 6: 评估与可视化
- [ ] Prompt 版本性能对比
- [ ] Skill 有效性热力图（scenario × skill）
- [ ] 学习曲线（随时间质量提升）
- [ ] MCP 工具使用统计

---

## 六、关键设计决策

1. **PromptSkillManager 是 LLM Agent，不是代码路由器**
   - 它用 LLM 理解任务，用记忆做决策，用学习改进
   - 不用 if-else，不用规则引擎

2. **Prompt 和 Skill 都能进化**
   - Prompt 通过版本管理进化
   - Skill 通过经验积累进化（有效性分数 per scenario）

3. **学习是 Loop Engine，不是一次性操作**
   - 每次任务执行都触发学习循环
   - 知识沉淀到记忆，不只是更新分数

4. **MCP 是运行时发现，不是预注入**
   - 避免上下文爆炸
   - 工具按需发现

5. **Plugin 是一等公民**
   - 所有扩展通过 plugin 机制
   - 不硬编码任何工具/skill/prompt

6. **不急于实现，先理解理念**
   - 先把设计文档完善
   - 确认理念正确后再逐步实现
   - 每个模块独立测试，确保符合设计

---

## 七、参考

- [Loop Engineering (Osmani, 2026)](https://agentshortlist.com/articles/loop-engineering)
- [SkillRouter (Alibaba, 2026)](https://arxiv.org/abs/2603.22455)
- [MemRL (2026)](https://arxiv.org/abs/2601.03192)
- [MCP + A2A Convergence (2026)](https://zylos.ai/research/2026-03-26-agent-interoperability-protocols-mcp-a2a-acp-convergence/)
- [Skill Scaling Laws (2026)](https://arxiv.org/abs/2605.16508)
- [Dive into Claude Code (2026)](https://github.com/VILA-Lab/Dive-into-Claude-Code)
- [OpenClaw Skills System](https://docs.openclaw.ai/tools/skills)
