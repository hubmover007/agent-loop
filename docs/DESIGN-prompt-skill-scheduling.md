# Design: Prompt Engineering + Skill Library + Free Scheduling

> **状态**: 设计阶段（TODO，待实现）
> **日期**: 2026-06-30
> **优先级**: 高

## 问题

Agent-Loop 当前有：
- ✅ 外部 Agent 调度（Codex/Claude Code/Gemini）
- ✅ 内部 Worker Agent
- ✅ OpenClaw Skill 桥接（22个 skill 作为 tool）

但缺少：
- ❌ **动态 Prompt 组合** — 所有 agent 用同一个 system prompt
- ❌ **Skill 智能路由** — skill 作为 tool 注册，但 LLM 不知道何时用哪个
- ❌ **Agent + Skill 联合调度** — agent 和 skill 独立决策，没协同

## 调研结论

### 主流方案

| 方案 | 核心思路 | 可借鉴 |
|------|---------|--------|
| CrewAI | Role+Goal+Backstory 模板，skill 注入 task prompt | 模板系统 |
| OpenClaw | skill 列表压缩注入，lazy-load SKILL.md | lazy-load 机制 |
| MCP | 运行时动态发现工具 | 工具发现协议 |
| CUA-Skill | 参数化执行图 + 组合规则 | skill 组合性 |
| Meta Context Engineering | skill 自我进化 | 动态 skill 演化 |

### 关键论文发现
- "When Single-Agent with Skills Replace Multi-Agent Systems" — skill 库规模和 agent 数量的相变点
- "Confidence-Aware Routing" — 按任务复杂度动态选 agent role + model scale

## 设计：三层动态架构

### Layer 1: Prompt Composer（提示词组合器）

```python
class PromptComposer:
    """动态组合 system prompt for each task."""

    def compose(self, agent: Agent, task: Task, skills: list[Skill],
                context: MemoryContext) -> str:
        return self.template.render(
            role=agent.role_prompt,           # "你是一个 Python 专家"
            goal=task.scope,                  # "修复 auth.py 的空指针"
            skills=[s.compact_desc for s in skills],  # skill 摘要列表
            context=context.summary,          # 从 UnifiedRetriever 获取
            constraints=agent.constraints,     # "不要修改配置文件"
        )
```

**Prompt 结构**：
```
[Base Prompt]
  你是 {role}。当前任务：{goal}

[Skill Prompt] (lazy-load, 只注入摘要)
  可用技能：
  - feishu-doc: 创建/读取飞书文档
  - web-search: 网页搜索
  - code-debug: 代码调试
  需要时用 read 加载完整 SKILL.md

[Context Prompt]
  相关记忆：
  {context from UnifiedRetriever}

[Constraints]
  {agent.constraints}
```

### Layer 2: Skill Router（技能路由器）

```python
class SkillRouter:
    """根据任务需求匹配最佳 skill 组合。"""

    def route(self, task: Task) -> list[Skill]:
        # 1. 关键词匹配
        matched = self._keyword_match(task.scope)

        # 2. 能力匹配
        caps = self._extract_required_caps(task.scope)
        matched += self._capability_match(caps)

        # 3. 去重 + 排序（按相关度）
        return self._rank(matched, task.scope)
```

**Skill 元数据格式**：
```yaml
name: feishu-doc
description: 创建/读取/更新飞书云文档
capabilities: [document, feishu, write]
triggers:
  keywords: [飞书文档, 创建文档, 读取文档, doc]
  patterns: ["创建.*文档", "读取.*doc"]
compact_desc: "feishu-doc: 创建/读取飞书文档"
full_path: skills/feishu-doc/SKILL.md
version: "1.0"
```

### Layer 3: Agent + Skill 联合调度

```python
class JointScheduler:
    """Agent 和 Skill 联合路由。"""

    async def schedule(self, task: Task) -> Schedule:
        # 1. 分析任务
        analysis = await self.llm.analyze(task.scope)
        # → {type: "coding", caps: ["code", "debug"], complexity: "high"}

        # 2. 联合匹配
        agent_type = self._match_agent(analysis)  # internal / codex / claude
        skills = self.skill_router.route(task)

        # 3. 组合 prompt
        prompt = self.prompt_composer.compose(
            agent=agent, task=task, skills=skills, context=context
        )

        # 4. 分派
        return Schedule(agent_type=agent_type, skills=skills, prompt=prompt)
```

**联合决策矩阵**：

| 任务类型 | 推荐 Agent | 推荐 Skill | Prompt 策略 |
|---------|-----------|-----------|------------|
| 编码/调试 | Codex / Claude Code | code-debug, web-search | 简洁 + 代码上下文 |
| 飞书操作 | 内部 Worker | feishu-doc/sheet/task | 飞书 API 指南 |
| 运维 | 内部 Worker | ssh, exec, memory-search | 操作手册 + 安全约束 |
| 分析/研究 | Claude Code | web-search, web-fetch | 详细 + 多角度 |
| 多模态 | Gemini | image-generate, video-generate | 创意引导 |

## 实现计划（TODO）

### Phase 1: Skill 元数据系统（1-2天）
- [ ] 定义 SkillManifest 格式（YAML）
- [ ] 为 22 个 OpenClaw skill 编写 manifest
- [ ] 实现 SkillRouter（关键词 + 能力匹配）
- [ ] 测试：给定任务 → 正确匹配 skill

### Phase 2: Prompt 模板系统（1-2天）
- [ ] 选择模板引擎（Jinja2 或简单字符串模板）
- [ ] 定义 prompt 模板（base + skill + context + constraints）
- [ ] 实现 PromptComposer
- [ ] 测试：不同任务 → 不同 prompt 结构

### Phase 3: 联合调度器（2-3天）
- [ ] 实现 JointScheduler（agent + skill 联合决策）
- [ ] 集成到 AgentManagerAgent.assign()
- [ ] 集成到 ExternalAgentBridge.dispatch()（外部 agent 也收到组合 prompt）
- [ ] 测试：端到端，任务 → 联合调度 → 执行

### Phase 4: 动态 Skill 演化（未来）
- [ ] Skill 质量评估（哪些 skill 被频繁使用？哪些从不使用？）
- [ ] Skill 自动生成（从成功任务中提取新 skill）
- [ ] Skill 版本管理（A/B 测试不同 prompt）
- [ ] MCP 集成（运行时动态发现外部工具）

### Phase 5: 外部 Agent Skill 传递（未来）
- [ ] 外部 Agent（Codex/Claude）收到组合后的 prompt + skill 描述
- [ ] 外部 Agent 可以请求加载完整 SKILL.md
- [ ] 跨 Agent 的 skill 执行结果回收

## 关键设计决策

1. **Lazy-load 而非全量注入** — 参考 OpenClaw，只注入 skill 摘要，用到才加载全文
2. **Skill 作为一等公民** — 不只是 tool，有元数据、触发条件、能力声明
3. **联合路由 > 独立路由** — Agent 和 Skill 一起决策，不是先选 agent 再选 skill
4. **模板化 prompt** — 可审计、可 A/B 测试、可版本管理
5. **外部 Agent 也能用 skill** — 组合后的 prompt 传给 Codex/Claude，它们也受益

## 参考

- [OpenClaw Skills System](https://docs.openclaw.ai/tools/skills)
- [CrewAI Skills](https://docs.crewai.com/en/concepts/skills)
- [MCP Protocol](https://github.com/modelcontextprotocol)
- [CUA-Skill Paper](https://arxiv.org/html/2602.12430v4)
- [Meta Context Engineering](https://github.com/muratcankoylan/agent-skills-for-context-engineering)
```
