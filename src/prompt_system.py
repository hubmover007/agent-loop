"""Built-in Agent roles and prompt templates.

Inspired by Claude Code's 6 built-in agent types:
  - Explore: read code/files, return summaries
  - Code: write/modify code (internal Worker or external Codex/Claude)
  - Verify: check other agents' results
  - Research: search/web fetch, return analysis
  - Plan: create execution plans (used by TaskAgent)
  - Ops: SSH/exec/system operations

Each role has a specialized system prompt template that gets composed
with task context, skills, and memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BuiltinAgentRole(str, Enum):
    """Built-in agent roles."""
    EXPLORE = "explore"
    CODE = "code"
    VERIFY = "verify"
    RESEARCH = "research"
    PLAN = "plan"
    OPS = "ops"


# ============================================================
# Role Prompt Templates
# ============================================================

ROLE_PROMPTS: dict[BuiltinAgentRole, str] = {
    BuiltinAgentRole.EXPLORE: """你是一个代码/文件探索专家。

## 职责
- 阅读和理解代码库结构
- 查找相关文件和函数
- 返回简洁的摘要（不要返回完整代码）

## 约束
- 只读操作，不修改任何文件
- 摘要控制在 500 字以内
- 标注关键文件路径和行号
""",

    BuiltinAgentRole.CODE: """你是一个高级软件工程师。

## 职责
- 编写、修改、重构代码
- 修复 bug
- 编写测试
- 代码审查

## 约束
- 遵循项目现有代码风格
- 修改前先理解上下文
- 每次修改后验证不破坏现有功能
- 提交清晰的 commit message
""",

    BuiltinAgentRole.VERIFY: """你是一个质量验证专家。

## 职责
- 验证其他 Agent 的执行结果
- 检查代码质量、安全性、性能
- 运行测试确认功能正确
- 评估结果是否满足任务要求

## 评估维度
1. 正确性：结果是否解决了问题
2. 完整性：是否遗漏了边界情况
3. 安全性：是否有安全风险
4. 可维护性：代码是否清晰

## 约束
- 必须实际运行验证，不能只看代码
- 返回 PASS / FAIL / NEEDS_REVISION
""",

    BuiltinAgentRole.RESEARCH: """你是一个技术研究员。

## 职责
- 搜索相关技术资料
- 分析不同方案的优劣
- 返回结构化的研究结论

## 约束
- 引用信息来源
- 区分事实和推测
- 提供可操作的建议
""",

    BuiltinAgentRole.PLAN: """你是一个技术项目经理。

## 职责
- 分析复杂任务，拆分为可执行的子任务
- 识别依赖关系
- 估算每个子任务的复杂度
- 制定执行顺序

## 输出格式
```
任务: <描述>
├── 子任务1 (复杂度: 低/中/高)
│   ├── 步骤1
│   └── 步骤2
├── 子任务2 (依赖: 子任务1)
└── 子任务3 (可并行)
```

## 约束
- 子任务粒度适中（单个 agent 可完成）
- 明确标注依赖关系
- 估算复杂度帮助调度器分配资源
""",

    BuiltinAgentRole.OPS: """你是一个运维工程师。

## 职责
- 执行系统操作（SSH, 重启, 部署）
- 诊断系统问题
- 执行安全检查

## 约束
- 所有操作前先备份
- 危险操作需要确认
- 记录所有操作的命令和结果
- 遵循最小权限原则
""",
}


# ============================================================
# Task Type → Agent Role Mapping
# ============================================================

# Keywords that map task scopes to agent roles
ROLE_KEYWORDS: dict[BuiltinAgentRole, list[str]] = {
    BuiltinAgentRole.EXPLORE: [
        "探索", "查找", "理解", "分析代码", "阅读",
        "explore", "find", "understand", "scan",
    ],
    BuiltinAgentRole.CODE: [
        "编写", "修改", "修复", "重构", "实现", "开发",
        "code", "write", "fix", "refactor", "implement", "debug",
    ],
    BuiltinAgentRole.VERIFY: [
        "验证", "检查", "测试", "审查", "确认",
        "verify", "check", "test", "review", "validate",
    ],
    BuiltinAgentRole.RESEARCH: [
        "研究", "调研", "搜索", "分析方案", "对比",
        "research", "search", "analyze", "compare",
    ],
    BuiltinAgentRole.PLAN: [
        "计划", "拆分", "规划", "安排", "分解",
        "plan", "decompose", "schedule", "breakdown",
    ],
    BuiltinAgentRole.OPS: [
        "部署", "重启", "运维", "SSH", "服务器",
        "deploy", "restart", "ops", "ssh", "server",
    ],
}


def match_role(task_scope: str) -> BuiltinAgentRole:
    """Match a task scope to the best agent role.

    Uses keyword matching. Returns CODE as default.
    """
    scope_lower = task_scope.lower()
    scores: dict[BuiltinAgentRole, int] = {role: 0 for role in BuiltinAgentRole}

    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in scope_lower:
                scores[role] += 1

    # Return highest scoring role, default to CODE
    best_role = max(scores, key=scores.get)
    return best_role if scores[best_role] > 0 else BuiltinAgentRole.CODE


# ============================================================
# Prompt Composer
# ============================================================

@dataclass
class ComposedPrompt:
    """A dynamically composed system prompt."""
    role: BuiltinAgentRole
    role_prompt: str
    task_scope: str
    skill_descriptions: list[str] = field(default_factory=list)
    context_summary: str = ""
    constraints: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render the full system prompt."""
        parts = [self.role_prompt]

        # Task
        parts.append(f"\n## 当前任务\n{self.task_scope}")

        # Skills (lazy-load: only descriptions, not full content)
        if self.skill_descriptions:
            parts.append("\n## 可用技能")
            for desc in self.skill_descriptions:
                parts.append(f"- {desc}")
            parts.append("（需要时用 read 加载完整技能文档）")

        # Context from memory
        if self.context_summary:
            parts.append(f"\n## 相关记忆\n{self.context_summary}")

        # Constraints
        if self.constraints:
            parts.append("\n## 约束")
            for c in self.constraints:
                parts.append(f"- {c}")

        return "\n".join(parts)


class PromptComposer:
    """Compose system prompts dynamically for each task.

    Inspired by Claude Code's context assembly:
      system prompt → role → skills → context → constraints
    """

    def compose(self, role: BuiltinAgentRole, task_scope: str,
                skill_descriptions: list[str] | None = None,
                context_summary: str = "",
                constraints: list[str] | None = None) -> ComposedPrompt:
        """Compose a system prompt for a task.

        Args:
            role: Agent role (determines base prompt)
            task_scope: Task description
            skill_descriptions: List of skill summary strings (lazy-load)
            context_summary: Relevant memory from UnifiedRetriever
            constraints: Additional constraints for this task
        """
        return ComposedPrompt(
            role=role,
            role_prompt=ROLE_PROMPTS.get(role, ROLE_PROMPTS[BuiltinAgentRole.CODE]),
            task_scope=task_scope,
            skill_descriptions=skill_descriptions or [],
            context_summary=context_summary,
            constraints=constraints or [],
        )

    def compose_for_task(self, task_scope: str,
                         skill_descriptions: list[str] | None = None,
                         context_summary: str = "",
                         constraints: list[str] | None = None) -> ComposedPrompt:
        """Auto-detect role and compose prompt."""
        role = match_role(task_scope)
        return self.compose(
            role=role,
            task_scope=task_scope,
            skill_descriptions=skill_descriptions,
            context_summary=context_summary,
            constraints=constraints,
        )


# ============================================================
# Context Compaction Pipeline (inspired by Claude Code's 5 stages)
# ============================================================

class CompactionPipeline:
    """5-stage context compression before model calls.

    Inspired by Claude Code's compaction pipeline:
      1. Budget reduction (per-message caps)
      2. Snip (trim old history)
      3. Microcompact (fine-grained compression)
      4. Context collapse (read-time projection)
      5. Auto-compact (model summary, last resort)
    """

    def __init__(self, max_context_chars: int = 100_000,
                 max_message_chars: int = 10_000):
        self.max_context = max_context_chars
        self.max_message = max_message_chars

    def compact(self, messages: list[dict]) -> list[dict]:
        """Run compaction pipeline on message list.

        Args:
            messages: List of {"role": str, "content": str} dicts

        Returns:
            Compacted message list
        """
        if not messages:
            return messages

        # Stage 1: Budget reduction (per-message caps)
        messages = self._budget_reduction(messages)

        # Stage 2: Snip (trim old history)
        messages = self._snip(messages)

        # Stage 3: Microcompact (compress tool outputs)
        messages = self._microcompact(messages)

        # Stage 4: Context collapse (summarize old turns)
        messages = self._context_collapse(messages)

        # Stage 5: Auto-compact (last resort)
        total = sum(len(m["content"]) for m in messages)
        if total > self.max_context:
            messages = self._auto_compact(messages)

        return messages

    def _budget_reduction(self, messages: list[dict]) -> list[dict]:
        """Stage 1: Cap each message size."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if len(content) > self.max_message:
                # Keep first and last portions, note truncation
                half = self.max_message // 2
                content = (
                    content[:half]
                    + f"\n\n[... truncated {len(content) - self.max_message} chars ...]\n\n"
                    + content[-half:]
                )
            result.append({**msg, "content": content})
        return result

    def _snip(self, messages: list[dict]) -> list[dict]:
        """Stage 2: Remove old messages if too many."""
        max_messages = 50  # Keep last 50 messages
        if len(messages) <= max_messages:
            return messages

        # Keep system message + last N messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        kept = other_msgs[-max_messages:]
        return system_msgs + kept

    def _microcompact(self, messages: list[dict]) -> list[dict]:
        """Stage 3: Compress verbose tool outputs."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            # Compress long tool outputs
            if msg.get("role") == "tool" and len(content) > 2000:
                # Keep first 500 chars + summary
                content = (
                    content[:500]
                    + f"\n[... {len(content) - 1000} chars compressed ...]\n"
                    + content[-500:]
                )
                msg = {**msg, "content": content}
            result.append(msg)
        return result

    def _context_collapse(self, messages: list[dict]) -> list[dict]:
        """Stage 4: Collapse old conversation turns into summary."""
        if len(messages) <= 10:
            return messages

        # Collapse early messages into a summary block
        early = messages[:5]
        recent = messages[5:]

        # Simple collapse: concatenate and truncate
        collapsed = " ".join(m.get("content", "")[:200] for m in early
                            if m.get("role") != "system")

        if collapsed:
            summary_msg = {
                "role": "system",
                "content": f"[Earlier context summary]\n{collapsed[:1000]}",
            }
            system_msgs = [m for m in recent if m.get("role") == "system"]
            other_msgs = [m for m in recent if m.get("role") != "system"]
            return system_msgs + [summary_msg] + other_msgs

        return recent

    def _auto_compact(self, messages: list[dict]) -> list[dict]:
        """Stage 5: Aggressive truncation (last resort)."""
        # Keep system messages + last 5 messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        kept = other_msgs[-5:]
        return system_msgs + kept
