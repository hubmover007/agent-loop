"""Prompt-Skill Learning System — agent self-improvement through experience.

Core Philosophy:
  - Prompt ≠ static text. Prompt = behavior pattern that agent learns and refines.
  - Skill ≠ tool description. Skill = crystallized experience (workflow + code + scenarios).
  - The key is NOT "how to combine" but "how agent learns to use weapons":
    * Did this prompt+skill combination work well?
    * Which skills are actually useful in which scenarios?
    * How to avoid failed combinations?
    * How to sediment success into memory?
  - Loop Engine thinking: use → record → reflect → sediment → improve → next time better

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │              Learning Loop (per task)                │
  │                                                      │
  │  1. Select prompt + skills (based on past experience)│
  │  2. Execute task with telemetry                      │
  │  3. Evaluate outcome (success/quality/efficiency)    │
  │  4. Record to Memory Pool (episode layer)            │
  │  5. Reflect: what worked? what didn't?               │
  │  6. Update prompt/skill effectiveness scores         │
  │  7. Sediment insights back to prompt/skill           │
  │     (prompt evolves, skill accumulates experience)   │
  └─────────────────────────────────────────────────────┘

This module implements:
  - ExperienceRecord: what happened when agent used prompt+skill
  - SkillEffectiveness: tracked per-skill effectiveness per scenario
  - PromptEvolution: prompt versions that improve over time
  - LearningMemory: integration with Memory Pool (episode layer)
  - ReflectionEngine: LLM-powered reflection on what worked
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Experience Record — the atomic unit of learning
# ============================================================

class OutcomeType(str, Enum):
    """Task outcome classification."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ExperienceRecord:
    """Record of a single prompt+skill usage on a task.

    This is the atomic unit of learning. Every time an agent uses
    a prompt+skill combination to execute a task, we record:
    - What was the task?
    - Which prompt version was used?
    - Which skills were loaded?
    - What was the outcome?
    - How efficient was it?
    - What did the agent reflect on?

    These records feed back into:
    - Skill effectiveness scoring
    - Prompt evolution
    - Memory Pool (episode layer)
    """

    record_id: str = ""
    task_id: str = ""
    task_scope: str = ""

    # What was used
    agent_role: str = ""           # "code" | "ops" | "research" | ...
    agent_id: str = ""             # internal worker id or external:codex:xxx
    prompt_version: str = ""       # prompt template version
    skills_used: list[str] = field(default_factory=list)  # skill names loaded
    skills_actually_used: list[str] = field(default_factory=list)  # skills the agent actually invoked

    # What happened
    outcome: OutcomeType = OutcomeType.SUCCESS
    quality_score: float = 0.0     # 0.0-1.0 from AgentEvaluator
    execution_time_s: float = 0.0
    token_usage: int = 0
    retry_count: int = 0

    # Reflection (agent's own assessment)
    reflection: str = ""           # What worked? What didn't? What to try next time?
    lessons_learned: list[str] = field(default_factory=list)

    # Timestamps
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "task_id": self.task_id,
            "task_scope": self.task_scope,
            "agent_role": self.agent_role,
            "agent_id": self.agent_id,
            "prompt_version": self.prompt_version,
            "skills_used": self.skills_used,
            "skills_actually_used": self.skills_actually_used,
            "outcome": self.outcome.value,
            "quality_score": self.quality_score,
            "execution_time_s": self.execution_time_s,
            "token_usage": self.token_usage,
            "retry_count": self.retry_count,
            "reflection": self.reflection,
            "lessons_learned": self.lessons_learned,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# ============================================================
# Skill Effectiveness — tracked per skill per scenario
# ============================================================

@dataclass
class SkillEffectiveness:
    """Track how effective a skill is in different scenarios.

    For each skill, we track:
    - How many times it was loaded
    - How many times it was actually used (agent invoked it)
    - Success rate when used
    - Average quality score when used
    - Which task types it works best for
    - Which task types it fails for

    This lets the agent learn "skill X is great for coding tasks
    but useless for research tasks."
    """

    skill_name: str = ""

    # Usage stats
    times_loaded: int = 0           # How many times agent received this skill
    times_actually_used: int = 0    # How many times agent actually invoked it
    times_succeeded: int = 0
    times_failed: int = 0

    # Quality stats
    quality_scores: list[float] = field(default_factory=list)  # All quality scores
    avg_quality: float = 0.0

    # Scenario breakdown: task_type → {success_count, fail_count, avg_quality}
    scenario_stats: dict[str, dict] = field(default_factory=dict)

    # Effectiveness score (0.0-1.0): higher = more useful
    effectiveness_score: float = 0.0

    # Last updated
    last_used_at: str = ""

    def record_usage(self, task_type: str, outcome: OutcomeType,
                     quality: float, actually_used: bool) -> None:
        """Record a single usage of this skill."""
        self.times_loaded += 1
        if actually_used:
            self.times_actually_used += 1
            self.quality_scores.append(quality)
            self.avg_quality = sum(self.quality_scores) / len(self.quality_scores)

            if outcome in (OutcomeType.SUCCESS, OutcomeType.PARTIAL):
                self.times_succeeded += 1
            else:
                self.times_failed += 1

        # Scenario breakdown
        if task_type not in self.scenario_stats:
            self.scenario_stats[task_type] = {
                "loaded": 0, "used": 0, "succeeded": 0, "failed": 0,
                "quality_scores": [],
            }
        stats = self.scenario_stats[task_type]
        stats["loaded"] += 1
        if actually_used:
            stats["used"] += 1
            stats["quality_scores"].append(quality)
            if outcome in (OutcomeType.SUCCESS, OutcomeType.PARTIAL):
                stats["succeeded"] += 1
            else:
                stats["failed"] += 1

        # Recalculate effectiveness
        self._recalculate_effectiveness()
        self.last_used_at = datetime.utcnow().isoformat()

    def _recalculate_effectiveness(self) -> None:
        """Calculate effectiveness score.

        Formula:
          - Usage rate: times_actually_used / times_loaded (agent finds it relevant)
          - Success rate: times_succeeded / times_actually_used (when used, does it work?)
          - Quality factor: avg_quality (when used, how good is the result?)
          - effectiveness = usage_rate * 0.3 + success_rate * 0.3 + quality * 0.4
        """
        if self.times_loaded == 0:
            self.effectiveness_score = 0.0
            return

        usage_rate = self.times_actually_used / self.times_loaded
        success_rate = (
            self.times_succeeded / self.times_actually_used
            if self.times_actually_used > 0 else 0.0
        )
        quality = self.avg_quality

        self.effectiveness_score = (
            usage_rate * 0.3 + success_rate * 0.3 + quality * 0.4
        )

    def best_scenario(self) -> str | None:
        """Return the task type where this skill is most effective."""
        if not self.scenario_stats:
            return None

        best_type = None
        best_score = 0.0
        for task_type, stats in self.scenario_stats.items():
            if stats["used"] == 0:
                continue
            sr = stats["succeeded"] / stats["used"] if stats["used"] > 0 else 0
            qs = sum(stats["quality_scores"]) / len(stats["quality_scores"]) if stats["quality_scores"] else 0
            score = sr * 0.5 + qs * 0.5
            if score > best_score:
                best_score = score
                best_type = task_type

        return best_type


# ============================================================
# Prompt Evolution — prompts that improve over time
# ============================================================

@dataclass
class PromptVersion:
    """A version of a prompt template.

    Prompts evolve based on experience. Each version tracks:
    - The template content
    - Why it was changed (what lesson triggered the change)
    - Performance metrics (avg quality, success rate)
    - When it was created and retired
    """

    version_id: str = ""
    role: str = ""                 # "code" | "ops" | ...
    template: str = ""
    change_reason: str = ""        # Why this version was created
    parent_version: str = ""       # Previous version it evolved from

    # Performance tracking
    times_used: int = 0
    success_count: int = 0
    fail_count: int = 0
    avg_quality: float = 0.0
    quality_scores: list[float] = field(default_factory=list)

    created_at: str = ""
    retired_at: str | None = None  # None = active

    def record_outcome(self, quality: float, success: bool) -> None:
        self.times_used += 1
        self.quality_scores.append(quality)
        self.avg_quality = sum(self.quality_scores) / len(self.quality_scores)
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1

    @property
    def success_rate(self) -> float:
        return self.success_count / self.times_used if self.times_used > 0 else 0.0

    @property
    def is_active(self) -> bool:
        return self.retired_at is None


class PromptEvolution:
    """Manage prompt versions and evolution.

    Prompts start with a base template and evolve based on experience.
    When a prompt consistently underperforms, a new version is created
    with improvements informed by reflection.

    Evolution triggers:
    1. Success rate < 50% over 10+ uses → create improved version
    2. Avg quality < 0.6 over 10+ uses → create improved version
    3. Agent reflection suggests specific improvement
    4. Manual override (user requests change)
    """

    def __init__(self):
        # role → list of versions (most recent last)
        self._versions: dict[str, list[PromptVersion]] = {}
        self._active: dict[str, PromptVersion] = {}  # role → current active version

    def register_version(self, role: str, template: str,
                         change_reason: str = "initial",
                         parent_version: str = "") -> PromptVersion:
        """Register a new prompt version."""
        # Retire current active version
        if role in self._active:
            self._active[role].retired_at = datetime.utcnow().isoformat()

        version = PromptVersion(
            version_id=f"{role}_v{len(self._versions.get(role, [])) + 1}",
            role=role,
            template=template,
            change_reason=change_reason,
            parent_version=parent_version or (self._active[role].version_id if role in self._active else ""),
            created_at=datetime.utcnow().isoformat(),
        )

        if role not in self._versions:
            self._versions[role] = []
        self._versions[role].append(version)
        self._active[role] = version

        logger.info("Prompt evolution: %s → %s (%s)",
                    role, version.version_id, change_reason)
        return version

    def get_active(self, role: str) -> PromptVersion | None:
        return self._active.get(role)

    def record_outcome(self, role: str, quality: float, success: bool) -> None:
        """Record a task outcome for the active prompt version."""
        version = self._active.get(role)
        if version:
            version.record_outcome(quality, success)

    def should_evolve(self, role: str, min_uses: int = 10,
                      min_success_rate: float = 0.5,
                      min_quality: float = 0.6) -> bool:
        """Check if a prompt should be evolved."""
        version = self._active.get(role)
        if not version or version.times_used < min_uses:
            return False
        if version.success_rate < min_success_rate:
            return True
        if version.avg_quality < min_quality:
            return True
        return False

    def get_history(self, role: str) -> list[PromptVersion]:
        """Get all versions for a role (for audit)."""
        return self._versions.get(role, [])


# ============================================================
# Learning Memory — integration with Memory Pool
# ============================================================

class LearningMemory:
    """Store and retrieve learning experiences.

    Experiences are stored in the Memory Pool's episode layer
    (and as JSONL files for auditability).

    Key operations:
    - record(experience): Save an experience record
    - recall(task_scope): Retrieve relevant past experiences
    - get_skill_stats(): Get effectiveness stats for all skills
    - get_prompt_history(): Get prompt evolution history
    """

    def __init__(self, storage_dir: str = "data/learning"):
        self.storage_dir = storage_dir
        self._experiences: list[ExperienceRecord] = []
        self._skill_stats: dict[str, SkillEffectiveness] = {}
        self._prompt_evolution = PromptEvolution()

        # File paths for persistent storage
        self._experiences_file = f"{storage_dir}/experiences.jsonl"
        self._skill_stats_file = f"{storage_dir}/skill_stats.json"
        self._prompt_history_file = f"{storage_dir}/prompt_history.json"

    def record(self, experience: ExperienceRecord) -> None:
        """Record an experience and update all stats."""
        if not experience.record_id:
            experience.record_id = f"exp_{len(self._experiences) + 1}"
        if not experience.started_at:
            experience.started_at = datetime.utcnow().isoformat()
            experience.completed_at = datetime.utcnow().isoformat()

        self._experiences.append(experience)

        # Update skill effectiveness stats
        for skill_name in experience.skills_used:
            if skill_name not in self._skill_stats:
                self._skill_stats[skill_name] = SkillEffectiveness(skill_name=skill_name)

            actually_used = skill_name in experience.skills_actually_used
            task_type = experience.agent_role or "unknown"
            self._skill_stats[skill_name].record_usage(
                task_type=task_type,
                outcome=experience.outcome,
                quality=experience.quality_score,
                actually_used=actually_used,
            )

        # Update prompt evolution stats
        if experience.agent_role and experience.prompt_version:
            self._prompt_evolution.record_outcome(
                role=experience.agent_role,
                quality=experience.quality_score,
                success=experience.outcome in (OutcomeType.SUCCESS, OutcomeType.PARTIAL),
            )

        logger.debug("Recorded experience %s (outcome=%s, quality=%.2f)",
                     experience.record_id, experience.outcome.value,
                     experience.quality_score)

    def recall(self, task_scope: str, task_type: str | None = None,
               limit: int = 5) -> list[ExperienceRecord]:
        """Recall relevant past experiences.

        Simple keyword matching for now. Can be upgraded to semantic search.
        """
        relevant = []
        scope_lower = task_scope.lower()
        task_words = set(scope_lower.split())

        for exp in self._experiences:
            # Match by task type
            if task_type and exp.agent_role == task_type:
                relevant.append(exp)
                continue

            # Match by keyword overlap
            exp_words = set(exp.task_scope.lower().split())
            overlap = len(exp_words & task_words)
            if overlap > 0:
                relevant.append(exp)

        # Sort by relevance (overlap count) then by recency
        relevant.sort(key=lambda e: (
            len(set(e.task_scope.lower().split()) & task_words),
            e.completed_at,
        ), reverse=True)

        return relevant[:limit]

    def get_skill_stats(self) -> dict[str, SkillEffectiveness]:
        return dict(self._skill_stats)

    def get_prompt_evolution(self) -> PromptEvolution:
        return self._prompt_evolution

    def get_summary(self) -> dict:
        """Get learning summary for debugging/dashboard."""
        return {
            "total_experiences": len(self._experiences),
            "success_rate": (
                sum(1 for e in self._experiences
                    if e.outcome == OutcomeType.SUCCESS)
                / len(self._experiences) if self._experiences else 0
            ),
            "avg_quality": (
                sum(e.quality_score for e in self._experiences)
                / len(self._experiences) if self._experiences else 0
            ),
            "skills_tracked": len(self._skill_stats),
            "prompt_roles_tracked": len(self._prompt_evolution._active),
        }


# ============================================================
# Reflection Engine — LLM-powered reflection on experience
# ============================================================

class ReflectionEngine:
    """Generate reflections from experience records.

    After a task completes, the reflection engine:
    1. Analyzes what prompt+skill combination was used
    2. Evaluates the outcome
    3. Generates lessons learned
    4. Suggests improvements for next time

    This is the "thinking" part of the learning loop.
    """

    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    async def reflect(self, experience: ExperienceRecord,
                      past_experiences: list[ExperienceRecord] | None = None) -> str:
        """Generate a reflection on a completed task.

        Args:
            experience: The just-completed experience
            past_experiences: Similar past experiences for comparison

        Returns:
            Reflection text (what worked, what didn't, what to try next time)
        """
        if not self.llm:
            # Fallback: simple rule-based reflection
            return self._rule_based_reflect(experience, past_experiences)

        # LLM-powered reflection
        prompt = self._build_reflection_prompt(experience, past_experiences)
        response = await self.llm.complete(prompt)
        return response.text

    def _rule_based_reflect(self, exp: ExperienceRecord,
                            past: list[ExperienceRecord] | None) -> str:
        """Simple rule-based reflection (no LLM needed)."""
        parts = []

        # Outcome assessment
        if exp.outcome == OutcomeType.SUCCESS:
            parts.append(f"任务成功完成，质量评分 {exp.quality_score:.2f}。")
        elif exp.outcome == OutcomeType.FAILURE:
            parts.append(f"任务失败。质量评分 {exp.quality_score:.2f}。")
        else:
            parts.append(f"任务部分完成。质量评分 {exp.quality_score:.2f}。")

        # Skill usage analysis
        loaded = set(exp.skills_used)
        used = set(exp.skills_actually_used)
        unused = loaded - used
        if unused:
            parts.append(f"加载了但未使用的技能: {', '.join(unused)}。")
            parts.append("下次可以考虑不加载这些技能以节省 token。")
        if used:
            parts.append(f"实际使用的技能: {', '.join(used)}。")

        # Retry analysis
        if exp.retry_count > 0:
            parts.append(f"重试了 {exp.retry_count} 次。考虑优化初始 prompt 以减少重试。")

        # Past comparison
        if past:
            similar_success = sum(1 for e in past
                                  if e.outcome == OutcomeType.SUCCESS)
            parts.append(f"类似任务历史: {len(past)} 次, 成功 {similar_success} 次。")

        # Lessons
        lessons = []
        if exp.quality_score > 0.8:
            lessons.append("当前 prompt+skill 组合效果好，保持。")
        elif exp.quality_score < 0.5:
            lessons.append("当前 prompt+skill 组合效果差，需要改进。")

        if lessons:
            parts.append("经验教训: " + " ".join(lessons))

        return "\n".join(parts)

    @staticmethod
    def _build_reflection_prompt(exp: ExperienceRecord,
                                 past: list[ExperienceRecord] | None) -> str:
        """Build prompt for LLM reflection."""
        past_summary = ""
        if past:
            past_summary = "\n".join(
                f"- 任务: {e.task_scope}, 结果: {e.outcome.value}, 质量: {e.quality_score:.2f}"
                for e in past[:3]
            )

        return f"""请反思以下任务执行过程，总结经验教训：

## 刚完成的任务
- 任务: {exp.task_scope}
- Agent 角色: {exp.agent_role}
- 加载的技能: {exp.skills_used}
- 实际使用的技能: {exp.skills_actually_used}
- 结果: {exp.outcome.value}
- 质量评分: {exp.quality_score:.2f}
- 执行时间: {exp.execution_time_s:.1f}s
- 重试次数: {exp.retry_count}

## 类似任务的历史
{past_summary or "无"}

## 请输出
1. 什么做得好？
2. 什么做得不好？
3. 下次类似任务应该怎么改进？
4. 哪些技能对这个场景有用？哪些没用？
5. 建议的 prompt 改进方向

请简洁回答（200字以内）。
"""


# ============================================================
# Learning Loop — the main orchestration
# ============================================================

class LearningLoop:
    """Orchestrates the learning loop: select → execute → evaluate → learn.

    This is the core of the Loop Engine thinking applied to prompt+skill:
    1. SELECT: Choose prompt+skills based on past experience
    2. EXECUTE: Run the task with telemetry
    3. EVALUATE: Assess outcome (quality, efficiency)
    4. RECORD: Save experience to learning memory
    5. REFLECT: Generate insights (what worked, what didn't)
    6. EVOLVE: Update prompt/skill effectiveness, trigger evolution if needed
    7. SEDIMENT: Insights flow back to Memory Pool (episode layer)
    """

    def __init__(self, memory: LearningMemory,
                 reflector: ReflectionEngine | None = None):
        self.memory = memory
        self.reflector = reflector or ReflectionEngine()

    async def before_task(self, task_scope: str, task_type: str,
                          available_skills: list[str]) -> dict:
        """Phase 1: Select prompt+skills based on past experience.

        Returns:
            Dict with: selected_skills, prompt_version, past_experiences
        """
        # Recall past experiences for this task type
        past = self.memory.recall(task_scope, task_type)

        # Get skill effectiveness stats
        skill_stats = self.memory.get_skill_stats()
        skill_scores = {}
        for skill_name in available_skills:
            stats = skill_stats.get(skill_name)
            if stats:
                # Prefer skills with high effectiveness for this task type
                scenario = stats.scenario_stats.get(task_type, {})
                scenario_score = 0.5  # default
                if scenario.get("used", 0) > 0:
                    sr = scenario["succeeded"] / scenario["used"]
                    qs = (sum(scenario["quality_scores"]) / len(scenario["quality_scores"])
                          if scenario["quality_scores"] else 0)
                    scenario_score = sr * 0.5 + qs * 0.5

                skill_scores[skill_name] = scenario_score
            else:
                # New skill — give it a chance
                skill_scores[skill_name] = 0.5  # neutral

        # Select top skills (balance exploitation vs exploration)
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, score in sorted_skills[:5]]  # Top 5

        # Get active prompt version
        prompt_version = self.memory.get_prompt_evolution().get_active(task_type)

        return {
            "selected_skills": selected,
            "prompt_version": prompt_version.version_id if prompt_version else "default",
            "past_experiences": past,
            "skill_scores": skill_scores,
        }

    async def after_task(self, experience: ExperienceRecord) -> str:
        """Phase 2-7: Evaluate → Record → Reflect → Evolve → Sediment.

        Returns:
            Reflection text
        """
        # Phase 3: Record
        self.memory.record(experience)

        # Phase 4: Recall past for comparison
        past = self.memory.recall(
            experience.task_scope,
            experience.agent_role,
            limit=5,
        )
        # Exclude self
        past = [e for e in past if e.record_id != experience.record_id]

        # Phase 5: Reflect
        reflection = await self.reflector.reflect(experience, past)
        experience.reflection = reflection

        # Phase 6: Check if prompt should evolve
        if experience.agent_role:
            should_evolve = self.memory.get_prompt_evolution().should_evolve(
                experience.agent_role
            )
            if should_evolve:
                logger.info("Prompt for %s needs evolution (low performance)",
                           experience.agent_role)
                # In production, this would trigger an LLM call to generate improved prompt
                # For now, just log it

        return reflection
