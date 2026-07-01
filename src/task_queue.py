"""Task Queue & Interrupt Control System.

Handles:
  - InputQueue: priority-ordered queue for user inputs
  - TaskDispatcher: routes to default agent (simple) or TaskAgent (complex)
  - InterruptController: handles stop/cancel/pause/resume signals
  - TaskHandle: trackable handle for running tasks
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InputPriority(Enum):
    """Priority levels for input queue."""
    INTERRUPT = 0   # Highest: interrupt signals processed immediately
    URGENT = 1      # Urgent
    NORMAL = 2      # Normal
    BACKGROUND = 3  # Background tasks


class TaskState(Enum):
    """State of a dispatched task."""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedInput:
    """A user input in the queue."""
    id: str
    content: str
    priority: InputPriority
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_handle: TaskHandle | None = None


@dataclass
class TaskHandle:
    """Trackable handle for a running task.

    Provides cancellation, pause/resume support via asyncio Events.
    Tasks should call check_cancelled() and wait_if_paused() at safe points
    during execution.
    """
    task_id: str
    input_id: str
    state: TaskState = TaskState.QUEUED
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _pause_event: asyncio.Event = field(default_factory=lambda: asyncio.Event())

    def __post_init__(self):
        self._pause_event.set()  # Not paused by default

    def cancel(self):
        """Request cancellation."""
        self._cancel_event.set()
        self.state = TaskState.CANCELLED

    def pause(self):
        """Pause the task."""
        self._pause_event.clear()
        self.state = TaskState.PAUSED

    def resume(self):
        """Resume the paused task."""
        self._pause_event.set()
        self.state = TaskState.RUNNING

    def complete(self, result: Any = None):
        """Mark task as completed."""
        self.state = TaskState.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc)

    async def check_cancelled(self) -> bool:
        """Check if cancellation was requested.

        Returns True if the task has been cancelled.
        """
        return self._cancel_event.is_set()

    async def wait_if_paused(self):
        """Block if paused, return immediately if running."""
        await self._pause_event.wait()

    def is_done(self) -> bool:
        """Check if task is in a terminal state."""
        return self.state in (TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED)


class InputQueue:
    """Priority-ordered async input queue.

    Features:
    - Priority ordering (interrupt > urgent > normal > background)
    - Non-blocking enqueue
    - Async dequeue with await
    - Queue size tracking
    """

    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._counter = 0  # Ensures FIFO within same priority

    async def put(self, content: str, priority: InputPriority = InputPriority.NORMAL) -> str:
        """Add input to queue. Returns input_id."""
        input_id = f"input_{self._counter}"
        self._counter += 1
        item = QueuedInput(id=input_id, content=content, priority=priority)
        await self._queue.put((priority.value, self._counter, item))
        return input_id

    async def get(self) -> QueuedInput:
        """Get next input from queue (blocks if empty)."""
        _, _, item = await self._queue.get()
        return item

    def qsize(self) -> int:
        return self._queue.qsize()


class InterruptController:
    """Handles user interrupt signals (stop/cancel/pause/resume).

    Detects interrupt keywords in user input and routes them
    to the appropriate TaskHandle.
    """

    INTERRUPT_KEYWORDS = {"stop", "停", "停下", "停止", "打断"}
    CANCEL_KEYWORDS = {"cancel", "取消", "取消上一个", "撤销"}
    PAUSE_KEYWORDS = {"pause", "暂停", "等等", "wait"}
    RESUME_KEYWORDS = {"resume", "继续", "continue", "go"}

    def __init__(self):
        self._current_task: TaskHandle | None = None
        self._task_history: list[TaskHandle] = []

    @property
    def current_task(self) -> TaskHandle | None:
        return self._current_task

    @property
    def task_history(self) -> list[TaskHandle]:
        return list(self._task_history)

    def detect_signal(self, text: str) -> str | None:
        """Detect interrupt signal from user text. Returns signal type or None."""
        text_lower = text.lower().strip()
        if text_lower in self.INTERRUPT_KEYWORDS:
            return "interrupt"
        if text_lower in self.CANCEL_KEYWORDS:
            return "cancel"
        if text_lower in self.PAUSE_KEYWORDS:
            return "pause"
        if text_lower in self.RESUME_KEYWORDS:
            return "resume"
        return None

    async def handle_signal(self, signal: str) -> str:
        """Handle an interrupt signal. Returns response message."""
        if signal == "interrupt":
            if self._current_task and not self._current_task.is_done():
                self._current_task.cancel()
                return "已打断当前任务"
            return "当前没有运行中的任务"

        elif signal == "cancel":
            if self._current_task and not self._current_task.is_done():
                self._current_task.cancel()
                return "已取消当前任务"
            return "当前没有运行中的任务"

        elif signal == "pause":
            if self._current_task and self._current_task.state == TaskState.RUNNING:
                self._current_task.pause()
                return "已暂停，说'继续'恢复"
            return "当前没有可暂停的任务"

        elif signal == "resume":
            if self._current_task and self._current_task.state == TaskState.PAUSED:
                self._current_task.resume()
                return "已恢复执行"
            return "当前没有暂停的任务"

        return "未知信号"

    def set_current_task(self, handle: TaskHandle):
        if self._current_task:
            self._task_history.append(self._current_task)
        self._current_task = handle


class TaskDispatcher:
    """Dispatches inputs to default agent or task system.

    Decision logic:
    - Interrupt signals → InterruptController
    - Simple questions (greetings, factual, single-step) → DefaultAgent direct answer
    - Complex tasks (multi-step, requires tools, requires decomposition) → TaskAgent
    """

    def __init__(self, main_loop=None, interrupt_controller=None):
        self.main_loop = main_loop
        self.interrupt_controller = interrupt_controller or InterruptController()
        self.input_queue = InputQueue()

    async def dispatch(self, content: str) -> str:
        """Process one input. Returns response string."""
        # 1. Check for interrupt signals
        signal = self.interrupt_controller.detect_signal(content)
        if signal:
            return await self.interrupt_controller.handle_signal(signal)

        # 2. Create task handle
        handle = TaskHandle(
            task_id=f"task_{asyncio.get_event_loop().time()}",
            input_id=f"input_0",
        )
        self.interrupt_controller.set_current_task(handle)

        # 3. Decide: simple vs complex
        if self._is_simple(content):
            # Simple: direct LLM call
            handle.state = TaskState.RUNNING
            try:
                result = await self._run_with_cancel_check(handle, content)
                handle.complete(result)
                return result
            except asyncio.CancelledError:
                handle.state = TaskState.CANCELLED
                return "任务已取消"
        else:
            # Complex: go through MainLoop (decompose → agents)
            handle.state = TaskState.RUNNING
            try:
                result = await self._run_with_cancel_check(handle, content, complex=True)
                handle.complete(result)
                return result
            except asyncio.CancelledError:
                handle.state = TaskState.CANCELLED
                return "任务已取消"

    def _is_simple(self, text: str) -> bool:
        """Heuristic: is this simple enough for direct answer?

        Simple if:
        - Short (<= 200 chars)
        - No action verbs (部署/修复/创建/分析/检查)
        - Greeting / factual question / single concept
        """
        if len(text) > 200:
            return False
        action_verbs = [
            "部署", "修复", "创建", "分析", "检查",
            "deploy", "fix", "create", "analyze", "check", "build", "implement",
        ]
        return not any(v in text.lower() for v in action_verbs)

    async def _run_with_cancel_check(self, handle: TaskHandle, content: str,
                                     complex: bool = False) -> str:
        """Run task with cancellation checking at safe points."""
        if handle.is_done():
            raise asyncio.CancelledError()

        await handle.wait_if_paused()

        if await handle.check_cancelled():
            raise asyncio.CancelledError()

        if complex and self.main_loop:
            result = await self.main_loop.run(content, task_handle=handle)
        else:
            # Simple LLM call (mock if no main_loop/llm)
            if self.main_loop and getattr(self.main_loop, 'llm', None):
                resp = await self.main_loop.llm.chat([
                    {"role": "user", "content": content}
                ])
                result = resp.content
            else:
                result = f"[mock response for: {content}]"

        if await handle.check_cancelled():
            raise asyncio.CancelledError()

        return result


__all__ = [
    "InputPriority",
    "TaskState",
    "QueuedInput",
    "TaskHandle",
    "InputQueue",
    "InterruptController",
    "TaskDispatcher",
]
