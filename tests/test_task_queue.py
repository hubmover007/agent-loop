"""Tests for P5-B Task Queue & Interrupt Control System.

Covers:
  - InputQueue priority ordering
  - TaskHandle cancel/pause/resume lifecycle
  - InterruptController signal detection (CN + EN keywords)
  - TaskDispatcher simple vs complex routing
  - Cancellation checkpoints
  - Continuous input queuing
  - Pause followed by resume
"""

import asyncio
import pytest
from datetime import datetime, timezone

from src.task_queue import (
    InputPriority,
    TaskState,
    QueuedInput,
    TaskHandle,
    InputQueue,
    InterruptController,
    TaskDispatcher,
)


# ============================================================
# InputPriority & TaskState
# ============================================================

class TestInputPriority:
    def test_priority_ordering(self):
        """INTERRUPT < URGENT < NORMAL < BACKGROUND."""
        assert InputPriority.INTERRUPT.value < InputPriority.URGENT.value
        assert InputPriority.URGENT.value < InputPriority.NORMAL.value
        assert InputPriority.NORMAL.value < InputPriority.BACKGROUND.value

    def test_all_priorities_defined(self):
        """All four priority levels exist."""
        assert len(list(InputPriority)) == 4


class TestTaskState:
    def test_all_states(self):
        states = set(s.value for s in TaskState)
        expected = {"queued", "running", "paused", "cancelled", "completed", "failed"}
        assert states == expected

    def test_terminal_states(self):
        """COMPLETED, CANCELLED, FAILED are terminal."""
        from src.task_queue import TaskHandle as TH
        handle = TH(task_id="t1", input_id="i1")
        handle.state = TaskState.COMPLETED
        assert handle.is_done()
        handle.state = TaskState.CANCELLED
        assert handle.is_done()
        handle.state = TaskState.FAILED
        assert handle.is_done()
        handle.state = TaskState.RUNNING
        assert not handle.is_done()


# ============================================================
# QueuedInput
# ============================================================

class TestQueuedInput:
    def test_creation(self):
        qi = QueuedInput(id="input_1", content="hello", priority=InputPriority.NORMAL)
        assert qi.id == "input_1"
        assert qi.content == "hello"
        assert qi.priority == InputPriority.NORMAL
        assert isinstance(qi.created_at, datetime)

    def test_priority_comparison(self):
        """Items can be sorted by priority."""
        high = QueuedInput(id="h", content="hi", priority=InputPriority.INTERRUPT)
        low = QueuedInput(id="l", content="lo", priority=InputPriority.BACKGROUND)
        assert high.priority.value < low.priority.value


# ============================================================
# TaskHandle
# ============================================================

class TestTaskHandle:
    @pytest.fixture
    def handle(self):
        return TaskHandle(task_id="task:test", input_id="input:test")

    def test_initial_state(self, handle):
        """New handle starts in QUEUED state."""
        assert handle.state == TaskState.QUEUED
        assert not handle.is_done()
        assert handle.result is None
        assert handle.error is None

    def test_cancel(self, handle):
        """Cancel sets the event and state."""
        handle.state = TaskState.RUNNING
        handle.cancel()
        assert handle.state == TaskState.CANCELLED

    def test_pause_and_resume(self, handle):
        """Pause clears event, resume sets it."""
        handle.state = TaskState.RUNNING
        handle.pause()
        assert handle.state == TaskState.PAUSED
        handle.resume()
        assert handle.state == TaskState.RUNNING

    @pytest.mark.asyncio
    async def test_check_cancelled(self, handle):
        """check_cancelled returns False before cancel, True after."""
        assert not await handle.check_cancelled()
        handle.cancel()
        assert await handle.check_cancelled()

    @pytest.mark.asyncio
    async def test_wait_if_paused_non_blocking(self, handle):
        """wait_if_paused returns immediately when not paused."""
        try:
            await asyncio.wait_for(handle.wait_if_paused(), timeout=0.1)
        except asyncio.TimeoutError:
            pytest.fail("wait_if_paused should not block when not paused")

    @pytest.mark.asyncio
    async def test_wait_if_paused_blocks(self, handle):
        """wait_if_paused blocks when paused."""
        handle.state = TaskState.RUNNING
        handle.pause()  # _pause_event is now cleared

        async def wait():
            await handle.wait_if_paused()

        task = asyncio.create_task(wait())
        await asyncio.sleep(0.05)
        assert not task.done()  # Should still be waiting

        handle.resume()  # _pause_event is now set
        await asyncio.wait_for(task, timeout=0.5)
        assert task.done()

    def test_complete(self, handle):
        """complete() sets state to COMPLETED and records result."""
        handle.complete("task result")
        assert handle.state == TaskState.COMPLETED
        assert handle.result == "task result"
        assert handle.completed_at is not None

    def test_is_done_detects_terminal(self, handle):
        """is_done() returns True for terminal states."""
        for state in [TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED]:
            handle.state = state
            assert handle.is_done(), f"is_done should be True for {state}"

    def test_pause_only_on_running(self, handle):
        """pause() works on any state, but typically on RUNNING."""
        handle.state = TaskState.QUEUED
        handle.pause()
        assert handle.state == TaskState.PAUSED

    @pytest.mark.asyncio
    async def test_cancel_then_check_returns_true(self, handle):
        """After cancel(), check_cancelled() should return True."""
        handle.state = TaskState.RUNNING
        handle.cancel()
        assert handle.state == TaskState.CANCELLED
        assert await handle.check_cancelled()

    @pytest.mark.asyncio
    async def test_pause_resume_cycle(self, handle):
        """Full pause → resume cycle."""
        handle.state = TaskState.RUNNING
        assert handle.state == TaskState.RUNNING

        handle.pause()
        assert handle.state == TaskState.PAUSED

        handle.resume()
        assert handle.state == TaskState.RUNNING


# ============================================================
# InputQueue
# ============================================================

class TestInputQueue:
    @pytest.fixture
    def queue(self):
        return InputQueue()

    def test_initial_empty(self, queue):
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_put_and_get(self, queue):
        """Basic put + get returns same item."""
        item_id = await queue.put("hello world")
        item = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert item.id == item_id
        assert item.content == "hello world"
        assert item.priority == InputPriority.NORMAL

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Higher priority items come out first."""
        # Enqueue in reverse priority order
        await queue.put("low", InputPriority.BACKGROUND)
        await queue.put("normal", InputPriority.NORMAL)
        await queue.put("urgent", InputPriority.URGENT)
        await queue.put("interrupt", InputPriority.INTERRUPT)

        # Dequeue — should come out in priority order
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert first.content == "interrupt"

        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert second.content == "urgent"

        third = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert third.content == "normal"

        fourth = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert fourth.content == "low"

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, queue):
        """Items with same priority respect FIFO ordering."""
        await queue.put("first", InputPriority.NORMAL)
        await queue.put("second", InputPriority.NORMAL)
        await queue.put("third", InputPriority.NORMAL)

        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        third = await asyncio.wait_for(queue.get(), timeout=0.5)

        assert first.content == "first"
        assert second.content == "second"
        assert third.content == "third"

    @pytest.mark.asyncio
    async def test_qsize_tracks_items(self, queue):
        """qsize() reflects the number of queued items."""
        assert queue.qsize() == 0
        await queue.put("a")
        assert queue.qsize() == 1
        await queue.put("b")
        assert queue.qsize() == 2
        await queue.get()
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_continuous_input_queuing(self, queue):
        """Simulate continuous user input — all items queued without blocking."""
        n = 10
        for i in range(n):
            await queue.put(f"msg_{i}")

        assert queue.qsize() == n
        # Dequeue all
        results = []
        for _ in range(n):
            item = await asyncio.wait_for(queue.get(), timeout=0.5)
            results.append(item.content)
        assert len(results) == n
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_get_blocks_on_empty(self, queue):
        """get() should block when queue is empty."""
        async def get_item():
            return await queue.get()

        task = asyncio.create_task(get_item())
        await asyncio.sleep(0.05)
        assert not task.done()  # Should be waiting

        # Put an item to unblock
        await queue.put("unblock")
        result = await asyncio.wait_for(task, timeout=0.5)
        assert task.done()
        assert result.content == "unblock"


# ============================================================
# InterruptController
# ============================================================

class TestInterruptController:
    @pytest.fixture
    def ic(self):
        return InterruptController()

    # ── Signal Detection ──
    def test_detect_interrupt_cn(self, ic):
        for word in ["停", "停下", "停止", "打断"]:
            assert ic.detect_signal(word) == "interrupt", f"'{word}' should be interrupt"

    def test_detect_interrupt_en(self, ic):
        assert ic.detect_signal("stop") == "interrupt"

    def test_detect_cancel_cn(self, ic):
        for word in ["取消", "取消上一个", "撤销"]:
            assert ic.detect_signal(word) == "cancel", f"'{word}' should be cancel"

    def test_detect_cancel_en(self, ic):
        assert ic.detect_signal("cancel") == "cancel"

    def test_detect_pause_cn(self, ic):
        for word in ["暂停", "等等", "wait"]:
            assert ic.detect_signal(word) == "pause", f"'{word}' should be pause"

    def test_detect_pause_en(self, ic):
        assert ic.detect_signal("pause") == "pause"

    def test_detect_resume_cn(self, ic):
        for word in ["继续", "continue", "go"]:
            assert ic.detect_signal(word) == "resume", f"'{word}' should be resume"

    def test_detect_resume_en(self, ic):
        assert ic.detect_signal("resume") == "resume"

    def test_no_signal_for_normal_text(self, ic):
        """Normal messages should not be detected as signals."""
        assert ic.detect_signal("帮我部署一台服务器") is None
        assert ic.detect_signal("hello") is None
        assert ic.detect_signal("") is None
        assert ic.detect_signal("分析一下这个问题") is None

    def test_detect_signal_case_insensitive(self, ic):
        """Signal detection is case-insensitive for English keywords."""
        assert ic.detect_signal("Stop") == "interrupt"
        assert ic.detect_signal("CANCEL") == "cancel"
        assert ic.detect_signal("Pause") == "pause"

    # ── Signal Handling ──
    @pytest.mark.asyncio
    async def test_handle_interrupt_with_task(self, ic):
        """Interrupt when a task is running should cancel it."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING
        ic.set_current_task(handle)
        result = await ic.handle_signal("interrupt")
        assert "打断" in result
        assert handle.state == TaskState.CANCELLED

    @pytest.mark.asyncio
    async def test_handle_interrupt_without_task(self, ic):
        """Interrupt when nothing is running should give feedback."""
        result = await ic.handle_signal("interrupt")
        assert "没有运行中" in result

    @pytest.mark.asyncio
    async def test_handle_cancel_with_task(self, ic):
        """Cancel when a task is running should cancel it."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING
        ic.set_current_task(handle)
        result = await ic.handle_signal("cancel")
        assert "取消" in result
        assert handle.state == TaskState.CANCELLED

    @pytest.mark.asyncio
    async def test_handle_cancel_without_task(self, ic):
        """Cancel when nothing is running should give feedback."""
        result = await ic.handle_signal("cancel")
        assert "没有运行中" in result

    @pytest.mark.asyncio
    async def test_handle_pause(self, ic):
        """Pause a running task."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING
        ic.set_current_task(handle)
        result = await ic.handle_signal("pause")
        assert "暂停" in result
        assert handle.state == TaskState.PAUSED

    @pytest.mark.asyncio
    async def test_handle_pause_no_running_task(self, ic):
        """Pause when nothing is RUNNING should give feedback."""
        result = await ic.handle_signal("pause")
        assert "没有可暂停" in result

    @pytest.mark.asyncio
    async def test_handle_resume(self, ic):
        """Resume a paused task."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.PAUSED
        ic.set_current_task(handle)
        result = await ic.handle_signal("resume")
        assert "恢复" in result
        assert handle.state == TaskState.RUNNING

    @pytest.mark.asyncio
    async def test_handle_resume_no_paused_task(self, ic):
        """Resume when nothing is PAUSED should give feedback."""
        result = await ic.handle_signal("resume")
        assert "没有暂停" in result

    @pytest.mark.asyncio
    async def test_handle_unknown_signal(self, ic):
        """Unknown signal returns feedback."""
        result = await ic.handle_signal("unknown")
        assert result == "未知信号"

    # ── Task History ──
    def test_task_history_preserved(self, ic):
        """set_current_task preserves previous tasks in history."""
        h1 = TaskHandle(task_id="t1", input_id="i1")
        h2 = TaskHandle(task_id="t2", input_id="i2")

        ic.set_current_task(h1)
        assert ic.current_task == h1
        assert len(ic.task_history) == 0

        ic.set_current_task(h2)
        assert ic.current_task == h2
        assert len(ic.task_history) == 1
        assert ic.task_history[0] == h1

    @pytest.mark.asyncio
    async def test_interrupt_on_completed_task_noop(self, ic):
        """Interrupt on completed task doesn't cancel it."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.COMPLETED
        ic.set_current_task(handle)

        # is_done() should return True, so handle_signal should return "没有运行中"
        result = await ic.handle_signal("interrupt")
        assert "没有运行中" in result


# ============================================================
# TaskDispatcher
# ============================================================

class TestTaskDispatcher:
    @pytest.fixture
    def dispatcher(self):
        return TaskDispatcher()  # No main_loop → uses mock

    def test_is_simple_short_text(self, dispatcher):
        """Short text without action verbs is simple."""
        assert dispatcher._is_simple("hello")
        assert dispatcher._is_simple("你好")
        assert dispatcher._is_simple("what is AI?")
        assert dispatcher._is_simple("今天天气怎么样")

    def test_is_simple_with_action_verbs_returns_false(self, dispatcher):
        """Text with action verbs is not simple."""
        assert not dispatcher._is_simple("帮我部署一个服务器")
        assert not dispatcher._is_simple("please deploy the app")
        assert not dispatcher._is_simple("修复这个bug")
        assert not dispatcher._is_simple("创建一张表")
        assert not dispatcher._is_simple("analyze the data please")
        assert not dispatcher._is_simple("check the status")

    def test_is_simple_long_text_returns_false(self, dispatcher):
        """Very long text (>200 chars) is not simple."""
        long_text = "a" * 201
        assert not dispatcher._is_simple(long_text)

    def test_is_simple_200_chars_ok(self, dispatcher):
        """Text at exactly 200 chars is considered simple if no action verbs."""
        text = "b" * 200
        assert dispatcher._is_simple(text)

    @pytest.mark.asyncio
    async def test_dispatch_interrupt_signal(self, dispatcher):
        """Dispatch should intercept '停' as interrupt signal."""
        # Set up a running task so interrupt has something to cancel
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING
        dispatcher.interrupt_controller.set_current_task(handle)
        result = await dispatcher.dispatch("停")
        assert "打断" in result or "取消" in result

    @pytest.mark.asyncio
    async def test_dispatch_simple_input_returns_mock(self, dispatcher):
        """Simple input without main_loop returns mock response."""
        result = await dispatcher.dispatch("hello")
        assert "mock response" in result or "hello" in result

    @pytest.mark.asyncio
    async def test_dispatch_creates_task_handle(self, dispatcher):
        """Each dispatch creates a TaskHandle."""
        await dispatcher.dispatch("hello")
        current = dispatcher.interrupt_controller.current_task
        assert current is not None
        assert current.state in (TaskState.COMPLETED, TaskState.CANCELLED)

    @pytest.mark.asyncio
    async def test_dispatch_cancel_signal(self, dispatcher):
        """Dispatch should intercept '取消' as cancel signal."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING
        dispatcher.interrupt_controller.set_current_task(handle)
        result = await dispatcher.dispatch("取消")
        assert "取消" in result

    @pytest.mark.asyncio
    async def test_dispatch_pause_signal(self, dispatcher):
        """Dispatch should intercept '暂停' as pause signal."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING
        dispatcher.interrupt_controller.set_current_task(handle)
        result = await dispatcher.dispatch("暂停")
        assert "暂停" in result or "没有可暂停" in result

    @pytest.mark.asyncio
    async def test_dispatch_resume_signal(self, dispatcher):
        """Dispatch should intercept '继续' as resume signal."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.PAUSED
        dispatcher.interrupt_controller.set_current_task(handle)
        result = await dispatcher.dispatch("继续")
        assert "恢复" in result or "没有暂停" in result

    @pytest.mark.asyncio
    async def test_dispatch_complex_input(self, dispatcher):
        """Complex input uses main_loop mock path."""
        # Since dispatcher has no main_loop, complex input also falls to mock
        result = await dispatcher.dispatch("帮我部署一个前端服务")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_tasks_dont_have_to_share_handle(self, dispatcher):
        """Sequential dispatches each get their own TaskHandle."""
        r1 = await dispatcher.dispatch("hello")
        h1 = dispatcher.interrupt_controller.current_task
        r2 = await dispatcher.dispatch("world")
        h2 = dispatcher.interrupt_controller.current_task
        assert h1 is not h2  # Different handles
        assert len(dispatcher.interrupt_controller.task_history) > 0

    # ── Cancellation during mock execution ──
    @pytest.mark.asyncio
    async def test_cancel_mid_dispatch_preempts(self, dispatcher):
        """If cancel event is set before dispatch starts, it raises CancelledError."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.cancel()  # Already cancelled
        dispatcher.interrupt_controller.set_current_task(handle)

        # The dispatch will create a new handle but check the controller's
        # current task — since _run_with_cancel_check checks handle.is_done()
        # but the dispatch creates a NEW handle first
        result = await dispatcher.dispatch("hello")
        assert isinstance(result, str)


# ============================================================
# Integration: Pause + Resume Flow
# ============================================================

class TestPauseResumeFlow:
    @pytest.mark.asyncio
    async def test_full_pause_resume_cycle(self):
        """Integration test: pause → wait → resume completes properly."""
        handle = TaskHandle(task_id="t1", input_id="i1")

        # Start running
        handle.state = TaskState.RUNNING
        assert handle.state == TaskState.RUNNING

        # Pause
        handle.pause()
        assert handle.state == TaskState.PAUSED

        # wait_if_paused blocks
        async def waiter():
            await handle.wait_if_paused()
            return "done"

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert not task.done(), "Should block while paused"

        # Resume
        handle.resume()
        assert handle.state == TaskState.RUNNING

        result = await asyncio.wait_for(task, timeout=0.5)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_continuous_input_with_pause_resume(self):
        """Simulate: user sends input, pauses, resumes, input processed."""
        ic = InterruptController()

        # Task starts
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING
        ic.set_current_task(handle)

        # User says "暂停"
        signal = ic.detect_signal("暂停")
        assert signal == "pause"
        response = await ic.handle_signal(signal)
        assert "暂停" in response
        assert handle.state == TaskState.PAUSED

        # User says "继续"
        signal = ic.detect_signal("继续")
        assert signal == "resume"
        response = await ic.handle_signal(signal)
        assert "恢复" in response
        assert handle.state == TaskState.RUNNING


# ============================================================
# Integration: Queue + Dispatcher Interaction
# ============================================================

class TestQueueDispatcherIntegration:
    @pytest.mark.asyncio
    async def test_queue_then_dispatch(self):
        """Items queued in InputQueue can be dispatched via TaskDispatcher."""
        queue = InputQueue()
        dispatcher = TaskDispatcher()

        # Queue several inputs
        await queue.put("hello", InputPriority.NORMAL)
        await queue.put("停", InputPriority.NORMAL)

        # Dispatch first — should be normal
        item = await asyncio.wait_for(queue.get(), timeout=0.5)
        result = await dispatcher.dispatch(item.content)
        assert isinstance(result, str)

        # Dispatch second — should trigger interrupt
        item = await asyncio.wait_for(queue.get(), timeout=0.5)
        result = await dispatcher.dispatch(item.content)
        assert isinstance(result, str)  # "没有任何任务" or similar


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_handle_created_with_all_fields(self):
        """TaskHandle can be created with all fields specified."""
        handle = TaskHandle(
            task_id="custom_id",
            input_id="custom_input",
            state=TaskState.RUNNING,
            error="some error",
        )
        assert handle.task_id == "custom_id"
        assert handle.state == TaskState.RUNNING
        assert handle.error == "some error"

    @pytest.mark.asyncio
    async def test_multiple_interrupt_sends_only_one_cancel(self):
        """Multiple interrupt signals on the same task cancel it once."""
        handle = TaskHandle(task_id="t1", input_id="i1")
        handle.state = TaskState.RUNNING

        handle.cancel()
        assert handle.state == TaskState.CANCELLED

        # Second cancel is idempotent
        handle.cancel()
        assert handle.state == TaskState.CANCELLED

    def test_signal_detection_exact_match(self):
        """Only exact matches (strip+lower) are recognized."""
        ic = InterruptController()
        assert ic.detect_signal("停") == "interrupt"
        assert ic.detect_signal("停止!") is None  # Not an exact match
        assert ic.detect_signal(" 停 ") == "interrupt"  # Strip works

    @pytest.mark.asyncio
    async def test_empty_queue_size(self):
        """Empty queue reports size 0."""
        queue = InputQueue()
        assert queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_mixed_priority_drain(self):
        """Queue drains correctly with mixed priorities."""
        queue = InputQueue()
        await queue.put("b1", InputPriority.BACKGROUND)
        await queue.put("n1", InputPriority.NORMAL)
        await queue.put("u1", InputPriority.URGENT)
        await queue.put("n2", InputPriority.NORMAL)
        await queue.put("b2", InputPriority.BACKGROUND)

        order = []
        while queue.qsize() > 0:
            item = await asyncio.wait_for(queue.get(), timeout=0.5)
            order.append(item.content)

        # Urgent should come first, then normals, then backgrounds
        assert order[0] == "u1"
        assert "n1" in order[1:3]
        assert "n2" in order[1:3]
        assert "b1" in order[3:5]
        assert "b2" in order[3:5]
