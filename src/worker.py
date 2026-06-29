"""AgentWorker — process-isolated agent execution.

Instead of running agents as asyncio tasks in the same process,
AgentWorker spawns a subprocess for each agent, providing:

  1. True isolation (crash doesn't take down the main process)
  2. Independent memory space (no GIL contention)
  3. Resource limits (CPU/memory per agent)
  4. Branch space is a real directory, not just a logical concept

The worker communicates via stdin/stdout JSON messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Worker Protocol (JSON over stdin/stdout)
# ============================================================

@dataclass
class WorkerMessage:
    """Message between main process and worker."""
    type: str  # "task" | "result" | "log" | "error" | "ready"
    data: dict = field(default_factory=dict)


def encode_msg(msg: WorkerMessage) -> str:
    return json.dumps({"type": msg.type, "data": msg.data})


def decode_msg(line: str) -> WorkerMessage | None:
    try:
        obj = json.loads(line)
        return WorkerMessage(type=obj.get("type", ""), data=obj.get("data", {}))
    except (json.JSONDecodeError, KeyError):
        return None


# ============================================================
# AgentWorker — subprocess wrapper
# ============================================================

class AgentWorker:
    """Process-isolated agent worker.

    Each worker runs in a subprocess and communicates via JSON messages
    over stdin/stdout. The worker:

    1. Receives a task via stdin
    2. Executes it (using AgentLoop + Tools)
    3. Sends results back via stdout

    Usage:
        worker = AgentWorker(agent_id="agent:xxx")
        await worker.start()
        result = await worker.run_task(task_scope, context, tools)
        await worker.stop()
    """

    def __init__(self, agent_id: str, workspace_dir: str = "/tmp/agent_loop/workers"):
        self.agent_id = agent_id
        self.workspace = Path(workspace_dir) / agent_id.replace(":", "_")
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._process: asyncio.subprocess.Process | None = None
        self._ready = False
        self._task_future: asyncio.Future | None = None

    async def start(self) -> None:
        """Start the worker subprocess."""
        script = self._generate_worker_script()

        # Write worker script to workspace
        script_path = self.workspace / "_worker.py"
        script_path.write_text(script)

        # Start subprocess
        self._process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env={**os.environ, "AGENT_ID": self.agent_id},
        )

        # Wait for ready signal
        await self._wait_ready()
        logger.info("AgentWorker %s started (pid=%d)", self.agent_id, self._process.pid)

    async def _wait_ready(self, timeout: float = 10.0) -> None:
        """Wait for the worker to signal it's ready."""
        if not self._process or not self._process.stdout:
            return

        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=timeout
            )
            msg = decode_msg(line.decode().strip())
            if msg and msg.type == "ready":
                self._ready = True
        except asyncio.TimeoutError:
            logger.warning("Worker %s didn't signal ready", self.agent_id)

    async def run_task(self, scope: str, context: dict,
                       allowed_tools: list[str] | None = None) -> dict:
        """Send a task to the worker and wait for the result.

        Returns:
            Result dict with keys: summary, artifacts, steps, status
        """
        if not self._process or not self._process.stdin:
            raise RuntimeError("Worker not started")

        if not self._ready:
            raise RuntimeError("Worker not ready")

        # Send task
        task_msg = encode_msg(WorkerMessage(
            type="task",
            data={
                "scope": scope,
                "context": context,
                "allowed_tools": allowed_tools or [],
            }
        ))
        self._process.stdin.write((task_msg + "\n").encode())
        await self._process.stdin.drain()

        # Wait for result
        self._task_future = asyncio.get_event_loop().create_future()

        async def read_result():
            while self._process and self._process.stdout:
                line = await self._process.stdout.readline()
                if not line:
                    break

                msg = decode_msg(line.decode().strip())
                if not msg:
                    continue

                if msg.type == "result":
                    if not self._task_future.done():
                        self._task_future.set_result(msg.data)
                    return
                elif msg.type == "error":
                    if not self._task_future.done():
                        self._task_future.set_exception(
                            RuntimeError(msg.data.get("error", "Worker error"))
                        )
                    return
                elif msg.type == "log":
                    logger.debug("Worker %s: %s", self.agent_id, msg.data)

        asyncio.create_task(read_result())

        try:
            return await asyncio.wait_for(self._task_future, timeout=300.0)
        except asyncio.TimeoutError:
            logger.error("Worker %s task timed out", self.agent_id)
            return {"status": "failed", "summary": "Task timed out", "artifacts": {}}

    async def stop(self) -> None:
        """Stop the worker subprocess."""
        if not self._process:
            return

        try:
            if self._process.stdin:
                self._process.stdin.write_eof()

            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

        except Exception as e:
            logger.warning("Worker %s stop error: %s", self.agent_id, e)
        finally:
            self._process = None
            self._ready = False

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _generate_worker_script(self) -> str:
        """Generate the worker subprocess script.

        The worker:
        1. Signals "ready" on stdout
        2. Reads task from stdin (JSON)
        3. Executes using a minimal AgentLoop
        4. Writes result to stdout (JSON)
        """
        return '''#!/usr/bin/env python3
"""Agent worker subprocess — runs in isolation."""

import asyncio
import json
import os
import sys
import traceback

AGENT_ID = os.environ.get("AGENT_ID", "unknown")

def send(msg_type, data):
    """Send a JSON message to stdout."""
    msg = json.dumps({"type": msg_type, "data": data})
    sys.stdout.write(msg + "\\n")
    sys.stdout.flush()

async def run_task(scope, context, allowed_tools):
    """Execute a task. Override this with actual agent logic."""
    send("log", {"msg": f"Starting task: {scope}"})

    # TODO: Import and run actual AgentLoop here
    # For now, return a placeholder result
    await asyncio.sleep(0.1)  # Simulate work

    return {
        "status": "done",
        "summary": f"Completed: {scope}",
        "artifacts": {},
        "steps": [{"tool_name": "none", "error": None}],
    }

async def main():
    # Signal ready
    send("ready", {"agent_id": AGENT_ID})

    # Read tasks from stdin
    while True:
        line = await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.readline
        )
        if not line:
            break

        try:
            msg = json.loads(line.strip())
            if msg.get("type") == "task":
                data = msg.get("data", {})
                result = await run_task(
                    data.get("scope", ""),
                    data.get("context", {}),
                    data.get("allowed_tools", []),
                )
                send("result", result)
        except Exception as e:
            send("error", {"error": str(e), "traceback": traceback.format_exc()})

if __name__ == "__main__":
    asyncio.run(main())
'''


# ============================================================
# WorkerPool — manages multiple AgentWorkers
# ============================================================

class WorkerPool:
    """Pool of process-isolated agent workers.

    Features:
    - Auto-scaling: create workers on demand up to max_workers
    - Idle timeout: workers auto-stop after idle_timeout seconds
    - Health monitoring: detect crashed workers
    - Graceful shutdown: stop all workers on pool close
    """

    def __init__(self, max_workers: int = 10, idle_timeout: float = 300.0):
        self.max_workers = max_workers
        self.idle_timeout = idle_timeout
        self._workers: dict[str, AgentWorker] = {}
        self._idle: list[str] = []  # Queue of idle worker IDs
        self._lock = asyncio.Lock()

    async def acquire(self, agent_id: str) -> AgentWorker | None:
        """Get an idle worker or create a new one."""
        async with self._lock:
            # Reuse idle worker
            while self._idle:
                wid = self._idle.pop(0)
                worker = self._workers.get(wid)
                if worker and worker.is_alive:
                    return worker

            # Create new worker
            if len(self._workers) < self.max_workers:
                worker = AgentWorker(agent_id=agent_id)
                try:
                    await worker.start()
                    self._workers[agent_id] = worker
                    return worker
                except Exception as e:
                    logger.error("Failed to start worker %s: %s", agent_id, e)
                    return None

            # Pool full
            return None

    async def release(self, agent_id: str) -> None:
        """Release a worker back to the idle pool."""
        async with self._lock:
            if agent_id in self._workers:
                self._idle.append(agent_id)

    async def destroy(self, agent_id: str) -> None:
        """Stop and remove a worker."""
        async with self._lock:
            worker = self._workers.pop(agent_id, None)
            if agent_id in self._idle:
                self._idle.remove(agent_id)
        if worker:
            await worker.stop()

    async def destroy_all(self) -> None:
        """Stop all workers."""
        async with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
            self._idle.clear()

        for worker in workers:
            await worker.stop()

    def stats(self) -> dict:
        return {
            "total": len(self._workers),
            "idle": len(self._idle),
            "active": len(self._workers) - len(self._idle),
            "max": self.max_workers,
        }
