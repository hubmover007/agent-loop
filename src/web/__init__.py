"""Web API for Agent-Loop platform.

REST + WebSocket API for:
  - POST /api/chat           — submit user input, start Loop
  - GET  /api/status         — get system status
  - GET  /api/tasks          — list all tasks
  - GET  /api/tasks/{id}     — get task detail
  - GET  /api/agents         — list all agents
  - POST /api/tasks/{id}/cancel — cancel a task
  - WS   /ws/stream          — real-time loop progress stream

Designed for platform deployment (multi-user, horizontal scaling).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from ..core import TaskStatus, AgentStatus, LoopPhase
from ..loop_engine import LLMProvider, LoopConfig
from ..loop_engine.main_loop import MainLoop, LoopContext
from ..memory import MemoryPool
from ..task_manager import TaskManagerAgent

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None
    BaseModel = object  # fallback


# ============================================================
# Request/Response Models
# ============================================================

if FastAPI:

    class ChatRequest(BaseModel):
        message: str
        session_id: str | None = None
        thinking: bool = False
        max_tasks: int = 10


    class ChatResponse(BaseModel):
        session_id: str
        output: str
        tasks_created: int
        tasks_done: int
        tasks_failed: int
        errors: list[str]


    class StatusResponse(BaseModel):
        loop_phase: str
        pipeline: dict
        memory: dict
        agents: dict


# ============================================================
# App Factory
# ============================================================

def create_app(memory: MemoryPool, llm: LLMProvider,
               config: LoopConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI app.

    Usage:
        app = create_app(memory, llm)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    if FastAPI is None:
        raise ImportError(
            "FastAPI not installed. Install with: pip install fastapi uvicorn"
        )

    config = config or LoopConfig()
    main_loop = MainLoop(memory=memory, llm=llm, config=config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Agent-Loop API starting up")
        yield
        logger.info("Agent-Loop API shutting down")

    app = FastAPI(
        title="Agent-Loop API",
        description="Loop Engine for AI Agent orchestration",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store in app state
    app.state.main_loop = main_loop
    app.state.memory = memory
    app.state.llm = llm
    app.state.config = config
    app.state.active_loops: dict[str, LoopContext] = {}

    # ============================================================
    # Endpoints
    # ============================================================

    @app.get("/api/status")
    async def get_status():
        """Get system status overview."""
        tm = main_loop.task_manager
        pipeline_status = tm.status() if tm else {
            "total": 0, "pending": 0, "running": 0,
            "done": 0, "failed": 0, "inflight": 0,
            "agents_active": 0, "agents_idle": 0,
        }

        return {
            "loop_phase": "idle" if not app.state.active_loops else "running",
            "active_loops": len(app.state.active_loops),
            "pipeline": pipeline_status,
            "model": getattr(llm, "provider_name", "unknown"),
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Submit user input and run the full Loop cycle."""
        import uuid as uuid_mod
        session_id = request.session_id or f"session:{uuid_mod.uuid4().hex[:12]}"

        # Run MainLoop
        ctx = await main_loop.run(request.message)

        app.state.active_loops[session_id] = ctx

        return ChatResponse(
            session_id=session_id,
            output=ctx.final_output or "",
            tasks_created=len(ctx.task_ids),
            tasks_done=len(ctx.agent_results),
            tasks_failed=len(ctx.discarded_results),
            errors=ctx.errors,
        )

    @app.get("/api/tasks")
    async def list_tasks():
        """List all tasks in the registry."""
        tm = main_loop.task_manager
        if not tm:
            return {"tasks": []}
        return {"tasks": [t.to_dict() for t in tm.registry.all_tasks()]}

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get detailed task information."""
        tm = main_loop.task_manager
        if not tm:
            return JSONResponse({"error": "No task manager"}, status_code=503)

        task = tm.registry.get(task_id)
        if not task:
            return JSONResponse({"error": "Task not found"}, status_code=404)

        result = {
            **task.to_dict(),
        }
        if task.result:
            result["result"] = {
                "summary": task.result.summary,
                "artifacts": task.result.artifacts,
            }
        if task.evaluation:
            result["evaluation"] = {
                "scores": task.evaluation.scores,
                "overall": task.evaluation.overall,
                "action": task.evaluation.action,
                "reason": task.evaluation.reason,
            }
        return result

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str):
        """Cancel a running task."""
        tm = main_loop.task_manager
        if not tm:
            return JSONResponse({"error": "No task manager"}, status_code=503)

        await tm.cancel(task_id)
        return {"status": "cancelled", "task_id": task_id}

    @app.get("/api/agents")
    async def list_agents():
        """List all agents in the pool."""
        tm = main_loop.task_manager
        if not tm:
            return {"agents": []}

        agents = []
        for agent_id, agent in tm.worker_pool.agents.items():
            agents.append({
                "agent_id": agent_id,
                "status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
                "role": agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                "task_count": agent.task_count,
                "success_count": agent.success_count,
                "expertise": agent.expertise,
            })
        return {"agents": agents}

    # ============================================================
    # WebSocket: Real-time loop progress
    # ============================================================

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket):
        """WebSocket for real-time loop progress streaming.

        Client sends: {"message": "user input"}
        Server sends: {"phase": "REASON", "data": {...}, "phase": "DECOMPOSE", ...}
        """
        await ws.accept()
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)

                if "message" not in msg:
                    await ws.send_json({"error": "missing 'message'"})
                    continue

                # Run loop with progress callbacks
                ctx = LoopContext(
                    session_id=f"ws:{id(ws)}",
                    user_input=msg["message"],
                )

                # Hook into MainLoop phases to stream progress
                original_run = main_loop.run

                async def stream_phase(phase_name: str, data: dict):
                    await ws.send_json({"phase": phase_name, "data": data})

                # Execute
                try:
                    ctx = await main_loop.run(msg["message"])

                    await ws.send_json({
                        "phase": "DONE",
                        "data": {
                            "output": ctx.final_output,
                            "tasks": len(ctx.task_ids),
                            "results": len(ctx.agent_results),
                            "errors": ctx.errors,
                        }
                    })
                except Exception as e:
                    await ws.send_json({"phase": "ERROR", "data": {"error": str(e)}})

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e)

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Agent-Loop API",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": [
                "POST /api/chat",
                "GET /api/status",
                "GET /api/tasks",
                "GET /api/tasks/{id}",
                "POST /api/tasks/{id}/cancel",
                "GET /api/agents",
                "WS /ws/stream",
            ],
        }

    return app


# ============================================================
# CLI entry point
# ============================================================

def run_dev_server(host: str = "0.0.0.0", port: int = 8000,
                   memory_url: str = "surrealdb://localhost:8000",
                   llm_provider: str = "deepseek",
                   llm_api_key: str | None = None):
    """Run the API server in development mode.

    Usage:
        python -m agent_loop.web --host 0.0.0.0 --port 8000
    """
    import os
    import uvicorn

    # Initialize memory
    memory = MemoryPool(memory_url)

    # Initialize LLM
    from .llm import create_provider
    api_key = llm_api_key or os.environ.get("LLM_API_KEY", "")
    llm = create_provider(llm_provider, api_key=api_key)

    # Create app
    app = create_app(memory=memory, llm=llm)

    # Run
    uvicorn.run(app, host=host, port=port)
