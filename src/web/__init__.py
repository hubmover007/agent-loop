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

from pathlib import Path

from ..core import TaskStatus, AgentStatus, LoopPhase
from ..loop_engine import LLMProvider, LoopConfig
from ..loop_engine.main_loop import MainLoop, LoopContext
from ..memory import MemoryPool
from ..system_agents import TaskAgent, AgentManagerAgent, TaskRegistry
from ..task_queue import TaskDispatcher, TaskHandle, TaskState

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
    task_dispatcher = TaskDispatcher(main_loop=main_loop)

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
    app.state.task_dispatcher = task_dispatcher
    app.state.memory = memory
    app.state.llm = llm
    app.state.config = config
    app.state.active_loops: dict[str, LoopContext] = {}
    app.state.ws_clients: list[WebSocket] = []  # Connected WebSocket clients

    # ============================================================
    # Endpoints
    # ============================================================

    @app.get("/api/status")
    async def get_status():
        """Get system status overview."""
        ta = main_loop.task_agent
        am = main_loop.agent_manager
        reg = main_loop.task_registry
        tasks = reg.all_tasks() if reg else []
        pipeline_status = {
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t.status.value == "pending"),
            "running": sum(1 for t in tasks if t.status.value == "running"),
            "done": sum(1 for t in tasks if t.status.value == "done"),
            "failed": sum(1 for t in tasks if t.status.value == "failed"),
            "inflight": len(am._inflight) if am else 0,
            "agents_active": am.pool.active_count if am and am.pool else 0,
            "agents_idle": am.pool.idle_count if am and am.pool else 0,
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

        # Use TaskDispatcher for cancellation-aware dispatch
        output = await task_dispatcher.dispatch(request.message)

        ctx = LoopContext(user_input=request.message)
        ctx.final_output = output
        app.state.active_loops[session_id] = ctx

        return ChatResponse(
            session_id=session_id,
            output=output or "",
            tasks_created=0,
            tasks_done=0,
            tasks_failed=0,
            errors=[],
        )

    @app.get("/api/tasks")
    async def list_tasks():
        """List all tasks in the registry."""
        reg = main_loop.task_registry
        if not reg:
            return {"tasks": []}
        return {"tasks": [t.to_dict() for t in reg.all_tasks()]}

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get detailed task information."""
        reg = main_loop.task_registry
        if not reg:
            return JSONResponse({"error": "No task registry"}, status_code=503)

        task = reg.get(task_id)
        if not task:
            return JSONResponse({"error": "Task not found"}, status_code=404)

        result = {**task.to_dict()}
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
        am = main_loop.agent_manager
        if not am:
            return JSONResponse({"error": "No agent manager"}, status_code=503)
        await am.cancel(task_id)
        return {"status": "cancelled", "task_id": task_id}

    @app.get("/api/agents")
    async def list_agents():
        """List all agents in the pool."""
        am = main_loop.agent_manager
        if not am or not am.pool:
            return {"agents": []}

        agents = []
        for agent_id, agent in am.pool.agents.items():
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

    @app.get("/api/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        from ..metrics import get_collector
        from fastapi.responses import PlainTextResponse
        text = get_collector().format_prometheus()
        return PlainTextResponse(content=text)

    @app.get("/metrics")
    async def get_metrics_prom():
        """Prometheus metrics endpoint (standard path)."""
        from ..metrics import get_collector
        from fastapi.responses import PlainTextResponse
        text = get_collector().format_prometheus()
        return PlainTextResponse(content=text)

    @app.get("/api/traces")
    async def get_traces():
        """Get distributed traces from the Tracer."""
        tracer = getattr(main_loop, 'tracer', None)
        if tracer is None:
            return {"traces": []}
        return {"traces": tracer.get_traces()}

    @app.get("/api")
    async def api_info():
        """API info endpoint."""
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
                "GET /api/traces",
                "GET /metrics",
                "WS /ws/stream",
            ],
        }

    # Static file serving for frontend UI (must be last, after all routes)
    from starlette.staticfiles import StaticFiles
    web_root = Path(__file__).parent.parent.parent / "web"
    if web_root.exists():
        app.mount("/", StaticFiles(directory=str(web_root), html=True), name="static")

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
