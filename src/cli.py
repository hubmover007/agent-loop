#!/usr/bin/env python3
"""Agent-Loop CLI — command line interface.

Usage:
  agent-loop chat "your question"
  agent-loop serve --port 8000
  agent-loop status
  agent-loop init-config
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_init_config(args):
    """Initialize configuration file."""
    config_path = Path(args.config or "agent-loop.yaml")
    if config_path.exists():
        print(f"Config already exists: {config_path}")
        return

    template = """# Agent-Loop Configuration
memory:
  url: "surrealdb://localhost:8000"
  namespace: "agent_loop"
  database: "main"

llm:
  provider: "deepseek"  # deepseek | anthropic-bedrock | openai-compatible
  api_key: "${DEEPSEEK_API_KEY}"
  model: "deepseek-chat"

loop:
  max_reason_loops: 8
  reason_confidence_threshold: 0.85
  max_agent_concurrent: 10
  accept_threshold: 0.7

tools:
  brave_api_key: "${BRAVE_API_KEY}"
  code_workspace: "/tmp/agent_loop/workspace"

server:
  host: "0.0.0.0"
  port: 8000
"""
    config_path.write_text(template)
    print(f"Config created: {config_path}")


def cmd_chat(args):
    """Run a single chat through the Loop Engine."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    async def run():
        from src.memory import MemoryPool
        from src.llm import create_provider
        from src.loop_engine import LoopConfig
        from src.loop_engine.main_loop import MainLoop

        # Initialize
        memory = MemoryPool(args.memory_url)
        llm = create_provider(
            args.provider,
            api_key=args.api_key or os.environ.get("LLM_API_KEY", ""),
        )
        config = LoopConfig()
        loop = MainLoop(memory=memory, llm=llm, config=config)

        # Run
        print(f"User: {args.message}")
        print("---")
        ctx = await loop.run(args.message)

        print(f"Output: {ctx.final_output}")
        print(f"---")
        print(f"Tasks: {len(ctx.task_ids)} created, {len(ctx.agent_results)} done")
        if ctx.errors:
            print(f"Errors: {ctx.errors}")

    asyncio.run(run())


def cmd_serve(args):
    """Start the API server."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import uvicorn
    from src.memory import MemoryPool
    from src.llm import create_provider
    from src.loop_engine import LoopConfig
    from src.web import create_app

    memory = MemoryPool(args.memory_url)
    llm = create_provider(
        args.provider,
        api_key=args.api_key or os.environ.get("LLM_API_KEY", ""),
    )
    config = LoopConfig()
    app = create_app(memory=memory, llm=llm, config=config)

    print(f"Agent-Loop API serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_status(args):
    """Check system status."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.core import TaskStatus, AgentStatus, LoopPhase, MemoryLayer

    print("=== Agent-Loop System Status ===")
    print()
    print("Core Types:")
    print(f"  Task States: {[s.value for s in TaskStatus]}")
    print(f"  Agent States: {[s.value for s in AgentStatus]}")
    print(f"  Loop Phases: {[p.value for p in LoopPhase]}")
    print(f"  Memory Layers: {[l.value for l in MemoryLayer]}")
    print()

    # Check imports
    try:
        from src.memory import MemoryPool
        from src.loop_engine.main_loop import MainLoop
        from src.task_manager import TaskManagerAgent
        from src.tools.base import ToolRegistry
        from src.tools.ssh import SSHTool
        from src.tools.web import WebTool, CodeTool
        from src.web import create_app

        print("Modules: all imported ✅")

        reg = ToolRegistry()
        reg.register_defaults()
        print(f"Tools: {reg.tool_names}")
    except Exception as e:
        print(f"Import error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="agent-loop",
        description="Agent-Loop: Loop Engine for AI Agent orchestration",
    )
    sub = parser.add_subparsers(dest="command")

    # init-config
    p_init = sub.add_parser("init-config", help="Initialize configuration file")
    p_init.add_argument("--config", default="agent-loop.yaml")
    p_init.set_defaults(func=cmd_init_config)

    # chat
    p_chat = sub.add_parser("chat", help="Run a single chat through Loop Engine")
    p_chat.add_argument("message", help="User input message")
    p_chat.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_chat.add_argument("--provider", default="deepseek")
    p_chat.add_argument("--api-key", default=None)
    p_chat.set_defaults(func=cmd_chat)

    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_serve.add_argument("--provider", default="deepseek")
    p_serve.add_argument("--api-key", default=None)
    p_serve.set_defaults(func=cmd_serve)

    # status
    p_status = sub.add_parser("status", help="Check system status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
