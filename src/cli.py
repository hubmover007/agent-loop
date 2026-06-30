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

import yaml

logger = logging.getLogger(__name__)


def setup_logging():
    """Load logging configuration from config/logging.yaml, fallback to basicConfig."""
    import logging.config
    config_path = "config/logging.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
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
        from src.system_agents import TaskAgent, AgentManagerAgent, TaskRegistry
        from src.tools.base import ToolRegistry
        from src.tools.ssh import SSHTool
        from src.tools.web import WebTool, CodeTool
        from src.web import create_app
        from src.permissions import PermissionChecker, AgentPermissions
        from src.sandbox import SandboxManager

        print("Modules: all imported ✅")
        print(f"  + PermissionChecker loaded")
        print(f"  + SandboxManager loaded")

        reg = ToolRegistry()
        reg.register_defaults()
        print(f"Tools: {reg.tool_names}")
    except Exception as e:
        print(f"Import error: {e}")
        sys.exit(1)


def cmd_start(args):
    """Start the full Agent-Loop system."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    async def run():
        setup_logging()
        from src.memory import MemoryPool
        from src.llm import create_provider
        from src.loop_engine import LoopConfig, AgentLoop
        from src.loop_engine.main_loop import MainLoop
        from src.system_agents import (
            TaskAgent, AgentManagerAgent, TaskRegistry,
        )
        from src.llm_pool import LLMPool
        from src.cost_control import CostController
        from src.interaction import InteractionHub
        from src.agent_mailbox import MailRouter
        from src.persistence import PersistenceManager
        from src.permissions import PermissionChecker

        config_path = Path(args.config or "agent-loop.yaml")
        memory_url = args.memory_url or "surrealdb://localhost:8000"

        print("=== Agent-Loop Startup ===")
        print()

        # 1. Load config
        config = LoopConfig()
        print("[1/6] Config loaded")

        # 2. Create LLMPool + CostController
        llm_pool = LLMPool()
        try:
            llm_pool.initialize()
            print(f"[2/6] LLMPool initialized: {len(llm_pool.list_providers())} providers")
        except Exception as e:
            print(f"[2/6] LLMPool warning: {e}")

        cost_ctrl = CostController()
        print("[2/6] CostController initialized")

        # 3. Create PermissionChecker
        perm_checker = PermissionChecker()
        print("[3/6] PermissionChecker initialized")

        # 4. Create InteractionHub + MailRouter + PersistenceManager
        hub = InteractionHub()
        mail_router = MailRouter()
        persistence = PersistenceManager()
        print("[4/6] InteractionHub + MailRouter + PersistenceManager initialized")

        # 5. Create LLM and memory
        llm = create_provider(
            args.provider or "deepseek",
            api_key=args.api_key or os.environ.get("LLM_API_KEY", ""),
        )
        memory = MemoryPool(memory_url)
        print("[5/6] LLM + MemoryPool initialized")

        # 6. Create system agents
        registry = TaskRegistry()
        agent_loop = AgentLoop(
            tool_loop=None,  # will be set up later
            llm=llm,
            config=config,
        )
        task_agent = TaskAgent(llm=llm, registry=registry)
        manager = AgentManagerAgent(
            memory=memory,
            agent_loop=agent_loop,
            config=config,
            registry=registry,
            llm_pool=llm_pool,
            interaction_hub=hub,
            mail_router=mail_router,
            persistence=persistence,
        )

        # Enable AgentForker for parallel task capability
        try:
            manager.enable_forker()
            print("  + AgentForker enabled")
        except Exception as e:
            print(f"  + AgentForker skipped: {e}")

        print("[6/6] TaskAgent + AgentManagerAgent initialized")

        # 7. Show system info
        print()
        print("=== System Ready ===")
        print(f"  Providers: {len(llm_pool.list_providers())}")
        print(f"  Active agents: {len(manager.list_agents())}")
        print(f"  Risk threshold: {hub.risk_threshold}")
        print()
        print("System is ready for interactive use.")

    asyncio.run(run())


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

    # start
    p_start = sub.add_parser("start", help="Start full Agent-Loop system")
    p_start.add_argument("--config", default="agent-loop.yaml")
    p_start.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_start.add_argument("--provider", default="deepseek")
    p_start.add_argument("--api-key", default=None)
    p_start.set_defaults(func=cmd_start)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
