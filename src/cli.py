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


def cmd_init_config(args):
    """Initialize configuration files and directory structure."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Create agents/ directory structure
    agents_dir = config_dir / "agents"
    for sub in ["shared", "cards/personalities", "cards/roles", "private"]:
        (agents_dir / sub).mkdir(parents=True, exist_ok=True)

    # Create llm_pool.json template if missing
    llm_pool_path = config_dir / "llm_pool.json"
    if not llm_pool_path.exists():
        llm_pool_path.write_text(json.dumps({
            "providers": {
                "deepseek": {
                    "provider": "deepseek",
                    "api_key": "${DEEPSEEK_API_KEY}",
                    "model": "deepseek-chat",
                    "capabilities": ["general", "coding", "reasoning"],
                    "max_tokens": 8192
                }
            },
            "strategies": {
                "balanced": {"prefer": "deepseek"},
                "cheapest": {"prefer": "deepseek"},
                "most_capable": {"prefer": "deepseek"}
            }
        }, indent=2))
        print(f"Created: {llm_pool_path}")

    # Create permissions.json template if missing
    perm_path = config_dir / "permissions.json"
    if not perm_path.exists():
        perm_path.write_text(json.dumps({
            "templates": {
                "coder": {
                    "trust_level": "restricted",
                    "filesystem": {
                        "read_paths": ["state/**", "config/**", "*.md"],
                        "write_paths": ["state/agents/{agent_id}/**"],
                        "blocked_paths": ["~/.aws/**", "~/.openclaw/.env"]
                    },
                    "network": {"allowed_hosts": ["pypi.org"], "blocked_hosts": ["*"]},
                    "shell": {"allowed": True, "allowed_commands": ["git", "python3"], "timeout_seconds": 30},
                    "rate_limit": {"max_calls_per_minute": 60, "max_tokens_per_hour": 100000}
                },
                "admin": {
                    "trust_level": "admin",
                    "filesystem": {"read_paths": ["**"], "write_paths": ["**"]},
                    "network": {"allowed_hosts": ["*"]},
                    "shell": {"allowed": True},
                    "rate_limit": {"max_calls_per_minute": 200, "max_tokens_per_hour": 500000}
                }
            },
            "elevation": {"requires_approval": True, "min_tasks_for_elevation": 10, "min_success_rate": 0.7}
        }, indent=2))
        print(f"Created: {perm_path}")

    # Create agent-loop.yaml template if missing
    yaml_path = Path(args.config or "agent-loop.yaml")
    if not yaml_path.exists():
        yaml_path.write_text("""# Agent-Loop Configuration
memory:
  url: "surrealdb://localhost:8000"
  namespace: "agent_loop"
  database: "main"

llm:
  provider: "deepseek"
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
""")
        print(f"Created: {yaml_path}")

    print("Agent-Loop config initialized successfully.")


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


def cmd_stats(args):
    """Show memory stats."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    async def run():
        from src.memory import MemoryPool

        memory = MemoryPool(args.memory_url)
        await memory.connect()

        print("=== Memory Pool Stats ===")
        print()

        # Count facts
        facts = await memory.query_facts(limit=10000)
        entity_count = sum(1 for f in facts if f.get("fact_type") == "entity")
        fp_count = sum(1 for f in facts if f.get("fact_type") == "facetpoint")
        print(f"Facts: {len(facts)} total ({entity_count} entities, {fp_count} facetpoints)")

        # Count episodes
        episodes = await memory._db.query("SELECT count() FROM episode GROUP ALL")
        ep_rows = episodes if isinstance(episodes, list) else episodes.get("result", [])
        if ep_rows:
            print(f"Episodes: {ep_rows[0].get('count', 0)}")

        await memory.disconnect()

    asyncio.run(run())


def cmd_cleanup(args):
    """Clean up stale episodes from memory."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    async def run():
        from src.memory import MemoryPool

        memory = MemoryPool(args.memory_url)
        await memory.connect()

        print(f"=== Memory Cleanup (older than {args.days} days) ===")

        # Step 1: Consolidate unconsolidated old episodes into facts
        cons_result = await memory.consolidate_episodes_to_facts(days=args.days)
        print(f"Episodes consolidated → facts: {cons_result['consolidated']}")

        # Step 2: Delete stale episodes
        deleted = await memory.cleanup_stale_episodes(days=args.days)
        print(f"Stale episodes deleted: {deleted}")

        print("Cleanup complete.")
        await memory.disconnect()

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

    # stats
    p_stats = sub.add_parser("stats", help="Show memory pool stats")
    p_stats.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_stats.set_defaults(func=cmd_stats)

    # cleanup
    p_cleanup = sub.add_parser("cleanup", help="Clean up stale episodes from memory")
    p_cleanup.add_argument("--days", type=int, default=30,
                           help="Delete episodes older than N days (default: 30)")
    p_cleanup.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_cleanup.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
