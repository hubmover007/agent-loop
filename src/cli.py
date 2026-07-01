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


def cmd_anchor_list(args):
    """List all anchor files."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.memory.anchor import AnchorManager
    mgr = AnchorManager(args.anchor_dir)
    names = mgr.list_anchors()

    if not names:
        print("No anchor files found.")
        return

    print(f"Anchor files ({len(names)}):")
    for name in names:
        anchor = mgr.read_anchor(name)
        entry_count = len(anchor.entries) if anchor else 0
        print(f"  {name}.md — {entry_count} entries")


def cmd_anchor_show(args):
    """Show contents of an anchor file."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.memory.anchor import AnchorManager
    mgr = AnchorManager(args.anchor_dir)
    anchor = mgr.read_anchor(args.name)

    if not anchor:
        print(f"Anchor '{args.name}' not found.")
        sys.exit(1)

    print(anchor.to_markdown())


def cmd_anchor_sync(args):
    """Sync anchor files to SurrealDB fact table."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    async def run():
        from src.memory import MemoryPool
        from src.memory.anchor import AnchorManager

        memory = MemoryPool(args.memory_url)
        await memory.connect()

        mgr = AnchorManager(args.anchor_dir, memory_pool=memory)
        count = await mgr.sync_to_db(name=args.name or None)
        print(f"Synced {count} anchor facts to DB.")

        await memory.disconnect()

    asyncio.run(run())


def cmd_anchor_lookup(args):
    """Look up a specific key in an anchor file."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.memory.anchor import AnchorManager
    mgr = AnchorManager(args.anchor_dir)
    value = mgr.lookup(args.name, args.key)

    if value is None:
        print(f"Key '{args.key}' not found in anchor '{args.name}'.")
        sys.exit(1)

    print(f"{args.name}.{args.key} = {value}")


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


def cmd_consolidate(args):
    """Run LLM-driven memory consolidation."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    async def run():
        from src.memory import MemoryPool

        memory = MemoryPool(args.memory_url)
        if args.memory_url.startswith("surrealdb://") or args.memory_url.startswith("ws://"):
            await memory.connect()

        # Optional LLM provider
        llm = None
        if args.provider:
            from src.llm import create_provider
            llm = create_provider(
                args.provider,
                api_key=args.api_key or os.environ.get("LLM_API_KEY", ""),
            )

        print(f"=== Memory Consolidation ===")
        if args.dry_run:
            print("(DRY RUN — no changes will be made)")

        if args.dry_run:
            # Just count unconsolidated episodes
            episodes = await memory.get_unconsolidated_episodes(limit=50)
            print(f"Unconsolidated episodes: {len(episodes)}")
            if episodes:
                unconsolidated = [e for e in episodes if not e.get("consolidated", False)]
                print(f"  Pending consolidation: {len(unconsolidated)}")
                for ep in unconsolidated[:5]:
                    print(f"  - {ep.get('title', '?')[:60]}")
                if len(unconsolidated) > 5:
                    print(f"  ... and {len(unconsolidated) - 5} more")
            print(f"Min episodes threshold: {args.min_episodes}")
            print(f"Would run: {'NO' if len(episodes) < args.min_episodes else 'YES'}")
        else:
            result = await memory.consolidate(
                llm_provider=llm,
                min_episodes=args.min_episodes,
                max_episodes_per_run=args.max_episodes,
                prune_threshold=args.prune_threshold,
                enable_linking=not args.no_linking,
                enable_resolution=not args.no_resolution,
                enable_pruning=not args.no_pruning,
            )
            print(f"\nResult: {result.to_summary()}")
            if result.errors:
                for err in result.errors:
                    print(f"  Error: {err}")

        if args.memory_url.startswith("surrealdb://") or args.memory_url.startswith("ws://"):
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

    # anchor
    p_anchor = sub.add_parser("anchor", help="Manage anchor files")
    p_anchor_sub = p_anchor.add_subparsers(dest="anchor_command")

    p_anchor_list = p_anchor_sub.add_parser("list", help="List all anchor files")
    p_anchor_list.add_argument("--anchor-dir", default="state/anchors")
    p_anchor_list.set_defaults(func=cmd_anchor_list)

    p_anchor_show = p_anchor_sub.add_parser("show", help="Show anchor file contents")
    p_anchor_show.add_argument("name", help="Anchor name (without .md)")
    p_anchor_show.add_argument("--anchor-dir", default="state/anchors")
    p_anchor_show.set_defaults(func=cmd_anchor_show)

    p_anchor_sync = p_anchor_sub.add_parser("sync", help="Sync anchors to SurrealDB")
    p_anchor_sync.add_argument("--name", default=None, help="Specific anchor to sync (default: all)")
    p_anchor_sync.add_argument("--anchor-dir", default="state/anchors")
    p_anchor_sync.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_anchor_sync.set_defaults(func=cmd_anchor_sync)

    p_anchor_lookup = p_anchor_sub.add_parser("lookup", help="Look up a key in an anchor")
    p_anchor_lookup.add_argument("name", help="Anchor name (without .md)")
    p_anchor_lookup.add_argument("key", help="Key to look up")
    p_anchor_lookup.add_argument("--anchor-dir", default="state/anchors")
    p_anchor_lookup.set_defaults(func=cmd_anchor_lookup)

    # cleanup
    p_cleanup = sub.add_parser("cleanup", help="Clean up stale episodes from memory")
    p_cleanup.add_argument("--days", type=int, default=30,
                           help="Delete episodes older than N days (default: 30)")
    p_cleanup.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_cleanup.set_defaults(func=cmd_cleanup)

    # consolidate
    p_cons = sub.add_parser("consolidate", help="Run LLM-driven memory consolidation")
    p_cons.add_argument("--dry-run", action="store_true",
                        help="Show what would be consolidated without making changes")
    p_cons.add_argument("--min-episodes", type=int, default=3,
                        help="Minimum episodes before consolidation (default: 3)")
    p_cons.add_argument("--max-episodes", type=int, default=50,
                        help="Maximum episodes per run (default: 50)")
    p_cons.add_argument("--prune-threshold", type=float, default=0.3,
                        help="Confidence below which memories are pruned (default: 0.3)")
    p_cons.add_argument("--no-linking", action="store_true",
                        help="Disable graph edge creation phase")
    p_cons.add_argument("--no-resolution", action="store_true",
                        help="Disable contradiction resolution phase")
    p_cons.add_argument("--no-pruning", action="store_true",
                        help="Disable memory pruning phase")
    p_cons.add_argument("--memory-url", default="surrealdb://localhost:8000")
    p_cons.add_argument("--provider", default=None,
                        help="LLM provider for extraction (default: none, uses heuristic)")
    p_cons.add_argument("--api-key", default=None)
    p_cons.set_defaults(func=cmd_consolidate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
