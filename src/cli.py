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
import subprocess
import sys
from datetime import datetime
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


def _load_env() -> None:
    """Load .env file into os.environ. Idempotent (setdefault)."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_api_key(api_key_source: str) -> str:
    """Resolve API key from source string like 'env:EASYROUTER_API_KEY'.

    Supported formats:
        - "env:VAR_NAME" → read from os.environ
        - "none" → return "not-needed"
        - otherwise → return as-is (literal key)
    """
    if not api_key_source:
        return ""
    if api_key_source.startswith("env:"):
        env_var = api_key_source[4:]
        return os.environ.get(env_var, "")
    if api_key_source == "none":
        return "not-needed"
    return api_key_source  # literal key


def _create_llm_from_config(args: argparse.Namespace) -> "LLMProvider":
    """Create LLM provider from config, preferring LLMPool if available.

    Priority:
    1. config/llm_pool.json exists → load first enabled provider via OpenAICompatibleProvider
    2. args.provider is set → create_provider()
    3. Default → easyrouter with EASYROUTER_API_KEY env var
    """
    from src.llm_pool import LLMPool
    from src.llm import create_provider, OpenAICompatibleProvider

    # Try LLMPool first
    pool_path = Path("config/llm_pool.json")
    if pool_path.exists():
        try:
            pool = LLMPool(str(pool_path))
            pool.initialize()
            if pool._pool_config:
                for cfg in pool._pool_config.providers:
                    if cfg.enabled:
                        api_key = _resolve_api_key(cfg.api_key_source)
                        if api_key:
                            logger.info("Using LLM from LLMPool: %s (%s)", cfg.id, cfg.model)
                            return OpenAICompatibleProvider(
                                base_url=cfg.endpoint,
                                api_key=api_key,
                                default_model=cfg.model,
                            )
        except Exception as e:
            logger.warning("LLMPool load failed: %s, falling back", e)

    # Fallback: create_provider
    provider_type = args.provider or "easyrouter"
    if provider_type == "easyrouter":
        api_key = args.api_key or os.environ.get("EASYROUTER_API_KEY", "")
        logger.info("Using EasyRouter provider (fallback)")
        return create_provider("easyrouter", api_key=api_key)
    else:
        api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
        logger.info("Using provider: %s", provider_type)
        return create_provider(provider_type, api_key=api_key)


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
        from src.loop_engine import LoopConfig
        from src.loop_engine.main_loop import MainLoop

        # ── Load environment variables ──
        _load_env()

        # ── Initialize MemoryPool ──
        memory = MemoryPool(args.memory_url)
        if args.memory_url.startswith("ws://") or args.memory_url.startswith("surrealdb://"):
            await memory.connect()
            await memory.initialize_schema()

        # ── Initialize LLM (LLMPool preferred, fallback create_provider) ──
        llm = _create_llm_from_config(args)

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

        # Cleanup
        if hasattr(memory, 'disconnect'):
            await memory.disconnect()

    asyncio.run(run())


def cmd_serve(args):
    """Start the API server."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # ── Load environment variables ──
    _load_env()

    import uvicorn
    from src.memory import MemoryPool
    from src.loop_engine import LoopConfig
    from src.web import create_app

    memory = MemoryPool(args.memory_url)
    # MemoryPool will be connected on first use via the app lifecycle

    llm = _create_llm_from_config(args)
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

        # ── Load environment variables ──
        _load_env()

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
        memory_url = args.memory_url or "ws://localhost:8001/rpc"

        print("=== Agent-Loop Startup ===")
        print()

        # 1. Load config
        config = LoopConfig()
        print("[1/6] Config loaded")

        # 2. Create LLMPool + CostController
        llm_pool = LLMPool()
        pool_path = Path("config/llm_pool.json")
        if pool_path.exists():
            try:
                llm_pool.initialize()
                print(f"[2/6] LLMPool initialized: {len(llm_pool.list_providers())} providers")
            except Exception as e:
                print(f"[2/6] LLMPool warning: {e}")
        else:
            print(f"[2/6] LLMPool: no config/llm_pool.json found")

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
        llm = _create_llm_from_config(args)
        memory = MemoryPool(memory_url)
        if memory_url.startswith("ws://") or memory_url.startswith("surrealdb://"):
            try:
                await memory.connect()
                await memory.initialize_schema()
                print("[5/6] LLM + MemoryPool initialized (connected)")
            except Exception as e:
                print(f"[5/6] LLM initialized, MemoryPool connect failed: {e}")
        else:
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
        print(f"  Providers: {len(llm_pool.list_providers()) if pool_path.exists() else 0}")
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


def cmd_setup(args):
    """Interactive setup wizard for first-time configuration.

    Guides user through:
    1. LLM provider selection (EasyRouter / OpenAI / DeepSeek / local)
    2. API key input
    3. SurrealDB configuration (Docker / existing / skip)
    4. Embedding configuration
    5. Basic preferences (timezone, language)
    6. Write config files + .env
    7. Verify connectivity
    8. Initialize DB schema
    """
    print("🐕 Agent-Loop 配置向导")
    print("=" * 40)

    # ── 1. LLM Provider ──
    print("\n📋 1. LLM 配置")
    print("  EasyRouter: 聚合平台，支持 deepseek/gpt/gemini/glm/kimi 等 8+ 模型")
    print("  OpenAI: 直连 OpenAI API")
    print("  DeepSeek: 直连 DeepSeek API（便宜）")
    print("  Local: 本地模型（需额外配置）")

    provider = input("\n选择 LLM provider [easyrouter/openai/deepseek/local] (默认 easyrouter): ").strip() or "easyrouter"

    config = {"provider": provider, "models": []}

    if provider == "easyrouter":
        api_key = input("EasyRouter API Key: ").strip()
        if not api_key:
            print("⚠️  未输入 API Key，稍后可手动配置 .env")
        else:
            config["api_key"] = api_key

        print("\n可选模型:")
        models = [
            ("deepseek-v4-pro", "推理主力（推荐）"),
            ("deepseek-v4-flash", "快速轻量"),
            ("gpt-5.5", "GPT 推理+视觉"),
            ("gemini-2.5-flash", "Gemini 视觉"),
            ("glm-5-turbo", "GLM 快速"),
            ("kimi-k2.6", "Kimi 推理"),
            ("glm-5.1", "GLM 文本"),
            ("glm-5.2", "GLM 文本"),
        ]
        for i, (mid, desc) in enumerate(models, 1):
            print(f"  {i}. {mid} — {desc}")

        print("\n推荐全选（8个模型都已验证可用）")
        config["models"] = [m[0] for m in models]

    elif provider == "openai":
        api_key = input("OpenAI API Key: ").strip()
        config["api_key"] = api_key
        config["models"] = ["gpt-4o", "gpt-4o-mini"]

    elif provider == "deepseek":
        api_key = input("DeepSeek API Key: ").strip()
        config["api_key"] = api_key
        config["models"] = ["deepseek-chat", "deepseek-reasoner"]

    elif provider == "local":
        print("本地模型需要 ollama 或 vLLM，请参考文档配置")
        config["endpoint"] = input("本地 API endpoint (默认 http://localhost:11434/v1): ").strip() or "http://localhost:11434/v1"
        config["models"] = ["llama3"]

    # ── 2. SurrealDB ──
    print("\n📋 2. SurrealDB 配置")
    surreal_choice = input("SurrealDB 状态: [1]已有(默认) [2]Docker启动 [3]跳过: ").strip() or "1"

    surreal_config = {"url": "surrealdb://localhost:8001", "namespace": "agent_loop", "database": "main"}

    if surreal_choice == "2":
        print("启动 SurrealDB Docker 容器...")
        subprocess.run([
            "docker", "run", "-d", "--name", "agent-loop-surrealdb",
            "-p", "8001:8000",
            "surrealdb/surrealdb:latest",
            "start", "--user", "root", "--pass", "root"
        ], check=True)
        print("✅ SurrealDB 已启动 (localhost:8001)")
    elif surreal_choice == "3":
        surreal_config["enabled"] = False
        print("⚠️  跳过 SurrealDB，将使用 SQLite 后端")
    else:
        custom_url = input(f"SurrealDB URL (默认 {surreal_config['url']}): ").strip()
        if custom_url:
            surreal_config["url"] = custom_url

    # ── 3. Embedding ──
    print("\n📋 3. Embedding 配置")
    print("  mock: 测试用，不需要 API（推荐初次使用）")
    print("  openai: 用 OpenAI text-embedding-3-small")
    print("  local: 本地 sentence-transformers")

    emb_provider = input("选择 embedding [mock/openai/local] (默认 mock): ").strip() or "mock"
    emb_config = {"provider": emb_provider, "dimensions": 1536}

    if emb_provider == "openai":
        emb_key = input("OpenAI API Key (可复用上面的): ").strip() or config.get("api_key", "")
        emb_config["api_key"] = emb_key

    # ── 4. 基本偏好 ──
    print("\n📋 4. 基本配置")
    timezone = input("时区 (默认 Asia/Shanghai): ").strip() or "Asia/Shanghai"
    language = input("语言 [zh/en] (默认 zh): ").strip() or "zh"

    # ── 5. 写配置文件 ──
    print("\n📝 写入配置文件...")

    # 5a. .env 文件（已存在则追加不覆盖）
    env_path = Path(".env")
    existing_env = set()
    if env_path.exists():
        existing_env = set(line.split("=", 1)[0] for line in env_path.read_text().strip().split("\n") if line.strip() and not line.startswith("#") and "=" in line)

    new_env_lines = []
    if provider == "easyrouter" and "api_key" in config:
        if "EASYROUTER_API_KEY" not in existing_env:
            new_env_lines.append(f"EASYROUTER_API_KEY={config['api_key']}")
    if provider == "openai" and "api_key" in config:
        if "OPENAI_API_KEY" not in existing_env:
            new_env_lines.append(f"OPENAI_API_KEY={config['api_key']}")
    if provider == "deepseek" and "api_key" in config:
        if "DEEPSEEK_API_KEY" not in existing_env:
            new_env_lines.append(f"DEEPSEEK_API_KEY={config['api_key']}")
    if emb_provider == "openai" and "api_key" in emb_config:
        if "OPENAI_API_KEY" not in existing_env:
            new_env_lines.append(f"OPENAI_API_KEY={emb_config['api_key']}")

    if "LOG_LEVEL" not in existing_env:
        new_env_lines.append("LOG_LEVEL=INFO")
    if "TZ" not in existing_env:
        new_env_lines.append(f"TZ={timezone}")

    if new_env_lines:
        if env_path.exists():
            with open(env_path, "a") as f:
                f.write("\n".join(new_env_lines) + "\n")
            print(f"  ✅ {env_path} (已追加)")
        else:
            env_path.write_text("\n".join(new_env_lines) + "\n")
            print(f"  ✅ {env_path} (已创建)")
    else:
        print(f"  ✅ {env_path} (无需更新)")

    # 5b. agent-loop.yaml
    yaml_config = {
        "memory": {
            "url": surreal_config["url"],
            "namespace": surreal_config["namespace"],
            "database": surreal_config["database"],
            "embedding": emb_config,
        },
        "llm": {
            "provider": provider,
            "endpoint": config.get("endpoint", ""),
        },
        "loop": {
            "max_reason_loops": 8,
            "reason_confidence_threshold": 0.85,
            "max_agent_concurrent": 10,
            "accept_threshold": 0.7,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
        },
    }

    yaml_path = Path("agent-loop.yaml")
    yaml_path.write_text(yaml.dump(yaml_config, default_flow_style=False, allow_unicode=True))
    print(f"  ✅ {yaml_path}")

    # 5c. config/llm_pool.json
    _generate_llm_pool(config, provider)
    print(f"  ✅ config/llm_pool.json")

    # 5d. 创建目录
    for d in ["state/anchors", "state/agents", "logs", "config/agents"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  ✅ 目录结构")

    # ── 6. 验证连通性 ──
    print("\n🔌 验证连通性...")

    if provider == "easyrouter" and "api_key" in config:
        try:
            import httpx
            resp = httpx.get(
                "https://easyrouter.io/v1/models",
                headers={"Authorization": f"Bearer {config['api_key']}"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                print("  ✅ EasyRouter API 连通")
            else:
                print(f"  ⚠️  EasyRouter API 返回 {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️  EasyRouter API 连接失败: {e}")

    if surreal_config.get("enabled", True):
        try:
            async def check_db():
                from surrealdb import AsyncSurreal
                db = AsyncSurreal(surreal_config["url"])
                await db.connect()
                await db.signin({"user": "root", "pass": "root"})
                await db.use(surreal_config["namespace"], surreal_config["database"])
                await db.query("SELECT 1")
            asyncio.run(check_db())
            print("  ✅ SurrealDB 连通")
        except Exception as e:
            print(f"  ⚠️  SurrealDB 连接失败: {e}")

    # ── 7. 初始化 DB Schema ──
    if surreal_config.get("enabled", True):
        print("\n🗄️  初始化数据库 Schema...")
        try:
            async def init_schema():
                from surrealdb import AsyncSurreal
                db = AsyncSurreal(surreal_config["url"])
                await db.connect()
                await db.signin({"user": "root", "pass": "root"})
                await db.use(surreal_config["namespace"], surreal_config["database"])

                schema_path = Path("src/memory/schema.surql")
                if schema_path.exists():
                    schema_sql = schema_path.read_text()
                    for statement in schema_sql.split(";\n"):
                        lines = [l for l in statement.split("\n")
                                 if l.strip() and not l.strip().startswith("--")]
                        stmt = "\n".join(lines).strip()
                        if stmt:
                            try:
                                await db.query(stmt)
                            except Exception:
                                pass  # ignore "already exists"
                print("  ✅ Schema 初始化完成")
            asyncio.run(init_schema())
        except Exception as e:
            print(f"  ⚠️  Schema 初始化失败: {e}")

    # ── 8. 创建默认锚点 ──
    print("\n📍 创建默认锚点...")
    _create_default_anchors(timezone, language)
    print("  ✅ state/anchors/")

    # ── 完成 ──
    print("\n" + "=" * 40)
    print("✅ 配置完成！")
    print("\n🚀 启动系统:")
    print("  agent-loop start         # 启动完整系统")
    print("  agent-loop serve         # 只启动 API 服务")
    print("  agent-loop chat \"你好\"    # 单次对话")
    print("  agent-loop status        # 查看状态")
    print("  agent-loop test          # 跑测试")
    print("\n📖 文档: https://github.com/hubmover007/agent-loop")


def _generate_llm_pool(config, provider):
    """Generate llm_pool.json based on provider choice."""
    if provider == "easyrouter":
        models_config = [
            {"id": "easyrouter-deepseek-v4-pro", "model": "deepseek-v4-pro",
             "capabilities": ["general", "coding", "reasoning", "analysis"],
             "modality": ["text"], "tags": ["primary", "reasoning"]},
            {"id": "easyrouter-deepseek-v4-flash", "model": "deepseek-v4-flash",
             "capabilities": ["general", "coding", "quick"],
             "modality": ["text"], "tags": ["fast", "cheap"]},
            {"id": "easyrouter-gpt-5.5", "model": "gpt-5.5",
             "capabilities": ["general", "coding", "reasoning", "vision"],
             "modality": ["text", "image"], "tags": ["vision", "reasoning"]},
            {"id": "easyrouter-gemini-2.5-flash", "model": "gemini-2.5-flash",
             "capabilities": ["general", "vision"],
             "modality": ["text", "image"], "tags": ["vision"]},
            {"id": "easyrouter-glm-5-turbo", "model": "glm-5-turbo",
             "capabilities": ["general", "quick"],
             "modality": ["text"], "tags": ["fast", "cheap"]},
            {"id": "easyrouter-kimi-k2.6", "model": "kimi-k2.6",
             "capabilities": ["general", "reasoning"],
             "modality": ["text"], "tags": ["reasoning"]},
            {"id": "easyrouter-glm-5.1", "model": "glm-5.1",
             "capabilities": ["general"],
             "modality": ["text"], "tags": []},
            {"id": "easyrouter-glm-5.2", "model": "glm-5.2",
             "capabilities": ["general"],
             "modality": ["text"], "tags": []},
        ]
        providers = []
        for m in models_config:
            providers.append({
                "id": m["id"],
                "type": "openai",
                "endpoint": "https://easyrouter.io/v1",
                "model": m["model"],
                "api_key_source": "env:EASYROUTER_API_KEY",
                "capabilities": m["capabilities"],
                "modality": m["modality"],
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
                "max_concurrent": 10,
                "verified": False,
                "enabled": True,
                "tags": m["tags"],
            })
    elif provider == "openai":
        providers = [
            {"id": "openai-gpt-4o", "type": "openai", "endpoint": "https://api.openai.com/v1",
             "model": "gpt-4o", "api_key_source": "env:OPENAI_API_KEY",
             "capabilities": ["general", "coding", "reasoning", "vision"],
             "modality": ["text", "image"], "enabled": True, "tags": ["primary"]},
            {"id": "openai-gpt-4o-mini", "type": "openai", "endpoint": "https://api.openai.com/v1",
             "model": "gpt-4o-mini", "api_key_source": "env:OPENAI_API_KEY",
             "capabilities": ["general", "quick"],
             "modality": ["text"], "enabled": True, "tags": ["fast"]},
        ]
    elif provider == "deepseek":
        providers = [
            {"id": "deepseek-chat", "type": "openai", "endpoint": "https://api.deepseek.com/v1",
             "model": "deepseek-chat", "api_key_source": "env:DEEPSEEK_API_KEY",
             "capabilities": ["general", "coding"], "modality": ["text"],
             "enabled": True, "tags": ["primary"]},
            {"id": "deepseek-reasoner", "type": "openai", "endpoint": "https://api.deepseek.com/v1",
             "model": "deepseek-reasoner", "api_key_source": "env:DEEPSEEK_API_KEY",
             "capabilities": ["reasoning"], "modality": ["text"],
             "enabled": True, "tags": ["reasoning"]},
        ]
    else:  # local
        providers = [
            {"id": "local-default", "type": "openai", "endpoint": "http://localhost:11434/v1",
             "model": "llama3", "api_key_source": "none",
             "capabilities": ["general"], "modality": ["text"],
             "enabled": True, "tags": ["local"]},
        ]

    pool_config = {
        "providers": providers,
        "selection": {
            "default_strategy": "balanced",
            "task_mapping": {
                "reasoning": "most_capable",
                "coding": "cheapest",
                "quick": "cheapest",
                "general": "balanced",
            },
            "strategies": {
                "cheapest": {"sort_by": "cost_per_1m_input", "ascending": True},
                "most_capable": {"sort_by": "cost_per_1m_input", "ascending": False},
                "balanced": {"weights": {"cost": 0.5, "capability_match": 0.5}},
            },
        },
    }

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    (config_dir / "llm_pool.json").write_text(json.dumps(pool_config, indent=2))


def _create_default_anchors(timezone, language):
    """Create default anchor files during setup."""
    anchor_dir = Path("state/anchors")
    anchor_dir.mkdir(parents=True, exist_ok=True)

    system_md = f"""# System Anchor

## Config
- **timezone**: {timezone}
- **language**: {language}
- **created_at**: {datetime.now().isoformat()}
"""
    (anchor_dir / "system.md").write_text(system_md)


def cmd_test(args):
    """Run test suite."""
    cmd = ["python3", "-m", "pytest", "tests/", "-q"]
    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    subprocess.run(cmd)


def cmd_doctor(args):
    """System health check."""
    print("🐕 Agent-Loop Doctor")
    print("=" * 40)

    checks = []

    # 1. Python
    checks.append(("Python 3.12+", sys.version_info >= (3, 12), sys.version))

    # 2. Dependencies
    try:
        import httpx  # noqa: F811
        import pydantic, jsonschema  # noqa: F401
        checks.append(("Core dependencies", True, "all imported"))
    except ImportError as e:
        checks.append(("Core dependencies", False, str(e)))

    # 3. SurrealDB
    try:
        async def check_db():
            from surrealdb import AsyncSurreal
            db = AsyncSurreal("ws://localhost:8001/rpc")
            await db.connect()
            await db.signin({"user": "root", "pass": "root"})
            await db.use("agent_loop", "main")
            await db.query("SELECT 1")
        asyncio.run(asyncio.wait_for(check_db(), timeout=5))
        checks.append(("SurrealDB", True, "localhost:8001"))
    except Exception as e:
        checks.append(("SurrealDB", False, str(e)[:80]))

    # 4. Config files
    for f in ["agent-loop.yaml", "config/llm_pool.json", ".env"]:
        checks.append((f"Config: {f}", Path(f).exists(), str(Path(f))))

    # 5. LLM API
    env_key = os.environ.get("EASYROUTER_API_KEY", "")
    if env_key:
        try:
            import httpx
            resp = httpx.get("https://easyrouter.io/v1/models",
                           headers={"Authorization": f"Bearer {env_key}"}, timeout=5)
            checks.append(("LLM API (EasyRouter)", resp.status_code == 200, f"{resp.status_code}"))
        except Exception as e:
            checks.append(("LLM API (EasyRouter)", False, str(e)[:60]))
    else:
        checks.append(("LLM API", False, "No API key in .env"))

    # 6. Schema
    try:
        async def check_schema():
            from surrealdb import AsyncSurreal
            db = AsyncSurreal("ws://localhost:8001/rpc")
            await db.connect()
            await db.signin({"user": "root", "pass": "root"})
            await db.use("agent_loop", "main")
            r = await db.query("INFO FOR TABLE fact")
            return r
        r = asyncio.run(asyncio.wait_for(check_schema(), timeout=5))
        checks.append(("DB Schema", bool(r), "initialized" if r else "not found"))
    except Exception as e:
        checks.append(("DB Schema", False, str(e)[:50]))

    # Print results
    all_ok = True
    for name, ok, detail in checks:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}: {detail}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("✅ 所有检查通过！系统健康。")
    else:
        print("⚠️  部分检查未通过，运行 'agent-loop setup' 修复。")

    return 0 if all_ok else 1


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

    # setup
    p_setup = sub.add_parser("setup", help="Interactive setup wizard for first-time configuration")
    p_setup.set_defaults(func=cmd_setup)

    # test
    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p_test.add_argument("--coverage", action="store_true", help="Run with coverage report")
    p_test.set_defaults(func=cmd_test)

    # doctor
    p_doctor = sub.add_parser("doctor", help="System health check")
    p_doctor.set_defaults(func=cmd_doctor)

    # init-config
    p_init = sub.add_parser("init-config", help="Initialize configuration file")
    p_init.add_argument("--config", default="agent-loop.yaml")
    p_init.set_defaults(func=cmd_init_config)

    # chat
    p_chat = sub.add_parser("chat", help="Run a single chat through Loop Engine")
    p_chat.add_argument("message", help="User input message")
    p_chat.add_argument("--memory-url", default="ws://localhost:8001/rpc",
                        help="SurrealDB WebSocket URL")
    p_chat.add_argument("--provider", default=None,
                        help="LLM provider (default: auto from llm_pool.json)")
    p_chat.add_argument("--api-key", default=None,
                        help="API key (default: from environment)")
    p_chat.set_defaults(func=cmd_chat)

    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--memory-url", default="ws://localhost:8001/rpc",
                         help="SurrealDB WebSocket URL")
    p_serve.add_argument("--provider", default=None,
                         help="LLM provider (default: auto from llm_pool.json)")
    p_serve.add_argument("--api-key", default=None,
                         help="API key (default: from environment)")
    p_serve.set_defaults(func=cmd_serve)

    # status
    p_status = sub.add_parser("status", help="Check system status")
    p_status.set_defaults(func=cmd_status)

    # start
    p_start = sub.add_parser("start", help="Start full Agent-Loop system")
    p_start.add_argument("--config", default="agent-loop.yaml")
    p_start.add_argument("--memory-url", default="ws://localhost:8001/rpc",
                        help="SurrealDB WebSocket URL")
    p_start.add_argument("--provider", default=None,
                        help="LLM provider (default: auto from llm_pool.json)")
    p_start.add_argument("--api-key", default=None)
    p_start.set_defaults(func=cmd_start)

    # stats
    p_stats = sub.add_parser("stats", help="Show memory pool stats")
    p_stats.add_argument("--memory-url", default="ws://localhost:8001/rpc",
                        help="SurrealDB WebSocket URL")
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
    p_anchor_sync.add_argument("--memory-url", default="ws://localhost:8001/rpc",
                              help="SurrealDB WebSocket URL")
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
    p_cleanup.add_argument("--memory-url", default="ws://localhost:8001/rpc",
                           help="SurrealDB WebSocket URL")
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
    p_cons.add_argument("--memory-url", default="ws://localhost:8001/rpc",
                        help="SurrealDB WebSocket URL")
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
