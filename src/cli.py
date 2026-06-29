"""CLI entry point for Agent-Loop."""

import asyncio
import logging
import sys

import click

from .loop_engine import LoopConfig
from .memory import MemoryPool

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """Agent-Loop: Loop Engine AI Agent System."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


@main.command()
@click.option("--url", default="ws://127.0.0.1:8000", help="SurrealDB URL")
@click.option("--namespace", default="agent_loop", help="SurrealDB namespace")
@click.option("--database", default="memory", help="SurrealDB database")
@click.option("--user", default="root", help="SurrealDB user")
@click.option("--password", default="root", help="SurrealDB password")
def init(url, namespace, database, user, password):
    """Initialize Agent-Loop: set up SurrealDB schema."""
    async def _init():
        pool = MemoryPool(url=url, namespace=namespace, database=database,
                          user=user, password=password)
        await pool.connect()
        await pool.initialize_schema()
        click.echo("✅ Memory Pool schema initialized.")
        await pool.disconnect()

    asyncio.run(_init())


@main.command()
@click.option("--url", default="ws://127.0.0.1:8000", help="SurrealDB URL")
@click.option("--llm-provider", default="deepseek", help="LLM provider name")
@click.option("--llm-api-key", envvar="LLM_API_KEY", help="LLM API key")
@click.argument("input_text", nargs=-1)
def run(url, llm_provider, llm_api_key, input_text):
    """Run a single MainLoop cycle."""
    if not input_text:
        click.echo("Error: Please provide input text.", err=True)
        sys.exit(1)

    user_input = " ".join(input_text)

    async def _run():
        pool = MemoryPool(url=url)
        await pool.connect()

        # TODO: Wire up LLM provider based on --llm-provider
        from .loop_engine.main_loop import MainLoop
        # Placeholder LLM for now
        engine = MainLoop(memory=pool, llm=None)

        result = await engine.run(user_input)
        click.echo(result)

        await pool.disconnect()

    asyncio.run(_run())


@main.command()
def serve():
    """Start Agent-Loop as a server (API + WebSocket)."""
    click.echo("Server mode not yet implemented.")


@main.command()
def worker():
    """Start an Agent worker process."""
    click.echo("Worker mode not yet implemented.")


if __name__ == "__main__":
    main()
