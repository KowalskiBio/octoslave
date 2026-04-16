"""CLI entrypoint for octoslave — interactive TUI + one-shot run mode."""

import os
import sys
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from . import display
from .agent import make_client, run_agent, continue_agent
from .research import run_long_research, ROLES as RESEARCH_ROLES
from .config import (
    KNOWN_MODELS, DEFAULT_MODEL, BASE_URL,
    load_config, save_config,
)

# ---------------------------------------------------------------------------
# Prompt-toolkit style
# ---------------------------------------------------------------------------

_PT_STYLE = Style.from_dict(
    {
        "prompt":       "bold #cc44ff",
        "model-tag":    "#888888",
        "input":        "#ffffff",
        "bottom-toolbar": "bg:#1a001a #666666",
    }
)

_HISTORY_FILE = Path.home() / ".octoslave" / "history"


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------

@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("-m", "--model", default=None, help="Model to use")
@click.option("-d", "--dir", "working_dir", default=None, help="Working directory")
@click.option("--api-key", default=None, envvar="OCTOSLAVE_API_KEY")
@click.option("--base-url", default=None, envvar="OCTOSLAVE_BASE_URL")
@click.pass_context
def cli(ctx, model, working_dir, api_key, base_url):
    """OctoSlave — autonomous AI research & coding assistant.

    Run without arguments to enter interactive mode.
    """
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["working_dir"] = working_dir
    ctx.obj["api_key"] = api_key
    ctx.obj["base_url"] = base_url

    if ctx.invoked_subcommand is None:
        # Default: interactive TUI
        _interactive(ctx.obj)


# ---------------------------------------------------------------------------
# `run` sub-command — one-shot task
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("task")
@click.option("-m", "--model", default=None)
@click.option("-d", "--dir", "working_dir", default=None)
@click.option("--api-key", default=None, envvar="OCTOSLAVE_API_KEY")
@click.option("--base-url", default=None, envvar="OCTOSLAVE_BASE_URL")
@click.option("-i", "--interactive", is_flag=True, help="Stay interactive after task")
def run(task, model, working_dir, api_key, base_url, interactive):
    """Run a single TASK and exit (or continue interactively with -i).

    \b
    Examples:
      ac run "build a REST API for a todo app"
      ac run "research recent papers on RAG" --model qwen3-coder
      ac run "add unit tests" -i
    """
    cfg = _resolve_config(model, working_dir, api_key, base_url)
    display.print_header(cfg["model"], cfg["working_dir"])
    display.print_task(task)

    client = make_client(cfg["api_key"], cfg["base_url"])
    messages = run_agent(task, cfg["model"], cfg["working_dir"], client)

    if interactive:
        _repl_loop(client, cfg, messages)


# ---------------------------------------------------------------------------
# `config` sub-command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--api-key", default=None)
@click.option("--model", default=None)
@click.option("--base-url", default=None)
@click.option("--show", is_flag=True, help="Show current config")
def config(api_key, model, base_url, show):
    """Configure API key, default model, and base URL."""
    current = load_config()

    if show:
        key = current.get("api_key", "")
        masked = (key[:8] + "…" + key[-4:]) if len(key) > 12 else ("set" if key else "not set")
        display.console.print(f"[bold]api_key[/bold]      : {masked}")
        display.console.print(f"[bold]base_url[/bold]     : {current.get('base_url')}")
        display.console.print(f"[bold]default_model[/bold]: {current.get('default_model')}")
        return

    new_key = api_key or current.get("api_key", "")
    new_url = base_url or current.get("base_url", BASE_URL)
    new_model = model or current.get("default_model", DEFAULT_MODEL)

    if not any([api_key, model, base_url]):
        display.console.print("[bold]OctoSlave — setup[/bold]\n")
        new_key = click.prompt("API key", default=new_key, hide_input=True, show_default=False)
        new_url = click.prompt("Base URL", default=new_url)
        new_model = click.prompt("Default model", default=new_model)

    save_config(new_key, new_url, new_model)
    display.console.print("[bold green]Config saved.[/bold green]")


# ---------------------------------------------------------------------------
# `models` sub-command
# ---------------------------------------------------------------------------

@cli.command()
def models():
    """List available models."""
    display.console.print("[bold]Available models on e-INFRA CZ:[/bold]\n")
    cfg = load_config()
    default = cfg.get("default_model", DEFAULT_MODEL)
    for m in KNOWN_MODELS:
        marker = " [bold green]← default[/bold green]" if m == default else ""
        display.console.print(f"  {m}{marker}")
    display.console.print()
    display.console.print("[dim]Switch with: /model <name>  or  -m <name>[/dim]")


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def _interactive(ctx_obj: dict):
    cfg = _resolve_config(
        ctx_obj.get("model"),
        ctx_obj.get("working_dir"),
        ctx_obj.get("api_key"),
        ctx_obj.get("base_url"),
    )
    if not cfg["api_key"]:
        display.print_error(
            "No API key configured. Run `ots config` or set OCTOSLAVE_API_KEY."
        )
        sys.exit(1)

    display.print_welcome(cfg["model"], cfg["working_dir"])
    client = make_client(cfg["api_key"], cfg["base_url"])
    messages: list[dict] = []

    _repl_loop(client, cfg, messages)


def _repl_loop(client, cfg: dict, messages: list[dict]):
    """The main REPL: read input, handle slash commands, run agent."""
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)),
        style=_PT_STYLE,
        key_bindings=_make_keybindings(),
    )

    # Mutable state (passed by ref via dict)
    state = {
        "model": cfg["model"],
        "working_dir": cfg["working_dir"],
    }

    while True:
        try:
            user_input = session.prompt(
                _make_prompt(state),
                bottom_toolbar=_make_toolbar(state),
            ).strip()
        except KeyboardInterrupt:
            display.console.print("[dim]\n(Ctrl+C — use /exit or Ctrl+D to quit)[/dim]")
            messages = []        # clear pending state on interrupt
            continue
        except EOFError:
            display.console.print("[dim]\nBye.[/dim]")
            break

        if not user_input:
            continue

        # --- slash commands ---
        if user_input.startswith("/"):
            handled = _handle_slash(user_input, state, cfg, messages, client)
            if handled == "exit":
                break
            if handled == "clear":
                messages = []
            continue

        # --- normal task ---
        display.print_task(user_input)
        try:
            if messages:
                messages = continue_agent(messages, user_input, state["model"], state["working_dir"], client)
            else:
                messages = run_agent(user_input, state["model"], state["working_dir"], client)
        except KeyboardInterrupt:
            display.console.print("\n[dim]Interrupted.[/dim]")
            messages = []


def _handle_slash(cmd: str, state: dict, cfg: dict, messages: list, client) -> str | None:
    parts = cmd.split(None, 1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name in ("/exit", "/quit", "/q"):
        display.console.print("[dim]Bye.[/dim]")
        return "exit"

    if name in ("/help", "/?"):
        display.print_help()
        return "ok"

    if name == "/clear":
        display.console.clear()
        display.print_welcome(state["model"], state["working_dir"])
        return "clear"

    if name == "/model":
        if not arg:
            display.console.print("[bold]Available models:[/bold]")
            for m in KNOWN_MODELS:
                mark = " [green]←[/green]" if m == state["model"] else ""
                display.console.print(f"  {m}{mark}")
        else:
            state["model"] = arg
            display.console.print(f"[dim]Model set to[/dim] [bold magenta]{arg}[/bold magenta]")
            messages.clear()
        return "ok"

    if name == "/dir":
        if not arg:
            display.console.print(f"[dim]Working dir:[/dim] {state['working_dir']}")
        else:
            new_dir = str(Path(arg).expanduser().resolve())
            if not Path(new_dir).is_dir():
                display.print_error(f"Not a directory: {arg}")
            else:
                state["working_dir"] = new_dir
                display.console.print(f"[dim]Dir set to[/dim] {new_dir}")
                messages.clear()
        return "ok"

    if name == "/compact":
        if not messages:
            display.print_info("No conversation to compact.")
            return "ok"
        summary_task = (
            "Summarise this conversation so far into a compact context block that preserves "
            "all key findings, code written, hypotheses, and decisions. Keep it under 400 words."
        )
        try:
            new_msgs = continue_agent(messages, summary_task, state["model"], state["working_dir"], client)
            messages.clear()
            messages.extend(new_msgs[-2:])
            display.print_info("History compacted.")
        except Exception as e:
            display.print_error(str(e))
        return "ok"

    if name == "/long-research":
        _handle_long_research(arg, state, cfg, client)
        return "ok"

    display.print_error(f"Unknown command: {name}  (type /help)")
    return "ok"


def _handle_long_research(arg: str, state: dict, cfg: dict, client):
    """Parse /long-research flags and launch the research pipeline."""
    import shlex

    # Parse: /long-research <topic words> [--rounds N] [--all MODEL] [--overseer MODEL] [--resume]
    try:
        tokens = shlex.split(arg)
    except ValueError:
        tokens = arg.split()

    topic_parts: list[str] = []
    max_rounds = 5
    all_model: str | None = None
    overseer_model: str | None = None
    resume = False

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--rounds" and i + 1 < len(tokens):
            try:
                max_rounds = int(tokens[i + 1])
            except ValueError:
                display.print_error(f"--rounds expects an integer, got: {tokens[i+1]}")
                return
            i += 2
        elif t == "--all" and i + 1 < len(tokens):
            all_model = tokens[i + 1]
            i += 2
        elif t == "--overseer" and i + 1 < len(tokens):
            overseer_model = tokens[i + 1]
            i += 2
        elif t == "--resume":
            resume = True
            i += 1
        else:
            topic_parts.append(t)
            i += 1

    topic = " ".join(topic_parts).strip()
    if not topic:
        display.print_error(
            "Usage: /long-research <topic> [--rounds N] [--all MODEL] "
            "[--overseer MODEL] [--resume]"
        )
        return

    # Build per-role model overrides
    overrides: dict[str, str] = {}
    if all_model:
        for role in RESEARCH_ROLES:
            overrides[role] = all_model
    if overseer_model:
        overrides["orchestrator"] = overseer_model

    run_long_research(
        topic=topic,
        working_dir=state["working_dir"],
        client=client,
        max_rounds=max_rounds,
        model_overrides=overrides,
        resume=resume,
    )


# ---------------------------------------------------------------------------
# Prompt-toolkit helpers
# ---------------------------------------------------------------------------

def _make_prompt(state: dict):
    model_short = state["model"][:20]
    return HTML(f'<prompt>◆</prompt> <model-tag>[{model_short}]</model-tag> ')


def _make_toolbar(state: dict):
    wd = state["working_dir"]
    if len(wd) > 50:
        wd = "…" + wd[-47:]
    return HTML(f'<bottom-toolbar>  dir: {wd}   /help · /model · /clear · /exit</bottom-toolbar>')


def _make_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-l")
    def _clear_screen(event):
        event.app.renderer.clear()

    return kb


# ---------------------------------------------------------------------------
# Config resolution helper
# ---------------------------------------------------------------------------

def _resolve_config(model, working_dir, api_key, base_url) -> dict:
    saved = load_config()
    return {
        "api_key":     api_key or saved.get("api_key", ""),
        "base_url":    base_url or saved.get("base_url", BASE_URL),
        "model":       model or saved.get("default_model", DEFAULT_MODEL),
        "working_dir": str(Path(working_dir).resolve()) if working_dir else os.getcwd(),
    }


def main():
    cli()


if __name__ == "__main__":
    main()
