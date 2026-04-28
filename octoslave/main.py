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
from .vault import run_vault_improve
from .config import (
    KNOWN_MODELS, DEFAULT_MODEL, BASE_URL, OLLAMA_BASE_URL,
    NIM_BASE_URL, NIM_DEFAULT_MODEL, NIM_KNOWN_MODELS,
    PIPELINE_ROLES, EINFRA_ROLE_MODELS, NIM_ROLE_MODELS,
    load_config, save_config,
    ollama_is_running, ollama_list_models, ollama_pull_model,
    nim_list_models, einfra_list_models, list_models,
    assign_local_models,
    get_role_models, save_role_model, reset_role_models,
)

# ---------------------------------------------------------------------------
# Prompt-toolkit style
# ---------------------------------------------------------------------------

_PT_STYLE = Style.from_dict(
    {
        "prompt":         "bold #cc44ff",
        "prompt-local":   "bold #44ffaa",   # green tint in local mode
        "model-tag":      "#888888",
        "input":          "#ffffff",
        "bottom-toolbar": "bg:#1a001a #666666",
        "bottom-toolbar-local": "bg:#001a0a #666666",
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
@click.option("--local", is_flag=True, default=False, help="Use local Ollama models")
@click.option("-p", "--prompt-profile", default="base", help="Prompt profile to use (default: base, options: base, coder, analyst, biomedic)")
@click.option("--permission-mode", default=None,
              type=click.Choice(["autonomous", "controlled", "supervised"]),
              help="Permission mode: autonomous (default), controlled (ask before all edits), or supervised (ask before file edits only)")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose mode: show full diffs, complete tool output, and bash commands live")
@click.pass_context
def cli(ctx, model, working_dir, api_key, base_url, local, prompt_profile, permission_mode, verbose):
    """OctoSlave — autonomous AI research & coding assistant.

    Run without arguments to enter interactive mode.
    """
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["working_dir"] = working_dir
    ctx.obj["api_key"] = api_key
    ctx.obj["base_url"] = base_url
    ctx.obj["local"] = local
    ctx.obj["prompt_profile"] = prompt_profile
    ctx.obj["permission_mode"] = permission_mode
    ctx.obj["verbose"] = verbose
    if verbose:
        display.set_verbose(True)

    if ctx.invoked_subcommand is None:
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
@click.option("--local", is_flag=True, default=False, help="Use local Ollama models")
@click.option("-p", "--prompt-profile", default="base", help="Prompt profile to use (default: base, options: base, coder, analyst, biomedic)")
@click.option("-i", "--interactive", is_flag=True, help="Stay interactive after task")
@click.option("--permission-mode", default=None,
              type=click.Choice(["autonomous", "controlled", "supervised"]),
              help="Permission mode: autonomous (default), controlled (ask before all edits), or supervised (ask before file edits only)")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose mode: show full diffs, complete tool output, and bash commands live")
def run(task, model, working_dir, api_key, base_url, local, prompt_profile, interactive, permission_mode, verbose):
    """Run a single TASK and exit (or continue interactively with -i).

    \b
    Examples:
      ots run "build a REST API for a todo app"
      ots run "research recent papers on RAG" --model qwen3-coder
      ots run "add unit tests" -i
      ots run "explain this codebase" --local
      ots run "build a REST API" -p coder    # pure coding mode
      ots run "analyze this dataset" -p analyst  # data analysis mode
      ots run "edit files" --permission-mode controlled  # ask before each edit
      ots run "reorganize notes" -v           # see every edit live
    """
    if verbose:
        display.set_verbose(True)
    cfg = _resolve_config(model, working_dir, api_key, base_url, local=local)
    cfg["prompt_profile"] = prompt_profile

    # Auto-create project dir if no explicit dir given
    if not working_dir:
        cfg["working_dir"] = _make_project_dir(task)

    # Override permission mode if specified
    if permission_mode:
        cfg["permission_mode"] = permission_mode
    else:
        saved_cfg = load_config()
        cfg["permission_mode"] = saved_cfg.get("permission_mode", "autonomous")
    
    display.print_header(cfg["model"], cfg["working_dir"], backend=cfg["backend"])
    
    # Show permission mode in header
    if cfg["permission_mode"] == "autonomous":
        mode_tag = "[bold green]autonomous[/bold green]"
    elif cfg["permission_mode"] == "controlled":
        mode_tag = "[bold yellow]controlled[/bold yellow]"
    else:
        mode_tag = "[bold cyan]supervised[/bold cyan]"
    display.console.print(f"[dim]permission mode: {mode_tag}[/dim]")
    display.console.print()
    
    display.print_task(task)

    client = make_client(cfg["api_key"], cfg["base_url"])
    messages = run_agent(
        task, cfg["model"], cfg["working_dir"], client, 
        prompt_profile, cfg["permission_mode"]
    )

    if interactive:
        _repl_loop(client, cfg, messages)


# ---------------------------------------------------------------------------
# `config` sub-command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--api-key", default=None)
@click.option("--nim-api-key", default=None, help="NVIDIA NIM API key")
@click.option("--model", default=None)
@click.option("--base-url", default=None)
@click.option("--ollama-url", default=None, help="Ollama base URL (default: http://localhost:11434/v1)")
@click.option("--nim-url", default=None, help="NVIDIA NIM base URL (default: https://integrate.api.nvidia.com/v1)")
@click.option("--permission-mode", default=None,
              type=click.Choice(["autonomous", "controlled", "supervised"]),
              help="Permission mode: autonomous (default), controlled (ask before all edits), or supervised (ask before file edits only)")
@click.option("--show", is_flag=True, help="Show current config")
def config(api_key, nim_api_key, model, base_url, ollama_url, nim_url, permission_mode, show):
    """Configure API key, default model, base URL, Ollama/NIM settings, and permission mode."""
    current = load_config()

    if show:
        key = current.get("api_key", "")
        masked = (key[:8] + "…" + key[-4:]) if len(key) > 12 else ("set" if key else "not set")
        nim_key = current.get("nim_api_key", "")
        nim_masked = (nim_key[:8] + "…" + nim_key[-4:]) if len(nim_key) > 12 else ("set" if nim_key else "not set")
        backend = current.get("backend", "einfra")
        perm_mode = current.get("permission_mode", "autonomous")
        display.console.print(f"[bold]backend[/bold]        : {backend}")
        display.console.print(f"[bold]api_key[/bold]        : {masked}")
        display.console.print(f"[bold]base_url[/bold]       : {current.get('base_url')}")
        display.console.print(f"[bold]default_model[/bold]  : {current.get('default_model')}")
        display.console.print(f"[bold]ollama_url[/bold]     : {current.get('ollama_url', OLLAMA_BASE_URL)}")
        display.console.print(f"[bold]nim_api_key[/bold]    : {nim_masked}")
        display.console.print(f"[bold]nim_url[/bold]        : {current.get('nim_url', NIM_BASE_URL)}")
        display.console.print(f"[bold]permission_mode[/bold]: {perm_mode}")
        if backend == "ollama":
            running = ollama_is_running(current.get("ollama_url", OLLAMA_BASE_URL))
            pulled = ollama_list_models(current.get("ollama_url", OLLAMA_BASE_URL))
            status = "[bold green]running[/bold green]" if running else "[bold red]not running[/bold red]"
            display.console.print(f"[bold]ollama status[/bold] : {status}")
            if pulled:
                display.console.print("[bold]pulled models[/bold] :")
                for m in pulled:
                    display.console.print(f"  {m}")
        return

    new_key = api_key or current.get("api_key", "")
    new_nim_key = nim_api_key or current.get("nim_api_key", "")
    new_url = base_url or current.get("base_url", BASE_URL)
    new_model = model or current.get("default_model", DEFAULT_MODEL)
    new_ollama = ollama_url or current.get("ollama_url", OLLAMA_BASE_URL)
    new_nim_url = nim_url or current.get("nim_url", NIM_BASE_URL)
    new_backend = current.get("backend", "einfra")
    new_perm_mode = permission_mode or current.get("permission_mode", "autonomous")

    if not any([api_key, nim_api_key, model, base_url, ollama_url, nim_url, permission_mode]):
        display.console.print("[bold]OctoSlave — setup[/bold]\n")
        display.console.print(
            "  [bold]einfra[/bold]  — e-INFRA CZ cloud API  "
            "(requires an API key; best model quality; recommended)\n"
            "  [bold]ollama[/bold]  — local models via Ollama "
            "(no API key; fully private; GPU strongly recommended)\n"
            "  [bold]nim[/bold]     — NVIDIA NIM cloud API    "
            "(requires an API key; access to NVIDIA-optimised models)\n"
        )
        new_backend = click.prompt(
            "Backend",
            default=new_backend,
            type=click.Choice(["einfra", "ollama", "nim"]),
        )

        if new_backend == "einfra":
            display.console.print(
                "\n  Get an API key at [link=https://llm.ai.e-infra.cz]llm.ai.e-infra.cz[/link] "
                "(free for Czech academic institutions).\n"
            )
            new_key = click.prompt(
                "API key (e-INFRA CZ)",
                default=new_key,
                hide_input=True,
                show_default=False,
            )
            new_url = click.prompt("Base URL (leave default unless self-hosting)", default=new_url)
            display.console.print(
                "\n  Suggested models:\n"
                "    [bold]deepseek-v3.2[/bold]          — best all-round default (reasoning + coding)\n"
                "    [bold]deepseek-v3.2-thinking[/bold] — extended chain-of-thought; slower\n"
                "    [bold]qwen3-coder-30b[/bold]        — strongest at code generation\n"
                "    [bold]qwen3.5-122b[/bold]           — fast reader; good for research\n"
                "    [bold]gpt-oss-120b[/bold]           — large context; clean writing\n"
                "  Run [bold]ots models[/bold] to see the full list.\n"
            )
            new_model = click.prompt("Default model", default=new_model)
        elif new_backend == "nim":
            display.console.print(
                "\n  Get an API key at [link=https://build.nvidia.com]build.nvidia.com[/link].\n"
            )
            new_nim_key = click.prompt(
                "API key (NVIDIA NIM)",
                default=new_nim_key,
                hide_input=True,
                show_default=False,
            )
            new_nim_url = click.prompt("NIM Base URL (leave default unless self-hosting)", default=new_nim_url)
            display.console.print(
                "\n  Suggested models:\n"
                "    [bold]meta/llama-3.3-70b-instruct[/bold]         — fast, strong all-round default\n"
                "    [bold]meta/llama-3.1-405b-instruct[/bold]        — largest Llama; best quality\n"
                "    [bold]nvidia/llama-3.1-nemotron-70b-instruct[/bold] — NVIDIA-tuned reasoning\n"
                "    [bold]deepseek-ai/deepseek-r1[/bold]             — extended chain-of-thought\n"
                "  Run [bold]ots models[/bold] to see the full list.\n"
            )
            new_model = click.prompt("Default model", default=current.get("default_model", NIM_DEFAULT_MODEL))
        else:
            new_ollama = click.prompt("Ollama URL", default=new_ollama)
            running = ollama_is_running(new_ollama)
            if not running:
                display.console.print(
                    "[yellow]  Ollama is not running — start it with: ollama serve[/yellow]\n"
                    "  Pull a model later with: ollama pull llama3.1:8b\n"
                )
                new_model = click.prompt("Default model (set now or update after pulling)", default=new_model)
            else:
                pulled = ollama_list_models(new_ollama)
                if pulled:
                    display.console.print(
                        "\n  Pulled models: " + ", ".join(pulled) + "\n"
                        "  Tip: pull a strong reasoning model for Tier A (orchestrator/evaluator)\n"
                        "       and a coder model for Tier B (coder/debugger).\n"
                    )
                    new_model = click.prompt("Default model", default=pulled[0], type=click.Choice(pulled))
                else:
                    display.console.print(
                        "\n  No models pulled yet. Recommended first pull:\n"
                        "    ollama pull llama3.1:8b   (5 GB — good all-round)\n"
                    )
                    new_model = click.prompt("Default model (set after pulling)", default="llama3.1:8b")

        # Ask about permission mode if not explicitly set
        display.console.print(
            "\n  [bold]Permission mode:[/bold]\n"
            "  [bold]autonomous[/bold]  — work without asking (default)\n"
            "  [bold]controlled[/bold]  — ask before file edits or commands\n"
            "  [bold]supervised[/bold]  — ask before file edits, auto-allow commands\n"
        )
        new_perm_mode = click.prompt(
            "Permission mode",
            default=new_perm_mode,
            type=click.Choice(["autonomous", "controlled", "supervised"]),
        )

    save_config(
        new_key, new_url, new_model,
        backend=new_backend,
        ollama_url=new_ollama,
        permission_mode=new_perm_mode,
        nim_api_key=new_nim_key,
        nim_url=new_nim_url,
    )
    display.console.print("[bold green]Config saved.[/bold green]")


# ---------------------------------------------------------------------------
# `models` sub-command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--local", is_flag=True, default=False, help="List local Ollama models instead")
def models(local):
    """List available models."""
    cfg = load_config()

    if local or cfg.get("backend") == "ollama":
        _print_local_models(cfg.get("ollama_url", OLLAMA_BASE_URL))
        return

    if cfg.get("backend") == "nim":
        nim_models = nim_list_models(cfg.get("nim_url", NIM_BASE_URL), cfg.get("nim_api_key", ""))
        if nim_models:
            display.console.print("[bold]Available models on NVIDIA NIM[/bold] [dim](live from API)[/dim]\n")
        else:
            nim_models = list(NIM_KNOWN_MODELS)
            display.console.print("[bold]Available models on NVIDIA NIM[/bold] [dim](static fallback)[/dim]\n")
        default = cfg.get("default_model", NIM_DEFAULT_MODEL)
        for m in nim_models:
            marker = " [bold green]← default[/bold green]" if m == default else ""
            display.console.print(f"  {m}{marker}")
        display.console.print()
        display.console.print("[dim]Switch with: /model <name>  or  -m <name>[/dim]")
        display.console.print("[dim]Switch backend: /einfra · /local · /nim[/dim]")
        return

    einfra_models = einfra_list_models(cfg.get("base_url", BASE_URL), cfg.get("api_key", ""))
    if einfra_models:
        display.console.print("[bold]Available models on e-INFRA CZ[/bold] [dim](live from API)[/dim]\n")
    else:
        einfra_models = list(KNOWN_MODELS)
        display.console.print("[bold]Available models on e-INFRA CZ[/bold] [dim](static fallback)[/dim]\n")
    default = cfg.get("default_model", DEFAULT_MODEL)
    for m in einfra_models:
        marker = " [bold green]← default[/bold green]" if m == default else ""
        display.console.print(f"  {m}{marker}")
    display.console.print()
    display.console.print("[dim]Switch with: /model <name>  or  -m <name>[/dim]")
    display.console.print("[dim]Use local Ollama models: /local  or  --local flag[/dim]")
    display.console.print("[dim]Use NVIDIA NIM: /nim[/dim]")


def _print_local_models(ollama_url: str):
    if not ollama_is_running(ollama_url):
        display.print_error(
            "Ollama is not running. Start it with: ollama serve"
        )
        return
    pulled = ollama_list_models(ollama_url)
    if not pulled:
        display.console.print("[dim]No models pulled yet.[/dim]")
        display.console.print("Pull a model with: [cyan]ollama pull mistral[/cyan]")
        return
    display.console.print("[bold]Pulled Ollama models:[/bold]\n")
    for m in pulled:
        display.console.print(f"  [bold bright_green]{m}[/bold bright_green]")
    display.console.print()
    display.console.print("[dim]Switch with: /model <name>[/dim]")
    display.console.print("[dim]Pull more with: /pull <model-name>[/dim]")


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

def _interactive(ctx_obj: dict):
    cfg = _resolve_config(
        ctx_obj.get("model"),
        ctx_obj.get("working_dir"),
        ctx_obj.get("api_key"),
        ctx_obj.get("base_url"),
        local=ctx_obj.get("local", False),
    )
    cfg["prompt_profile"] = ctx_obj.get("prompt_profile", "base")
    cfg["verbose"] = ctx_obj.get("verbose", False)
    cfg["explicit_dir"] = bool(ctx_obj.get("working_dir"))

    # Handle permission mode from CLI or config
    if ctx_obj.get("permission_mode"):
        cfg["permission_mode"] = ctx_obj["permission_mode"]
    else:
        saved_cfg = load_config()
        cfg["permission_mode"] = saved_cfg.get("permission_mode", "autonomous")

    is_local = cfg["backend"] == "ollama"

    if not is_local and not cfg["api_key"]:
        display.print_error(
            "No API key configured. Run `ots config` or set OCTOSLAVE_API_KEY.\n"
            "For local models: `ots --local` or `/local` in session.\n"
            "For NVIDIA NIM: run `ots config` and choose the nim backend."
        )
        sys.exit(1)

    display.print_welcome(cfg["model"], cfg["working_dir"], backend=cfg["backend"])
    
    # Show permission mode
    if cfg["permission_mode"] == "autonomous":
        mode_tag = "[bold green]autonomous[/bold green]"
    elif cfg["permission_mode"] == "controlled":
        mode_tag = "[bold yellow]controlled[/bold yellow]"
    else:
        mode_tag = "[bold cyan]supervised[/bold cyan]"
    display.console.print(f"[dim]permission mode: {mode_tag}[/dim]")
    display.console.print()
    
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

    state = {
        "model":        cfg["model"],
        "working_dir":  cfg["working_dir"],
        "backend":      cfg["backend"],
        "ollama_url":   cfg.get("ollama_url", OLLAMA_BASE_URL),
        "api_key":      cfg.get("api_key", ""),
        "base_url":     cfg.get("base_url", BASE_URL),
        "nim_api_key":  cfg.get("nim_api_key", ""),
        "nim_url":      cfg.get("nim_url", NIM_BASE_URL),
        "prompt_profile":  cfg.get("prompt_profile", "base"),
        "permission_mode": cfg.get("permission_mode", "autonomous"),
        "verbose": cfg.get("verbose", False),
    }
    if state["verbose"]:
        display.set_verbose(True)

    while True:
        try:
            user_input = session.prompt(
                _make_prompt(state),
                bottom_toolbar=_make_toolbar(state),
            ).strip()
        except KeyboardInterrupt:
            display.console.print("[dim]\n(Ctrl+C — use /exit or Ctrl+D to quit)[/dim]")
            messages = []
            continue
        except EOFError:
            display.console.print("[dim]\nBye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handled = _handle_slash(user_input, state, cfg, messages, client)
            if handled == "exit":
                break
            if handled == "clear":
                messages = []
            if handled == "new_client":
                # Backend switched — rebuild client and clear history
                client = make_client(state["api_key"], state["base_url"])
                messages = []
            continue

        # Auto-create project dir on first task if no explicit dir was set
        if not messages and not cfg.get("explicit_dir"):
            project_dir = _make_project_dir(user_input)
            if project_dir != state["working_dir"]:
                state["working_dir"] = project_dir
                display.console.print(
                    f"[dim]📁 project dir:[/dim] [bold]{project_dir}[/bold]"
                )

        display.print_task(user_input)
        try:
            if messages:
                messages = continue_agent(
                    messages, user_input, state["model"],
                    state["working_dir"], client,
                    state["permission_mode"]
                )
            else:
                messages = run_agent(
                    user_input, state["model"],
                    state["working_dir"], client,
                    state["prompt_profile"],
                    state["permission_mode"]
                )
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
        display.print_welcome(state["model"], state["working_dir"],
                               backend=state["backend"])
        return "clear"

    if name == "/verbose":
        new_state = not display.is_verbose()
        display.set_verbose(new_state)
        state["verbose"] = new_state
        status = "[bold green]ON[/bold green]" if new_state else "[bold red]OFF[/bold red]"
        display.console.print(f"[dim]Verbose mode:[/dim] {status}")
        return "ok"

    if name == "/model":
        if not arg:
            if state["backend"] == "ollama":
                _print_local_models(state["ollama_url"])
            elif state["backend"] == "nim":
                nim_models = nim_list_models(state.get("nim_url", NIM_BASE_URL), state.get("nim_api_key", ""))
                source = "[dim](live)[/dim]" if nim_models else "[dim](fallback)[/dim]"
                if not nim_models:
                    nim_models = list(NIM_KNOWN_MODELS)
                display.console.print(f"[bold]Available models on NVIDIA NIM:[/bold] {source}")
                for m in nim_models:
                    mark = " [green]←[/green]" if m == state["model"] else ""
                    display.console.print(f"  {m}{mark}")
            else:
                live = einfra_list_models(state.get("base_url", BASE_URL), state.get("api_key", ""))
                source = "[dim](live)[/dim]" if live else "[dim](fallback)[/dim]"
                models_to_show = live if live else list(KNOWN_MODELS)
                display.console.print(f"[bold]Available models on e-INFRA CZ:[/bold] {source}")
                for m in models_to_show:
                    mark = " [green]←[/green]" if m == state["model"] else ""
                    display.console.print(f"  {m}{mark}")
        else:
            state["model"] = arg
            display.console.print(
                f"[dim]Model set to[/dim] [bold magenta]{arg}[/bold magenta]"
            )
            messages.clear()
        return "ok"

    if name == "/local":
        return _handle_local_switch(arg, state, messages)

    if name == "/einfra":
        return _handle_einfra_switch(state, messages)

    if name == "/nim":
        return _handle_nim_switch(arg, state, messages)

    if name == "/pull":
        if not arg:
            display.print_error("Usage: /pull <model-name>  e.g. /pull llama3.2")
            return "ok"
        _do_pull(arg, state)
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

    if name == "/profile":
        from .agent import load_system_prompt
        if not arg:
            current = state.get("prompt_profile", "base")
            available = ["base", "coder", "analyst"]
            display.console.print(f"[dim]Current profile:[/dim] [bold]{current}[/bold]")
            display.console.print(f"[dim]Available profiles:[/dim] {', '.join(available)}")
            display.console.print("[dim]Usage: /profile <name>  e.g. /profile coder[/dim]")
        else:
            # Validate profile exists
            try:
                test_prompt = load_system_prompt(arg, state["working_dir"])
                state["prompt_profile"] = arg
                display.console.print(
                    f"[dim]Prompt profile set to[/dim] [bold magenta]{arg}[/bold magenta]"
                )
                display.console.print(
                    "[dim]Note: Profile will be used for the next task (new conversation).[/dim]"
                )
                messages.clear()
            except FileNotFoundError as e:
                display.print_error(str(e))
        return "ok"

    if name == "/permission":
        if not arg:
            current = state.get("permission_mode", "autonomous")
            available = ["autonomous", "controlled", "supervised"]
            display.console.print(f"[dim]Current permission mode:[/dim] [bold]{current}[/bold]")
            display.console.print(f"[dim]Available modes:[/dim] {', '.join(available)}")
            display.console.print(
                "[dim]Usage: /permission <mode>  e.g. /permission controlled[/dim]\n"
                "[dim]  autonomous — work without asking (default)[/dim]\n"
                "[dim]  controlled — ask before file edits or commands[/dim]\n"
                "[dim]  supervised — ask before file edits, auto-allow commands[/dim]"
            )
        else:
            arg = arg.lower()
            if arg not in ("autonomous", "controlled", "supervised"):
                display.print_error(
                    f"Invalid mode '{arg}'. Use 'autonomous', 'controlled', or 'supervised'."
                )
                return "ok"
            state["permission_mode"] = arg
            if arg == "autonomous":
                mode_tag = "[bold green]autonomous[/bold green]"
            elif arg == "controlled":
                mode_tag = "[bold yellow]controlled[/bold yellow]"
            else:
                mode_tag = "[bold cyan]supervised[/bold cyan]"
            display.console.print(
                f"[dim]Permission mode set to[/dim] {mode_tag}"
            )
            display.console.print(
                "[dim]Note: Mode will apply to the next tool execution.[/dim]"
            )
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
            new_msgs = continue_agent(messages, summary_task, state["model"],
                                       state["working_dir"], client)
            # Keep: system prompt (index 0) + the assistant's summary reply (last
            # assistant message). This guarantees the system prompt is always present.
            system_msg = next((m for m in new_msgs if m.get("role") == "system"), None)
            summary_msg = next(
                (m for m in reversed(new_msgs) if m.get("role") == "assistant"), None
            )
            messages.clear()
            if system_msg:
                messages.append(system_msg)
            if summary_msg:
                messages.append(summary_msg)
            display.print_info("History compacted.")
        except Exception as e:
            display.print_error(str(e))
        return "ok"

    if name == "/long-research":
        _handle_long_research(arg, state, cfg, client)
        return "ok"

    if name == "/research-roles":
        _handle_research_roles(arg, state, cfg)
        return "ok"

    if name == "/vault-improve":
        _handle_vault_improve(arg, state, client)
        return "ok"

    display.print_error(f"Unknown command: {name}  (type /help)")
    return "ok"


def _handle_local_switch(arg: str, state: dict, messages: list) -> str:
    """Switch to local Ollama backend. Optionally pass model name as arg."""
    ollama_url = state.get("ollama_url", OLLAMA_BASE_URL)

    if not ollama_is_running(ollama_url):
        display.print_error(
            "Ollama is not running.\n"
            "Start it with:  [bold]ollama serve[/bold]\n"
            "Then try /local again."
        )
        return "ok"

    pulled = ollama_list_models(ollama_url)
    if not pulled:
        display.print_error(
            "No models are pulled yet.\n"
            "Pull one with:  [bold]/pull mistral[/bold]  or  [bold]ollama pull mistral[/bold]"
        )
        return "ok"

    chosen = arg if arg else pulled[0]
    if chosen not in pulled:
        display.print_error(
            f"Model '{chosen}' is not pulled. Available: {', '.join(pulled)}"
        )
        return "ok"

    state["backend"] = "ollama"
    state["model"] = chosen
    state["api_key"] = "ollama"
    state["base_url"] = ollama_url

    # Persist backend switch
    saved = load_config()
    save_config(
        saved.get("api_key", ""),
        saved.get("base_url", BASE_URL),
        chosen,
        backend="ollama",
        ollama_url=ollama_url,
    )

    display.console.print(
        f"[bold bright_green]● Local mode[/bold bright_green] — using [bold]{chosen}[/bold] via Ollama"
    )
    display.console.print(
        f"[dim]  {len(pulled)} model(s) available: {', '.join(pulled)}[/dim]"
    )
    display.console.print("[dim]  Switch back: /einfra[/dim]")
    messages.clear()
    return "new_client"


def _handle_einfra_switch(state: dict, messages: list) -> str:
    """Switch back to e-INFRA CZ backend."""
    saved = load_config()
    api_key = saved.get("api_key", "")
    if not api_key:
        display.print_error(
            "No e-INFRA CZ API key configured. Run `ots config` first."
        )
        return "ok"

    state["backend"] = "einfra"
    state["model"] = DEFAULT_MODEL
    state["api_key"] = api_key
    state["base_url"] = saved.get("base_url", BASE_URL)

    save_config(
        api_key,
        state["base_url"],
        DEFAULT_MODEL,
        backend="einfra",
        ollama_url=state.get("ollama_url", OLLAMA_BASE_URL),
    )

    display.console.print(
        f"[bold bright_magenta]● e-INFRA CZ mode[/bold bright_magenta] — using [bold]{state['model']}[/bold]"
    )
    messages.clear()
    return "new_client"


def _handle_nim_switch(arg: str, state: dict, messages: list) -> str:
    """Switch to NVIDIA NIM backend. Optionally pass model name as arg."""
    saved = load_config()
    nim_api_key = saved.get("nim_api_key", "")
    if not nim_api_key:
        display.print_error(
            "No NVIDIA NIM API key configured.\n"
            "Run [bold]ots config[/bold] and choose the nim backend, "
            "or set OCTOSLAVE_NIM_API_KEY."
        )
        return "ok"

    nim_url = saved.get("nim_url", NIM_BASE_URL)
    chosen = arg if arg else NIM_DEFAULT_MODEL

    state["backend"] = "nim"
    state["model"] = chosen
    state["api_key"] = nim_api_key
    state["base_url"] = nim_url
    state["nim_api_key"] = nim_api_key
    state["nim_url"] = nim_url

    save_config(
        saved.get("api_key", ""),
        saved.get("base_url", BASE_URL),
        chosen,
        backend="nim",
        ollama_url=saved.get("ollama_url", OLLAMA_BASE_URL),
        nim_api_key=nim_api_key,
        nim_url=nim_url,
    )

    display.console.print(
        f"[bold bright_cyan]● NVIDIA NIM mode[/bold bright_cyan] — using [bold]{chosen}[/bold]"
    )
    display.console.print(
        "[dim]  Switch back: /einfra  or  /local[/dim]"
    )
    messages.clear()
    return "new_client"


def _do_pull(model_name: str, state: dict):
    """Pull a model via Ollama."""
    ollama_url = state.get("ollama_url", OLLAMA_BASE_URL)
    if not ollama_is_running(ollama_url):
        display.print_error("Ollama is not running. Start it with: ollama serve")
        return
    display.console.print(f"[dim]Pulling [bold]{model_name}[/bold] …[/dim]")
    ok = ollama_pull_model(model_name, ollama_url)
    if ok:
        display.console.print(f"[bold green]✓ {model_name} pulled successfully.[/bold green]")
        display.console.print(f"[dim]Use it with: /local {model_name}[/dim]")
    else:
        display.print_error(f"Failed to pull {model_name}.")


def _handle_research_roles(arg: str, state: dict, cfg: dict):
    """View or modify per-role model assignments for the research pipeline."""
    backend = state.get("backend", "einfra")
    tokens = arg.split() if arg else []

    # /research-roles reset [backend]
    if tokens and tokens[0] == "reset":
        target_backend = tokens[1] if len(tokens) > 1 else backend
        if target_backend not in ("einfra", "nim", "ollama"):
            display.print_error(f"Unknown backend '{target_backend}'. Use einfra, nim, or ollama.")
            return
        reset_role_models(target_backend)
        cfg.update(load_config())
        display.console.print(
            f"[dim]Custom role models cleared for backend[/dim] [bold]{target_backend}[/bold]"
        )
        return

    # /research-roles <role> <model>
    if len(tokens) >= 2:
        role_name = tokens[0].lower()
        model_name = tokens[1]
        if role_name not in PIPELINE_ROLES:
            display.print_error(
                f"Unknown role '{role_name}'. Valid roles: {', '.join(PIPELINE_ROLES)}"
            )
            return
        save_role_model(role_name, model_name, backend)
        cfg.update(load_config())
        display.console.print(
            f"[dim]Role[/dim] [bold cyan]{role_name}[/bold cyan] "
            f"[dim]→[/dim] [bold magenta]{model_name}[/bold magenta] "
            f"[dim](backend: {backend})[/dim]"
        )
        return

    # /research-roles — show current assignments
    effective = get_role_models(cfg)
    display.print_role_models_config(backend, effective, cfg.get(f"role_models_{backend}") or {})


def _handle_long_research(arg: str, state: dict, cfg: dict, client):
    """Parse /long-research flags and launch the research pipeline."""
    import shlex

    try:
        tokens = shlex.split(arg)
    except ValueError:
        tokens = arg.split()

    topic_parts: list[str] = []
    max_rounds = 5
    all_model: str | None = None
    overseer_model: str | None = None
    role_flag_overrides: dict[str, str] = {}
    resume = False
    num_parallel = 1
    scrape_mode = False

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
        elif t == "--parallel" and i + 1 < len(tokens):
            try:
                num_parallel = int(tokens[i + 1])
                if num_parallel < 1:
                    raise ValueError
            except ValueError:
                display.print_error(f"--parallel expects a positive integer, got: {tokens[i+1]}")
                return
            i += 2
        elif t == "--all" and i + 1 < len(tokens):
            all_model = tokens[i + 1]
            i += 2
        elif t == "--overseer" and i + 1 < len(tokens):
            overseer_model = tokens[i + 1]
            i += 2
        elif t == "--role" and i + 2 < len(tokens):
            role_name = tokens[i + 1].lower()
            role_model = tokens[i + 2]
            if role_name not in PIPELINE_ROLES:
                display.print_error(
                    f"Unknown role '{role_name}'. Valid roles: {', '.join(PIPELINE_ROLES)}"
                )
                return
            role_flag_overrides[role_name] = role_model
            i += 3
        elif t == "--resume":
            resume = True
            i += 1
        elif t == "--scrape":
            scrape_mode = True
            i += 1
        else:
            topic_parts.append(t)
            i += 1

    topic = " ".join(topic_parts).strip()
    if not topic:
        display.print_error(
            "Usage: /long-research <topic> [--rounds N] [--parallel N] "
            "[--all MODEL] [--overseer MODEL] [--role ROLE MODEL] [--resume] [--scrape]"
        )
        return

    overrides: dict[str, str] = {}

    if state["backend"] == "ollama" and not all_model:
        # Auto-assign local models across roles then apply any saved custom overrides
        pulled = ollama_list_models(state.get("ollama_url", OLLAMA_BASE_URL))
        if not pulled:
            display.print_error("No Ollama models available for local research.")
            return
        overrides = assign_local_models(pulled)
        # Apply saved custom ollama role overrides
        overrides.update(cfg.get("role_models_ollama") or {})
        display.print_local_research_assignment(overrides)
    elif all_model:
        for role in PIPELINE_ROLES:
            overrides[role] = all_model
    else:
        # Load per-backend defaults + any saved custom role overrides
        overrides = get_role_models(cfg)

    # --overseer flag takes priority over everything for the orchestrator
    if overseer_model:
        overrides["orchestrator"] = overseer_model

    # --role flags take final priority
    overrides.update(role_flag_overrides)

    run_long_research(
        topic=topic,
        working_dir=state["working_dir"],
        client=client,
        max_rounds=max_rounds,
        model_overrides=overrides,
        resume=resume,
        num_parallel=num_parallel,
        scrape_mode=scrape_mode,
    )


def _handle_vault_improve(arg: str, state: dict, client):
    """Parse /vault-improve flags and launch the autonomous vault pipeline."""
    import shlex

    try:
        tokens = shlex.split(arg)
    except ValueError:
        tokens = arg.split()

    vault_path: str | None = None
    resume = False
    model: str | None = None

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--resume":
            resume = True
            i += 1
        elif t == "--model" and i + 1 < len(tokens):
            model = tokens[i + 1]
            i += 2
        elif not t.startswith("--"):
            vault_path = str(Path(t).expanduser().resolve())
            i += 1
        else:
            display.print_error(f"Unknown flag: {t}")
            return

    if not vault_path:
        # Default to current working dir
        vault_path = state["working_dir"]

    if not Path(vault_path).is_dir():
        display.print_error(f"Not a directory: {vault_path}")
        return

    profile = state.get("prompt_profile", "base")

    display.console.print()
    display.console.print(
        f"[bold bright_white]🐙 VAULT IMPROVE[/bold bright_white]\n"
        f"[dim]vault   :[/dim] {vault_path}\n"
        f"[dim]profile :[/dim] {profile}\n"
        f"[dim]model   :[/dim] {model or 'role defaults'}\n"
        f"[dim]resume  :[/dim] {resume}"
    )
    display.console.print()

    try:
        run_vault_improve(
            vault_path=vault_path,
            client=client,
            prompt_profile=profile,
            model=model,
            resume=resume,
        )
    except KeyboardInterrupt:
        display.console.print("\n[dim]Vault improve interrupted. Run again with --resume to continue.[/dim]")


# ---------------------------------------------------------------------------
# Prompt-toolkit helpers
# ---------------------------------------------------------------------------

def _make_prompt(state: dict):
    model_short = state["model"][:20]
    backend = state.get("backend", "einfra")
    if backend == "ollama":
        return HTML(f'<prompt-local>◆</prompt-local> <model-tag>[local:{model_short}]</model-tag> ')
    if backend == "nim":
        return HTML(f'<prompt-local>◆</prompt-local> <model-tag>[nim:{model_short}]</model-tag> ')
    return HTML(f'<prompt>◆</prompt> <model-tag>[{model_short}]</model-tag> ')


def _make_toolbar(state: dict):
    wd = state["working_dir"]
    if len(wd) > 45:
        wd = "…" + wd[-43:]
    backend = state.get("backend", "einfra")
    if backend == "ollama":
        backend_tag = " [local]"
    elif backend == "nim":
        backend_tag = " [nim]"
    else:
        backend_tag = ""
    profile = state.get("prompt_profile", "base")
    perm_mode = state.get("permission_mode", "autonomous")
    if perm_mode == "autonomous":
        perm_short = "auto"
    elif perm_mode == "controlled":
        perm_short = "ctrl"
    else:
        perm_short = "supv"
    return HTML(
        f'<bottom-toolbar>  dir: {wd}{backend_tag}  profile:{profile}  perm:{perm_short}'
        f'   /help · /model · /profile · /permission · /local · /einfra · /nim · /clear · /exit</bottom-toolbar>'
    )


def _make_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-l")
    def _clear_screen(event):
        event.app.renderer.clear()

    return kb


# ---------------------------------------------------------------------------
# Project directory helper
# ---------------------------------------------------------------------------

def _make_project_dir(task: str) -> str:
    """
    Create ~/octoslave/projects/<slug> from the task description.
    Returns the absolute path (already created).
    """
    import re
    slug = task.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)       # remove punctuation
    slug = re.sub(r"[\s_]+", "-", slug)         # spaces → dashes
    slug = slug[:48].strip("-")                 # max 48 chars
    if not slug:
        slug = "project"
    projects_root = Path.home() / "octoslave" / "projects"
    project_dir = projects_root / slug
    project_dir.mkdir(parents=True, exist_ok=True)
    return str(project_dir)


# ---------------------------------------------------------------------------
# Config resolution helper
# ---------------------------------------------------------------------------

def _resolve_config(model, working_dir, api_key, base_url, local: bool = False) -> dict:
    saved = load_config()

    # Decide backend
    backend = "ollama" if local else saved.get("backend", "einfra")
    ollama_url = saved.get("ollama_url", OLLAMA_BASE_URL)

    if backend == "ollama":
        # Validate Ollama is reachable
        if not ollama_is_running(ollama_url):
            display.print_error(
                f"Ollama is not running at {ollama_url}.\n"
                "Start it with: [bold]ollama serve[/bold]"
            )
            sys.exit(1)
        pulled = ollama_list_models(ollama_url)
        if not pulled:
            display.print_error(
                "No models pulled in Ollama.\n"
                "Pull one with: [bold]ollama pull mistral[/bold]"
            )
            sys.exit(1)
        chosen_model = model or saved.get("default_model") or pulled[0]
        if chosen_model not in pulled:
            display.console.print(
                f"[dim]Model '{chosen_model}' not found locally, "
                f"using '{pulled[0]}' instead.[/dim]"
            )
            chosen_model = pulled[0]
        return {
            "api_key":     "ollama",
            "base_url":    ollama_url,
            "model":       chosen_model,
            "working_dir": str(Path(working_dir).resolve()) if working_dir else os.getcwd(),
            "backend":     "ollama",
            "ollama_url":  ollama_url,
            "nim_api_key": saved.get("nim_api_key", ""),
            "nim_url":     saved.get("nim_url", NIM_BASE_URL),
        }

    if backend == "nim":
        nim_api_key = saved.get("nim_api_key", "")
        if not nim_api_key:
            display.print_error(
                "No NVIDIA NIM API key configured.\n"
                "Run [bold]ots config[/bold] and choose the nim backend, "
                "or set OCTOSLAVE_NIM_API_KEY."
            )
            sys.exit(1)
        nim_url = saved.get("nim_url", NIM_BASE_URL)
        return {
            "api_key":     nim_api_key,
            "base_url":    nim_url,
            "model":       model or saved.get("default_model", NIM_DEFAULT_MODEL),
            "working_dir": str(Path(working_dir).resolve()) if working_dir else os.getcwd(),
            "backend":     "nim",
            "ollama_url":  ollama_url,
            "nim_api_key": nim_api_key,
            "nim_url":     nim_url,
        }

    # e-INFRA CZ backend
    return {
        "api_key":     api_key or saved.get("api_key", ""),
        "base_url":    base_url or saved.get("base_url", BASE_URL),
        "model":       model or saved.get("default_model", DEFAULT_MODEL),
        "working_dir": str(Path(working_dir).resolve()) if working_dir else os.getcwd(),
        "backend":     "einfra",
        "ollama_url":  ollama_url,
        "nim_api_key": saved.get("nim_api_key", ""),
        "nim_url":     saved.get("nim_url", NIM_BASE_URL),
    }


@cli.command("vault-improve")
@click.argument("vault_path", default=None, required=False)
@click.option("-p", "--profile", "prompt_profile", default="base",
              help="Prompt profile (default: base, options: base, coder, analyst, biomedic)")
@click.option("-m", "--model", default=None, help="Model override for all vault agents")
@click.option("--resume", is_flag=True, default=False, help="Resume interrupted run")
@click.option("--api-key", default=None, envvar="OCTOSLAVE_API_KEY")
@click.option("--base-url", default=None, envvar="OCTOSLAVE_BASE_URL")
def vault_improve_cmd(vault_path, prompt_profile, model, resume, api_key, base_url):
    """Autonomously improve every note in a vault (Obsidian / markdown folder).

    \b
    Examples:
      octoslave vault-improve ~/Brain2 --profile biomedic
      octoslave vault-improve ~/Brain2 --profile biomedic --resume
      octoslave vault-improve ~/Brain2 --model deepseek-v3.2-thinking
    """
    from .vault import run_vault_improve
    from .agent import make_client

    cfg = _resolve_config(None, vault_path, api_key, base_url)
    vault = str(Path(vault_path).expanduser().resolve()) if vault_path else os.getcwd()

    if not Path(vault).is_dir():
        display.print_error(f"Not a directory: {vault}")
        sys.exit(1)

    client = make_client(cfg["api_key"], cfg["base_url"])

    display.console.print()
    display.console.print(
        f"[bold bright_white]🐙 VAULT IMPROVE[/bold bright_white]\n"
        f"[dim]vault  :[/dim] {vault}\n"
        f"[dim]profile:[/dim] {prompt_profile}\n"
        f"[dim]model  :[/dim] {model or 'role defaults'}\n"
        f"[dim]resume :[/dim] {resume}"
    )
    display.console.print()

    try:
        run_vault_improve(
            vault_path=vault,
            client=client,
            prompt_profile=prompt_profile,
            model=model,
            resume=resume,
        )
    except KeyboardInterrupt:
        display.console.print("\n[dim]Interrupted. Run with --resume to continue.[/dim]")
        sys.exit(0)


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.option("--port", default=7860, show_default=True, help="Port to listen on")
@click.option("--no-browser", is_flag=True, default=False, help="Do not open browser automatically")
def web(host, port, no_browser):
    """Launch the OctoSlave web UI in a browser."""
    try:
        import uvicorn
    except ImportError:
        display.print_error(
            "uvicorn is not installed. Run:  pip install 'octoslave[web]'  or  pip install uvicorn fastapi"
        )
        sys.exit(1)

    url = f"http://{host}:{port}"
    display.console.print()
    display.console.print(
        f"  [bold #2ab89a]🐙 OctoSlave Web UI[/bold #2ab89a]  "
        f"[dim]starting at[/dim]  [bold cyan]{url}[/bold cyan]"
    )
    display.console.print("  [dim]Press Ctrl+C to stop.[/dim]\n")

    if not no_browser:
        import threading, webbrowser
        # Open browser after a short delay so the server is ready
        def _open():
            import time; time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    from .web.app import app as _web_app
    uvicorn.run(_web_app, host=host, port=port, log_level="warning")


def main():
    cli()


if __name__ == "__main__":
    main()
