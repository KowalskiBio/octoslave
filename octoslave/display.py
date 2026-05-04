"""Rich terminal display helpers for octoslave."""

import json
import sys
import time as _time
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

import threading as _threading

# ---------------------------------------------------------------------------
# Web UI event bridge  (no-op when not in web mode)
# ---------------------------------------------------------------------------

_tl = _threading.local()  # thread-local storage for per-session callbacks

# Permission request state — shared across threads (agent sets, WS handler resolves)
_perm_lock = _threading.Lock()
_perm_event: "_threading.Event | None" = None
_perm_result: bool = False


def set_event_callback(cb) -> None:
    """Register a structured-event callback for the current thread (web mode)."""
    _tl.emit = cb


def clear_event_callback() -> None:
    """Remove the callback for the current thread."""
    _tl.emit = None


def resolve_permission(allow: bool) -> None:
    """Called from the web handler to unblock a pending permission request."""
    global _perm_result, _perm_event
    with _perm_lock:
        _perm_result = allow
        if _perm_event is not None:
            _perm_event.set()


def _emit(event: dict) -> None:
    """Fire a structured JSON event to the registered callback; no-op otherwise."""
    cb = getattr(_tl, "emit", None)
    if cb is not None:
        try:
            cb(event)
        except Exception:
            pass


_THEME = Theme(
    {
        "tool.name": "bold cyan",
        "tool.arg": "dim white",
        "tool.ok": "dim green",
        "tool.err": "bold red",
        "info": "dim",
        "heading": "bold bright_white",
        "model": "bright_magenta",
        "prompt": "bold yellow",
        "mascot": "#1A6B5C",
    }
)

console = Console(theme=_THEME, highlight=False)
err_console = Console(stderr=True, theme=_THEME)

# ---------------------------------------------------------------------------
# Verbose mode
# ---------------------------------------------------------------------------

_verbose: bool = False


def set_verbose(v: bool) -> None:
    global _verbose
    _verbose = v


def is_verbose() -> bool:
    return _verbose


# ---------------------------------------------------------------------------
# Pixel-art octopus mascot  (20 chars wide)
#
# Encoding key (one ASCII char → rendered glyph + Rich style):
#   B  body           H  highlight (top of head)
#   W  eye white      *  pupil (◉)
#   M  mouth (▄)      T  tentacle
#   L  curl ╰         R  curl ╯
#   G  gold chain ◆   g  chain connector ─
#   P  pendant ◈      (space)  empty
# ---------------------------------------------------------------------------

_CHAR_MAP: dict[str, tuple[str, str | None]] = {
    "B": ("█", "bold #1A6B5C"),     # body — deep teal
    "H": ("█", "bold #2ab89a"),     # top-of-head highlight — lighter teal
    "W": ("█", "bold #ffffff"),     # eye whites
    "*": ("◉", "bold #0D3D30"),     # pupil — dark teal bullseye
    "M": ("▄", "bold #ff99cc"),     # mouth — pink lower-half block
    "T": ("█", "#0d3d30"),          # tentacles — darker teal
    "L": ("╰", "#0d3d30"),          # tentacle curl left
    "R": ("╯", "#0d3d30"),          # tentacle curl right
    "G": ("◆", "bold #D4A017"),     # gold chain link
    "g": ("─", "#C8980A"),          # gold chain connector
    "P": ("◈", "bold #F5D060"),     # gold pendant
    " ": (" ", None),
}

# fmt: off
_RAW_MASCOT = [
    "     HHHHHHHHHH     ",   # top of head dome (lighter teal)
    "   BBBBBBBBBBBBBB   ",
    "  BBBBBBBBBBBBBBBB  ",
    " BBBBBBBBBBBBBBBBBB ",
    " BBWWWWWBBBWWWWWBBB ",   # eye whites top  (5 wide, flush to body)
    " BBWW*WWBBBWW*WWBBB ",   # single centered pupil per eye
    " BBWWWWWBBBWWWWWBBB ",   # eye whites bottom
    " BBBBBBBBBBBBBBBBBB ",   # body
    "   BBBB MMMMM BBBB  ",   # cute pink mouth
    " GgGgGgGgGgGgGgGgGg ",   # gold chain draped across mantle
    "    BBBBB P BBBBB   ",   # body below chain + gold pendant ◈
    "  TT   TT   TT   TT ",   # tentacle stems ×4
    "  TT   TT   TT   TT ",   #   (two rows for length)
    " LTTR LTTR LTTR LTTR",   # tentacle curls ╰TT╯
]
# fmt: on

assert all(len(r) == 20 for r in _RAW_MASCOT), "mascot row width mismatch"
_GRID = [[_CHAR_MAP[c] for c in row] for row in _RAW_MASCOT]


def _render_mascot() -> Text:
    text = Text()
    for row in _GRID:
        for ch, style in row:
            if style:
                text.append(ch, style=style)
            else:
                text.append(ch)
        text.append("\n")
    return text


# ---------------------------------------------------------------------------
# Session header
# ---------------------------------------------------------------------------

def print_welcome(model: str, working_dir: str, backend: str = "einfra"):
    mascot = _render_mascot()

    tag = Text()
    tag.append(" OCTOSLAVE ", style="bold bright_white on #1A6B5C")
    if backend == "ollama":
        tag.append(" LOCAL ", style="bold bright_white on #004a20")
    elif backend == "nim":
        tag.append(" NIM ", style="bold bright_white on #004a5c")

    wd = working_dir if len(working_dir) <= 40 else "…" + working_dir[-38:]

    info = Text()
    if backend == "ollama":
        info.append("backend ", style="dim")
        info.append("ollama (local)", style="bold bright_green")
        info.append("   model ", style="dim")
        info.append(model, style="bold bright_green")
    elif backend == "nim":
        info.append("backend ", style="dim")
        info.append("NVIDIA NIM", style="bold bright_cyan")
        info.append("   model ", style="dim")
        info.append(model, style="bold bright_cyan")
    else:
        info.append("model ", style="dim")
        info.append(model, style="bold bright_magenta")
    info.append("   dir ", style="dim")
    info.append(wd, style="dim white")

    hint = Text("  /help for commands", style="dim")
    if backend == "ollama":
        hint = Text("  /help · /pull <model> · /einfra to switch back", style="dim")
    elif backend == "nim":
        hint = Text("  /help · /model · /einfra to switch back", style="dim")

    body = Text()
    body.append_text(mascot)
    body.append("\n")
    body.append_text(tag)
    body.append("\n")
    body.append_text(info)
    body.append("\n")
    body.append_text(hint)
    body.append("\n")

    if backend == "ollama":
        border = "bright_green"
    elif backend == "nim":
        border = "bright_cyan"
    else:
        border = "#2ab89a"
    console.print(
        Panel.fit(body, border_style=border, padding=(0, 2)),
        justify="center",
    )
    console.print()


def print_header(model: str, working_dir: str, backend: str = "einfra"):
    """Compact header for non-interactive (one-shot) runs."""
    if backend == "ollama":
        backend_str = "[bold bright_green]ollama (local)[/bold bright_green]"
        border = "bright_green"
    elif backend == "nim":
        backend_str = "[bold bright_cyan]NVIDIA NIM[/bold bright_cyan]"
        border = "bright_cyan"
    else:
        backend_str = "[model]e-INFRA CZ[/model]"
        border = "#2ab89a"
    console.print(
        Panel.fit(
            f"[heading]OctoSlave[/heading]  {backend_str}  [model]{model}[/model]\n"
            f"[info]dir: {working_dir}[/info]",
            border_style=border,
            padding=(0, 2),
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Task display
# ---------------------------------------------------------------------------

def print_task(task: str):
    console.print(Panel(task, title="[prompt]◆ Task[/prompt]", border_style="yellow", padding=(0, 1)))
    console.print()


# ---------------------------------------------------------------------------
# Streaming text
# ---------------------------------------------------------------------------

_stream_state = _threading.local()

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_CLEAR  = "\r" + " " * 32 + "\r"


def _spinning(stop_event: _threading.Event) -> None:
    """Background thread: animate a waiting indicator until stop_event is set."""
    i = 0
    while not stop_event.wait(timeout=0.1):
        frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
        elapsed = int(i * 0.1)
        sys.stdout.write(f"\r  {frame} waiting for model… {elapsed}s")
        sys.stdout.flush()
        i += 1
    sys.stdout.write(_SPINNER_CLEAR)
    sys.stdout.flush()


def _streaming_started() -> bool:
    return getattr(_stream_state, "started", False)


def stream_start():
    _stream_state.started = False
    _emit({"type": "stream_start"})
    # CLI-only spinner — skip when a web event callback is active
    if getattr(_tl, "emit", None) is not None:
        _stream_state.stop_event = None
        _stream_state.spinner_thread = None
        return
    stop = _threading.Event()
    _stream_state.stop_event = stop
    t = _threading.Thread(target=_spinning, args=(stop,), daemon=True)
    _stream_state.spinner_thread = t
    t.start()


def stream_chunk(text: str):
    _emit({"type": "token", "text": text})
    if not _streaming_started():
        # First token — kill the spinner and emit the response prefix
        stop: _threading.Event = getattr(_stream_state, "stop_event", None)
        if stop is not None:
            stop.set()
            t: _threading.Thread = getattr(_stream_state, "spinner_thread", None)
            if t is not None:
                t.join(timeout=0.3)   # wait for the clear to flush
        console.print("[bold green]●[/bold green] ", end="")
        _stream_state.started = True
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_end(had_content: bool):
    _emit({"type": "stream_end"})
    # Stop the spinner if no content ever arrived (e.g. tool-call-only response)
    stop: _threading.Event = getattr(_stream_state, "stop_event", None)
    if stop is not None:
        stop.set()
        t: _threading.Thread = getattr(_stream_state, "spinner_thread", None)
        if t is not None:
            t.join(timeout=0.3)
    if had_content:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _stream_state.started = False


# ---------------------------------------------------------------------------
# Tool display
# ---------------------------------------------------------------------------

# Icons per tool
_TOOL_ICONS = {
    "read_file":  "📄",
    "write_file": "✏️ ",
    "edit_file":  "🔧",
    "bash":       "⚡",
    "glob":       "🔍",
    "grep":       "🔎",
    "list_dir":   "📁",
    "web_search": "🌐",
    "web_fetch":  "🌍",
}


def print_tool_call(name: str, args: dict):
    _emit({"type": "tool_call", "name": name, "summary": _tool_summary(name, args)})
    icon = _TOOL_ICONS.get(name, "⚙")
    summary = _tool_summary(name, args)
    console.print(f"  [tool.name]{icon} {name}[/tool.name] [tool.arg]{summary}[/tool.arg]")

    if not _verbose:
        return

    from rich.markup import escape as _escape
    # Verbose: show full content of the call
    if name == "edit_file":
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        console.print("    [bold red]─── removing ─────────────────────────────[/bold red]")
        for ln in old.splitlines():
            console.print(f"    [red]- {_escape(ln)}[/red]")
        console.print("    [bold green]─── inserting ────────────────────────────[/bold green]")
        for ln in new.splitlines():
            console.print(f"    [green]+ {_escape(ln)}[/green]")
        console.print("    [dim]──────────────────────────────────────────[/dim]")

    elif name == "write_file":
        content = args.get("content", "")
        lines = content.splitlines()
        console.print(f"    [dim]─── content ({len(lines)} lines) ─────────────────[/dim]")
        for ln in lines:
            console.print(f"    [dim white]{_escape(ln)}[/dim white]")
        console.print("    [dim]──────────────────────────────────────────[/dim]")

    elif name == "bash":
        cmd = args.get("command", "")
        console.print(f"    [bold yellow]$ {cmd}[/bold yellow]")

    elif name == "web_fetch":
        console.print(f"    [dim]↳ {args.get('url', '')}[/dim]")

    elif name == "web_search":
        console.print(f"    [dim]↳ \"{args.get('query', '')}\"[/dim]")


def print_tool_result(name: str, result: str, success: bool):
    from rich.markup import escape as _escape
    _emit({"type": "tool_result", "name": name, "ok": success,
           "preview": (result.strip()[:400] if result.strip() else "")})
    if not result.strip():
        return
    if not success:
        console.print(f"    [tool.err]✗ {_escape(result.strip())}[/tool.err]")
        return
    lines = result.splitlines()
    if _verbose:
        for ln in lines:
            console.print(f"    [tool.ok]{_escape(ln)}[/tool.ok]")
    else:
        show = lines[:6]
        preview = "\n".join(f"    {_escape(ln)}" for ln in show)
        if len(lines) > 6:
            preview += f"\n    [info]… {len(lines) - 6} more lines[/info]"
        console.print(f"[tool.ok]{preview}[/tool.ok]")


def _tool_summary(name: str, args: dict) -> str:
    if name == "read_file":
        return args.get("path", "")
    if name in ("write_file", "edit_file"):
        return args.get("path", "")
    if name == "bash":
        cmd = args.get("command", "")
        return (cmd[:90] + "…") if len(cmd) > 90 else cmd
    if name == "glob":
        return args.get("pattern", "") + (f"  in {args['path']}" if args.get("path") else "")
    if name == "grep":
        return args.get("pattern", "") + (f"  in {args['path']}" if args.get("path") else "")
    if name == "list_dir":
        return args.get("path", ".")
    if name == "web_search":
        return args.get("query", "")
    if name == "web_fetch":
        url = args.get("url", "")
        return (url[:80] + "…") if len(url) > 80 else url
    return json.dumps(args)[:80]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def print_separator():
    console.print(Rule(style="dim"))


def print_info(msg: str):
    _emit({"type": "info", "text": msg})
    console.print(f"[info]{msg}[/info]")


def print_error(msg: str):
    _emit({"type": "error", "text": msg})
    err_console.print(f"[tool.err]Error:[/tool.err] {msg}")


def print_done(iterations: int):
    _emit({"type": "done", "iterations": iterations})
    console.print()
    console.print(f"[info]─── done ({iterations} iteration{'s' if iterations != 1 else ''}) ───[/info]")
    console.print()


def print_local_research_assignment(assignment: dict[str, str]):
    """Show which local model was assigned to each research role."""
    console.print()
    console.print("[bold bright_green]● Local research model assignment:[/bold bright_green]")
    # Group by model to keep output compact
    model_to_roles: dict[str, list[str]] = {}
    for role, model in assignment.items():
        model_to_roles.setdefault(model, []).append(role)
    for model, roles in model_to_roles.items():
        console.print(f"  [bold]{model}[/bold]  →  {', '.join(roles)}")
    console.print(
        "[dim]  (up to 3 local models used; roles distributed by priority tier)[/dim]\n"
    )


def print_role_models_config(backend: str, effective: dict, custom: dict):
    """Show current per-role model assignments and which ones are customised."""
    from rich.table import Table
    console.print()
    console.print(
        f"[bold white]Research role models[/bold white]  "
        f"[dim](backend: {backend})[/dim]"
    )
    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    table.add_column("Role", style="bold cyan", min_width=14)
    table.add_column("Model", style="")
    table.add_column("", style="dim")
    for role, model in effective.items():
        flag = "[yellow]custom[/yellow]" if role in custom else "[dim]default[/dim]"
        table.add_row(role, model, flag)
    console.print(table)
    console.print(
        "[dim]  /research-roles <role> <model>   — set a custom model for a role[/dim]\n"
        "[dim]  /research-roles reset             — clear all custom overrides[/dim]\n"
    )


def print_help():
    console.print(Panel(
        "[bold white]Slash commands[/bold white]\n\n"
        "  [cyan]/model [NAME][/cyan]           Switch model (or list if no name given)\n"
        "  [cyan]/dir [PATH][/cyan]             Change working directory\n"
        "  [cyan]/profile [NAME][/cyan]         Switch prompt profile (base/coder/analyst/biomedic)\n"
        "  [cyan]/permission [MODE][/cyan]      Switch permission mode (autonomous/controlled)\n"
        "  [cyan]/verbose[/cyan]                Toggle verbose mode (show full diffs & output live)\n"
        "  [cyan]/clear[/cyan]                  Clear screen and conversation history\n"
        "  [cyan]/compact[/cyan]                Summarise history to save context\n"
        "  [cyan]/long-research TOPIC[/cyan]    Launch multi-agent research pipeline\n"
        "  [cyan]/research-roles[/cyan]         View/set per-role models for research pipeline\n"
        "  [cyan]/vault-improve [PATH][/cyan]   Autonomously improve all notes in vault\n"
        "  [cyan]/help[/cyan]                   Show this help\n"
        "  [cyan]/exit[/cyan]                   Quit  (also Ctrl+D)\n\n"
        "[bold white]Backend switching:[/bold white]\n"
        "  [cyan]/local [MODEL][/cyan]          Switch to local Ollama models\n"
        "  [cyan]/einfra[/cyan]                 Switch to e-INFRA CZ\n"
        "  [cyan]/nim [MODEL][/cyan]            Switch to NVIDIA NIM\n"
        "  [cyan]/pull MODEL[/cyan]             Pull a new Ollama model\n\n"
        "[bold white]Permission modes:[/bold white]\n"
        "  [cyan]autonomous[/cyan]  — work without asking (default)\n"
        "  [cyan]controlled[/cyan]  — ask before file edits or commands\n"
        "  [cyan]supervised[/cyan]  — ask before file edits, auto-allow commands\n\n"
        "[bold white]/long-research flags:[/bold white]\n"
        "  [cyan]--rounds N[/cyan]              Number of research rounds (default 5)\n"
        "  [cyan]--parallel N[/cyan]            Run N parallel copies of researcher/hypothesis/evaluator\n"
        "  [cyan]--overseer MODEL[/cyan]        Model for orchestrator\n"
        "  [cyan]--all MODEL[/cyan]             Use one model for all agents\n"
        "  [cyan]--role ROLE MODEL[/cyan]       Override model for a specific role\n"
        "  [cyan]--resume[/cyan]                Resume an interrupted run\n"
        "  [cyan]--scrape[/cyan]                Scrape mode: crawl website tree\n"
        "  [dim](local mode: up to 3 pulled models auto-assigned across roles)[/dim]\n\n"
        "[bold white]/research-roles usage:[/bold white]\n"
        "  [cyan]/research-roles[/cyan]                   Show current role assignments\n"
        "  [cyan]/research-roles coder qwen3-coder[/cyan]  Set model for a specific role\n"
        "  [cyan]/research-roles reset[/cyan]              Reset to backend defaults\n\n"
        "[bold white]/vault-improve flags:[/bold white]\n"
        "  [cyan]PATH[/cyan]                  Vault directory (default: working dir)\n"
        "  [cyan]--model MODEL[/cyan]         Override model for all vault agents\n"
        "  [cyan]--resume[/cyan]              Resume interrupted pipeline\n"
        "  [dim]Pipeline: Planner → [Editor → Verifier → Fix] per batch → Reporter[/dim]\n\n"
        "[dim]Output per round: plots → 03_code/results/, HTML report → 07_report.html[/dim]\n"
        "[dim]Final master report: research/final_report.html  (open in browser)[/dim]\n\n"
        "[dim]Ctrl+C  pause current agent (progress saved)[/dim]",
        title="[prompt]Help[/prompt]",
        border_style="dim",
        padding=(0, 2),
    ))


# ---------------------------------------------------------------------------
# Multi-agent research display
# ---------------------------------------------------------------------------

def print_research_start(topic: str, max_rounds: int, roles: dict, overrides: dict):
    _emit({"type": "research_start", "topic": topic, "max_rounds": max_rounds})
    lines = Text()
    lines.append("🐙 AUTONOMOUS RESEARCH PIPELINE\n\n", style="bold bright_white")
    lines.append("Topic   : ", style="dim"); lines.append(topic + "\n", style="bold white")
    lines.append("Rounds  : ", style="dim"); lines.append(str(max_rounds) + "\n", style="bold white")
    lines.append("\nAgents  :\n", style="dim")
    for role, cfg in roles.items():
        model = overrides.get(role) or cfg["default_model"]
        lines.append(f"  {cfg['icon']}  ", style="")
        lines.append(f"{cfg['label']:<24}", style=cfg["color"])
        lines.append(f"→  {model}\n", style="dim")
    console.print(Panel(lines, border_style="bright_blue", padding=(0, 2)))
    console.print()


def print_round_header(round_num: int, max_rounds: int, round_dir: str):
    _emit({"type": "round_start", "round": round_num,
           "max_rounds": max_rounds, "dir": round_dir})
    bar_filled = "█" * round_num
    bar_empty  = "░" * (max_rounds - round_num)
    console.print()
    console.print(
        f"[bold bright_blue]{'─'*60}[/bold bright_blue]"
    )
    console.print(
        f"  [bold bright_white]ROUND {round_num} / {max_rounds}[/bold bright_white]  "
        f"[bright_blue]{bar_filled}[/bright_blue][dim]{bar_empty}[/dim]  "
        f"[dim]{round_dir}[/dim]"
    )
    console.print(
        f"[bold bright_blue]{'─'*60}[/bold bright_blue]"
    )
    console.print()


def print_agent_banner(role: str, model: str, round_num: int, max_rounds: int):
    cfg = _get_role_cfg(role)
    _emit({"type": "agent_start", "role": role, "label": cfg["label"],
           "icon": cfg["icon"], "model": model,
           "round": round_num, "max_rounds": max_rounds})
    console.print()
    console.print(
        Panel.fit(
            f"{cfg['icon']}  [{cfg['color']}]{cfg['label'].upper()}[/{cfg['color']}]"
            f"   [dim]model: {model}   round {round_num}/{max_rounds}[/dim]",
            border_style=cfg["color"].replace("bold ", ""),
            padding=(0, 2),
        )
    )
    console.print()


def print_agent_done(role: str, elapsed: float, iterations: int):
    _emit({"type": "agent_done", "role": role,
           "elapsed": f"{elapsed:.0f}s", "iterations": iterations})
    cfg = _get_role_cfg(role)
    console.print(
        f"\n  [{cfg['color']}]✓ {cfg['label']}[/{cfg['color']}] "
        f"[dim]done — {iterations} iter, {elapsed:.0f}s[/dim]"
    )
    console.print()


def print_round_done(round_num: int, round_dir: str):
    _emit({"type": "round_done", "round": round_num, "dir": round_dir})
    console.print(
        f"[dim]  Round {round_num} complete → {round_dir}[/dim]"
    )


def print_research_complete(rounds_done: int, research_dir: str):
    from pathlib import Path as _Path
    report_path = str(_Path(research_dir) / "final_report.html")
    _emit({"type": "research_complete", "rounds": rounds_done, "dir": research_dir, "report_path": report_path})
    console.print()
    console.print(Panel(
        f"[bold bright_white]🎉 Research complete[/bold bright_white]\n\n"
        f"[dim]Rounds completed : {rounds_done}[/dim]\n"
        f"[dim]Output directory : {research_dir}[/dim]\n\n"
        "[white]Key files:[/white]\n"
        f"  [bold cyan]{research_dir}/final_report.html[/bold cyan]    ← master HTML report (open in browser)\n"
        f"  [cyan]{research_dir}/findings.md[/cyan]              ← cumulative findings\n"
        f"  [cyan]{research_dir}/round_*/07_report.html[/cyan]   ← per-round HTML reports\n"
        f"  [cyan]{research_dir}/round_*/06_synthesis.md[/cyan]  ← per-round synthesis\n"
        f"  [cyan]{research_dir}/round_*/03_code/results/[/cyan] ← plots & data",
        border_style="bright_green",
        padding=(0, 2),
    ))
    console.print()


def request_permission(tool_name: str, args: dict, working_dir: str, permission_mode: str = "controlled") -> bool:
    """
    Request user permission for a tool that modifies state.
    Returns True if allowed, False if denied.
    
    In controlled mode, this displays a prompt and waits for user input.
    In supervised mode, the title reflects that bash commands are auto-approved.
    """
    # Build a descriptive message about what the tool will do
    icon = _TOOL_ICONS.get(tool_name, "⚙")
    
    if tool_name == "write_file":
        path = args.get("path", "")
        desc = f"create/overwrite file: {path}"
    elif tool_name == "edit_file":
        path = args.get("path", "")
        desc = f"edit file: {path}"
    elif tool_name == "bash":
        cmd = args.get("command", "")
        cmd_preview = (cmd[:60] + "...") if len(cmd) > 60 else cmd
        desc = f"run command: {cmd_preview}"
    else:
        desc = f"execute: {tool_name}"
    
    # Determine mode title and border color
    if permission_mode == "supervised":
        title = "[bold cyan]Supervised Mode[/bold cyan]"
        border_style = "cyan"
    else:
        title = "[bold cyan]Controlled Mode[/bold cyan]"
        border_style = "yellow"
    
    console.print()
    console.print(
        Panel(
            f"[bold yellow]⚠ Permission Required[/bold yellow]\n\n"
            f"[dim]{icon} {tool_name}[/dim]\n\n"
            f"OctoSlave wants to: [bold]{desc}[/bold]\n\n"
            f"Working directory: [dim]{working_dir}[/dim]",
            title=title,
            border_style=border_style,
            padding=(1, 2),
        )
    )
    console.print()

    # Web mode: emit a permission_request event and wait for the browser response
    if getattr(_tl, "emit", None) is not None:
        global _perm_event, _perm_result
        with _perm_lock:
            _perm_event = _threading.Event()
            _perm_result = False
        _emit({
            "type": "permission_request",
            "tool": tool_name,
            "desc": desc,
            "icon": icon,
            "working_dir": working_dir,
            "mode": permission_mode,
        })
        _perm_event.wait(timeout=300)   # 5-minute timeout → default deny
        with _perm_lock:
            result = _perm_result
            _perm_event = None
        return result

    # Console mode: block on user input
    while True:
        try:
            response = console.input(
                "[bold green]Allow?[/bold green] [cyan](y)[/cyan]/[red](n)[/red] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False

        if response in ("y", "yes", "ok", "allow"):
            return True
        elif response in ("n", "no", "deny"):
            return False
        else:
            console.print("[dim]Please enter 'y' for yes or 'n' for no.[/dim]")


def _get_role_cfg(role: str) -> dict:
    """Import ROLES lazily to avoid circular imports."""
    from .research import ROLES
    return ROLES.get(role, {"label": role, "icon": "⚙", "color": "white"})
