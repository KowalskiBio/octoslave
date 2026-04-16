"""Rich terminal display helpers for octoslave."""

import json
import sys
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

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
        "mascot": "magenta",
    }
)

console = Console(theme=_THEME, highlight=False)
err_console = Console(stderr=True, theme=_THEME)


# ---------------------------------------------------------------------------
# Pixel-art octopus mascot  (20 chars wide)
#
# Encoding key (one ASCII char → rendered glyph + Rich style):
#   B  body           H  highlight (top of head)
#   W  eye white      *  pupil (◉)
#   M  mouth (▄)      T  tentacle
#   L  curl ╰         R  curl ╯
#   (space)  empty
# ---------------------------------------------------------------------------

_CHAR_MAP: dict[str, tuple[str, str | None]] = {
    "B": ("█", "bold #9922dd"),     # body — rich purple
    "H": ("█", "bold #cc77ff"),     # top-of-head highlight — lighter purple
    "W": ("█", "bold #ffffff"),     # eye whites
    "*": ("◉", "bold #0a001a"),     # pupil — single centered bullseye
    "M": ("▄", "bold #ff99cc"),     # mouth — pink lower-half block
    "T": ("█", "#6600aa"),          # tentacles — deeper purple
    "L": ("╰", "#6600aa"),          # tentacle curl left
    "R": ("╯", "#6600aa"),          # tentacle curl right
    " ": (" ", None),
}

# fmt: off
_RAW_MASCOT = [
    "     HHHHHHHHHH     ",   # top of head dome
    "   BBBBBBBBBBBBBB   ",
    "  BBBBBBBBBBBBBBBB  ",
    " BBBBBBBBBBBBBBBBBB ",
    " BBWWWWWBBBWWWWWBBB ",   # eye whites top  (5 wide, flush to body)
    " BBWW*WWBBBWW*WWBBB ",   # single centered pupil per eye
    " BBWWWWWBBBWWWWWBBB ",   # eye whites bottom
    " BBBBBBBBBBBBBBBBBB ",   # body
    "   BBBB MMMMM BBBB  ",   # cute pink mouth
    "   BBBBBBBBBBBBBB   ",   # lower body
    "    BBBBBBBBBBBB    ",   # bottom of body
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

def print_welcome(model: str, working_dir: str):
    mascot = _render_mascot()

    tag = Text()
    tag.append(" OCTOSLAVE ", style="bold bright_white on #4a0080")

    wd = working_dir if len(working_dir) <= 40 else "…" + working_dir[-38:]

    info = Text()
    info.append("model ", style="dim")
    info.append(model, style="bold bright_magenta")
    info.append("   dir ", style="dim")
    info.append(wd, style="dim white")

    hint = Text("  /help for commands", style="dim")

    body = Text()
    body.append_text(mascot)
    body.append("\n")
    body.append_text(tag)
    body.append("\n")
    body.append_text(info)
    body.append("\n")
    body.append_text(hint)
    body.append("\n")

    console.print(
        Panel.fit(body, border_style="magenta", padding=(0, 2)),
        justify="center",
    )
    console.print()


def print_header(model: str, working_dir: str):
    """Compact header for non-interactive (one-shot) runs."""
    console.print(
        Panel.fit(
            f"[heading]OctoSlave[/heading]  [model]{model}[/model]\n"
            f"[info]dir: {working_dir}[/info]",
            border_style="magenta",
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

_streaming_started = False


def stream_start():
    global _streaming_started
    _streaming_started = False


def stream_chunk(text: str):
    global _streaming_started
    if not _streaming_started:
        console.print("[bold green]●[/bold green] ", end="")
        _streaming_started = True
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_end(had_content: bool):
    global _streaming_started
    if had_content:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _streaming_started = False


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
    icon = _TOOL_ICONS.get(name, "⚙")
    summary = _tool_summary(name, args)
    console.print(f"  [tool.name]{icon} {name}[/tool.name] [tool.arg]{summary}[/tool.arg]")


def print_tool_result(name: str, result: str, success: bool):
    if not result.strip():
        return
    if not success:
        console.print(f"    [tool.err]✗ {result.strip()}[/tool.err]")
        return
    lines = result.splitlines()
    preview = "\n".join(f"    {ln}" for ln in lines[:6])
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
    console.print(f"[info]{msg}[/info]")


def print_error(msg: str):
    err_console.print(f"[tool.err]Error:[/tool.err] {msg}")


def print_done(iterations: int):
    console.print()
    console.print(f"[info]─── done ({iterations} iteration{'s' if iterations != 1 else ''}) ───[/info]")
    console.print()


def print_help():
    console.print(Panel(
        "[bold white]Slash commands[/bold white]\n\n"
        "  [cyan]/model [NAME][/cyan]         Switch model (or list if no name given)\n"
        "  [cyan]/dir [PATH][/cyan]           Change working directory\n"
        "  [cyan]/clear[/cyan]                Clear screen and conversation history\n"
        "  [cyan]/compact[/cyan]              Summarise history to save context\n"
        "  [cyan]/long-research TOPIC[/cyan]  Launch multi-agent research pipeline\n"
        "  [cyan]/help[/cyan]                 Show this help\n"
        "  [cyan]/exit[/cyan]                 Quit  (also Ctrl+D)\n\n"
        "[bold white]/long-research flags:[/bold white]\n"
        "  [cyan]--rounds N[/cyan]            Number of research rounds (default 5)\n"
        "  [cyan]--overseer MODEL[/cyan]      Model for orchestrator (default mistral-small-4)\n"
        "  [cyan]--all MODEL[/cyan]           Use one model for all agents\n"
        "  [cyan]--resume[/cyan]              Resume an interrupted run\n\n"
        "[dim]Ctrl+C  pause current agent (progress saved)[/dim]",
        title="[prompt]Help[/prompt]",
        border_style="dim",
        padding=(0, 2),
    ))


# ---------------------------------------------------------------------------
# Multi-agent research display
# ---------------------------------------------------------------------------

def print_research_start(topic: str, max_rounds: int, roles: dict, overrides: dict):
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
    cfg = _get_role_cfg(role)
    console.print(
        f"\n  [{cfg['color']}]✓ {cfg['label']}[/{cfg['color']}] "
        f"[dim]done — {iterations} iter, {elapsed:.0f}s[/dim]"
    )
    console.print()


def print_round_done(round_num: int, round_dir: str):
    console.print(
        f"[dim]  Round {round_num} complete → {round_dir}[/dim]"
    )


def print_research_complete(rounds_done: int, research_dir: str):
    console.print()
    console.print(Panel(
        f"[bold bright_white]🎉 Research complete[/bold bright_white]\n\n"
        f"[dim]Rounds completed : {rounds_done}[/dim]\n"
        f"[dim]Output directory : {research_dir}[/dim]\n\n"
        "[white]Key files:[/white]\n"
        f"  [cyan]{research_dir}/findings.md[/cyan]          ← cumulative findings\n"
        f"  [cyan]{research_dir}/round_*/06_synthesis.md[/cyan]  ← per-round synthesis\n"
        f"  [cyan]{research_dir}/round_*/03_code/[/cyan]          ← all code & results",
        border_style="bright_green",
        padding=(0, 2),
    ))
    console.print()


def _get_role_cfg(role: str) -> dict:
    """Import ROLES lazily to avoid circular imports."""
    from .research import ROLES
    return ROLES.get(role, {"label": role, "icon": "⚙", "color": "white"})
