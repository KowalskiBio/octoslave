"""macOS .app entry point.

Detection:
  - No TTY (launched from Finder / double-click) → show GUI launcher
  - TTY present (launched from Terminal) → run normal CLI

On first run (no config) the setup wizard is shown first regardless of mode.
"""
import sys
import os
from pathlib import Path

_CONFIG_FILE = Path.home() / ".octoslave" / "config.json"


def _has_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def main():
    # ── First-run wizard
    if not _CONFIG_FILE.exists():
        try:
            from octoslave.wizard import run_wizard
            run_wizard()
        except Exception as exc:
            print(f"[OctoSlave] wizard error: {exc}", file=sys.stderr)

    # ── Decide mode
    if _has_tty():
        # Launched from Terminal — behave as normal CLI
        from octoslave.main import cli
        cli()
    else:
        # Launched from Finder — show GUI launcher
        from installers.macos.gui_launcher import run_launcher
        run_launcher()


if __name__ == "__main__":
    main()
