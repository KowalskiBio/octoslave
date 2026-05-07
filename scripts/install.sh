#!/usr/bin/env bash
#
# OctoSlave one-line installer.
#
#   curl -fsSL https://octoslave.karamazov.website/install | bash
#
# Strategy:
#   1. Pick a Python (>=3.10) — prefer python3.13/3.12/3.11/3.10, fall back to python3.
#   2. Ensure pipx is available (install with the chosen Python if missing).
#   3. Install or upgrade octoslave[all] into pipx.
#   4. Print a friendly "next steps" block.
#
# Environment overrides:
#   OCTOSLAVE_REF        git ref to install (default: latest PyPI release; fallback: main)
#   OCTOSLAVE_REPO       git URL  (default: https://github.com/karatedava/octoslave)
#   OCTOSLAVE_EXTRAS     pip extras spec (default: [all]; use "" to skip extras)
#   OCTOSLAVE_NO_BREW    skip the brew suggestion
#

set -euo pipefail

OCTOSLAVE_REPO="${OCTOSLAVE_REPO:-https://github.com/karatedava/octoslave}"
OCTOSLAVE_REF="${OCTOSLAVE_REF:-}"
OCTOSLAVE_EXTRAS="${OCTOSLAVE_EXTRAS-[all]}"

bold()   { printf "\033[1m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
info()   { printf "  \033[2m%s\033[0m\n" "$*"; }

bold "🐙  OctoSlave installer"
echo

# ── 1. Pick a Python ───────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
        if [ -n "$version" ]; then
            major=${version%%.*}
            minor=${version#*.}
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                PYTHON="$candidate"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    red "No Python ≥ 3.10 found on PATH."
    echo
    echo "  macOS:  brew install python@3.12"
    echo "  Linux:  sudo apt install python3.12 python3.12-venv pipx"
    exit 1
fi

green "✓ using $($PYTHON --version) at $(command -v "$PYTHON")"

# ── 2. Ensure pipx ─────────────────────────────────────────────────────────
if ! command -v pipx >/dev/null 2>&1; then
    yellow "pipx not found — installing it for you"
    "$PYTHON" -m pip install --user --quiet --upgrade pipx
    "$PYTHON" -m pipx ensurepath >/dev/null 2>&1 || true
    # pipx install puts a shim on PATH that may not exist yet in this shell
    PIPX_BIN_DIR="$($PYTHON -m pipx environment --value PIPX_BIN_DIR 2>/dev/null || echo "$HOME/.local/bin")"
    export PATH="$PIPX_BIN_DIR:$PATH"
fi

green "✓ pipx ready"

# ── 3. Install octoslave ───────────────────────────────────────────────────
SPEC="octoslave${OCTOSLAVE_EXTRAS}"
if [ -n "$OCTOSLAVE_REF" ]; then
    SPEC="git+${OCTOSLAVE_REPO}@${OCTOSLAVE_REF}#egg=octoslave${OCTOSLAVE_EXTRAS}"
fi

info "Installing $SPEC via pipx (this may take a minute)…"
echo

if pipx list 2>/dev/null | grep -q "package octoslave "; then
    pipx upgrade octoslave || pipx install --force "$SPEC"
else
    if ! pipx install "$SPEC" 2>/tmp/octoslave-install.log; then
        # If PyPI doesn't have it yet, fall back to git@main
        yellow "PyPI install failed — falling back to git+main"
        pipx install --force "git+${OCTOSLAVE_REPO}@main#egg=octoslave${OCTOSLAVE_EXTRAS}"
    fi
fi

echo
green "✓ octoslave installed"
echo

# ── 4. Next steps ──────────────────────────────────────────────────────────
bold "Next steps:"
echo
echo "  1.  Configure a backend (e-INFRA CZ / NVIDIA NIM / Ollama):"
echo "        ots config"
echo
echo "  2.  Try it:"
echo "        ots                         # interactive TUI"
echo "        ots web                     # browser UI"
echo "        ots run \"hello world\"       # one-shot task"
echo
echo "  3.  Multi-agent mode:"
echo "        ots run \"refactor X\" --parallel 3 --strategy best"
echo

if ! command -v ots >/dev/null 2>&1; then
    yellow "Note: 'ots' is not on your PATH yet — start a new shell, or run:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo
info "Docs: https://github.com/karatedava/octoslave"
