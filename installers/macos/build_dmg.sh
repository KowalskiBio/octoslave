#!/usr/bin/env bash
# Build OctoSlave.dmg for macOS
#
# Requirements (install once):
#   pip install pyinstaller
#   brew install create-dmg
#
# Usage:
#   cd <repo-root>
#   bash installers/macos/build_dmg.sh
#
# Output: dist/OctoSlave-macOS.dmg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"

# ── 1. Install / update PyInstaller if needed
echo "==> Checking PyInstaller…"
python3 -m pip install --quiet --upgrade pyinstaller

# ── 2. Build the .app bundle
echo "==> Building OctoSlave.app with PyInstaller…"
cd "${ROOT_DIR}"
python3 -m PyInstaller \
    --clean \
    --noconfirm \
    "${SCRIPT_DIR}/octoslave.spec"

APP_PATH="${DIST_DIR}/OctoSlave.app"

if [[ ! -d "${APP_PATH}" ]]; then
    echo "ERROR: OctoSlave.app not found at ${APP_PATH}"
    exit 1
fi

# ── 3. (Optional) Ad-hoc code sign so macOS Gatekeeper doesn't block it
#    Replace '-' with your Developer ID if you have one:
#      "Developer ID Application: Your Name (TEAMID)"
echo "==> Code-signing (ad-hoc)…"
codesign --force --deep --sign "-" "${APP_PATH}" || true

# ── 4. Create the DMG
DMG_PATH="${DIST_DIR}/OctoSlave-macOS.dmg"
echo "==> Creating DMG at ${DMG_PATH}…"
rm -f "${DMG_PATH}"

create-dmg \
    --volname "OctoSlave" \
    --volicon "${ROOT_DIR}/octoslave/web/static/logo.png" \
    --window-pos 200 120 \
    --window-size 660 420 \
    --icon-size 128 \
    --icon "OctoSlave.app" 180 200 \
    --hide-extension "OctoSlave.app" \
    --app-drop-link 480 200 \
    --no-internet-enable \
    "${DMG_PATH}" \
    "${APP_PATH}"

echo ""
echo "✓  DMG ready: ${DMG_PATH}"
echo "   Distribute this file — users just double-click, drag to Applications, done."
