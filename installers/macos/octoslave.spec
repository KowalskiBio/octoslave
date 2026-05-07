# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for macOS OctoSlave.app
#
# Build:
#   cd <repo-root>
#   pyinstaller installers/macos/octoslave.spec
#
# Output:  dist/OctoSlave.app
# DMG:     run  installers/macos/build_dmg.sh  after PyInstaller

import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent.parent          # repo root
OCTOSLAVE_PKG = ROOT / "octoslave"

block_cipher = None

# ---------------------------------------------------------------------------
# Data files bundled into the app
# ---------------------------------------------------------------------------
datas = [
    (str(OCTOSLAVE_PKG / "prompt_profiles"), "octoslave/prompt_profiles"),
    (str(OCTOSLAVE_PKG / "web" / "static"),  "octoslave/web/static"),
    # macOS GUI launcher modules
    (str(ROOT / "installers" / "macos" / "gui_launcher.py"), "installers/macos"),
]

# ---------------------------------------------------------------------------
# Hidden imports (modules loaded at runtime by FastAPI / uvicorn / click)
# ---------------------------------------------------------------------------
hidden = [
    # octoslave internals
    "octoslave.agent", "octoslave.config", "octoslave.display",
    "octoslave.logger", "octoslave.parallel", "octoslave.research",
    "octoslave.tools", "octoslave.tools_bio", "octoslave.vault",
    "octoslave.web.app", "octoslave.wizard",
    # installers
    "installers.macos.gui_launcher",
    # FastAPI / Starlette
    "fastapi", "fastapi.routing", "fastapi.middleware",
    "starlette", "starlette.routing", "starlette.responses",
    "starlette.staticfiles", "starlette.websockets",
    # uvicorn
    "uvicorn", "uvicorn.main", "uvicorn.config",
    "uvicorn.lifespan.on", "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.http.httptools_impl",
    # openai
    "openai", "openai._models", "openai.types",
    # rich / prompt_toolkit
    "rich", "rich.console", "rich.markdown",
    "prompt_toolkit", "prompt_toolkit.shortcuts",
    # misc
    "click", "requests", "bs4", "fitz",
    "openpyxl", "docx", "psutil", "multipart",
    # tkinter (bundled with Python on macOS)
    "tkinter", "tkinter.ttk", "tkinter.messagebox",
    # email / http stdlib extras required by httpx/openai
    "email.mime.multipart", "email.mime.text", "email.mime.base",
    "http.server",
]

a = Analysis(
    [str(ROOT / "installers" / "macos" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["rdkit", "anndata", "scipy", "pytesseract"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OctoSlave",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed .app — no terminal
    disable_windowed_traceback=False,
    argv_emulation=True,    # macOS: forward Finder argv
    target_arch=None,       # None = native; set "universal2" for fat binary
    codesign_identity=None,
    entitlements_file=None,
    icon=str(OCTOSLAVE_PKG / "web" / "static" / "logo.png"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OctoSlave",
)

app = BUNDLE(
    coll,
    name="OctoSlave.app",
    icon=str(OCTOSLAVE_PKG / "web" / "static" / "logo.png"),
    bundle_identifier="cz.einfra.octoslave",
    info_plist={
        "CFBundleShortVersionString": "0.2.0",
        "CFBundleVersion":            "0.2.0",
        "NSHighResolutionCapable":    True,
        "NSRequiresAquaSystemAppearance": False,  # respect Dark Mode
        "LSMinimumSystemVersion":     "11.0",
    },
)
