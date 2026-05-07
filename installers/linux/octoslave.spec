# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — Linux single-file binary (ots)
#
# Build:
#   cd <repo-root>
#   pyinstaller installers/linux/octoslave.spec
#
# Output: dist/ots   (used by build_appimage.sh to create the AppImage)

import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent.parent
OCTOSLAVE_PKG = ROOT / "octoslave"

block_cipher = None

datas = [
    (str(OCTOSLAVE_PKG / "prompt_profiles"), "octoslave/prompt_profiles"),
    (str(OCTOSLAVE_PKG / "web" / "static"),  "octoslave/web/static"),
]

hidden = [
    "octoslave.agent", "octoslave.config", "octoslave.display",
    "octoslave.logger", "octoslave.parallel", "octoslave.research",
    "octoslave.tools", "octoslave.tools_bio", "octoslave.vault",
    "octoslave.web.app", "octoslave.wizard",
    "fastapi", "fastapi.routing", "fastapi.middleware",
    "starlette", "starlette.routing", "starlette.responses",
    "starlette.staticfiles", "starlette.websockets",
    "uvicorn", "uvicorn.main", "uvicorn.config",
    "uvicorn.lifespan.on",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "openai", "openai._models", "openai.types",
    "rich", "rich.console", "rich.markdown",
    "prompt_toolkit", "prompt_toolkit.shortcuts",
    "click", "requests", "bs4", "fitz",
    "openpyxl", "docx", "psutil", "multipart",
    "tkinter", "tkinter.ttk", "tkinter.messagebox",
    "email.mime.multipart", "email.mime.text", "email.mime.base",
]

a = Analysis(
    [str(OCTOSLAVE_PKG / "main.py")],
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ots",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
