"""Thin entry point for PyInstaller CLI binaries (Windows + Linux).

Using a wrapper avoids relative-import issues: octoslave/main.py is full of
`from .config import ...` which fail when the script is frozen as __main__.
This file uses absolute imports, so PyInstaller can resolve everything cleanly.
"""
from octoslave.main import cli

if __name__ == "__main__":
    cli(standalone_mode=True)
