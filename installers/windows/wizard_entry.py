"""Entry point for the Windows first-run setup wizard (windowed, no console).

This is compiled into OctoSlave-Setup.exe by PyInstaller.
It is run automatically by the Inno Setup installer after file extraction,
and can also be re-run from the Start Menu as "Configure OctoSlave".
"""
import sys

def main():
    from octoslave.wizard import run_wizard
    run_wizard()

if __name__ == "__main__":
    main()
