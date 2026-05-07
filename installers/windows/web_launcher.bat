@echo off
rem Convenience wrapper — opens OctoSlave web UI in the default browser.
rem Placed in the installation directory by the Inno Setup installer.
start "" "%~dp0ots.exe" web
