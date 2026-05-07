@echo off
rem Build OctoSlave Windows Installer (.exe)
rem
rem Requirements (install once):
rem   pip install pyinstaller
rem   Download + install Inno Setup 6 from https://jrsoftware.org/isinfo.php
rem
rem Usage:  cd <repo-root>  &&  installers\windows\build_installer.bat
rem Output: dist\OctoSlave-Windows-Installer.exe

setlocal enabledelayedexpansion
set ROOT=%~dp0..\..

echo =^> Checking PyInstaller...
python -m pip install --quiet --upgrade pyinstaller
if errorlevel 1 (echo ERROR: pip/pyinstaller failed & exit /b 1)

echo =^> Building ots.exe (CLI binary)...
python -m PyInstaller --clean --noconfirm "%ROOT%\installers\windows\ots_cli.spec"
if errorlevel 1 (echo ERROR: ots.exe build failed & exit /b 1)

echo =^> Building OctoSlave-Setup.exe (first-run wizard)...
python -m PyInstaller --clean --noconfirm "%ROOT%\installers\windows\ots_wizard.spec"
if errorlevel 1 (echo ERROR: wizard build failed & exit /b 1)

rem Locate Inno Setup compiler (common install paths)
set ISCC=
for %%p in (
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) do (
    if exist %%p set ISCC=%%p
)

if not defined ISCC (
    echo ERROR: Inno Setup 6 not found.
    echo        Download from: https://jrsoftware.org/isinfo.php
    exit /b 1
)

echo =^> Running Inno Setup compiler...
%ISCC% "%ROOT%\installers\windows\installer.iss"
if errorlevel 1 (echo ERROR: Inno Setup build failed & exit /b 1)

echo.
echo Done! Installer: %ROOT%\dist\OctoSlave-Windows-Installer.exe
endlocal
