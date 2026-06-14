@echo off
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\start_mug_pipeline_gui.ps1" -ProjectRoot "%ROOT%"
if errorlevel 1 (
  echo.
  echo Mug pipeline GUI failed to start.
  pause
)
