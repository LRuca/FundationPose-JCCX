@echo off
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
python "%ROOT%\tools\yolo_key_capture_viewer.py" --project-root "%ROOT%" --live-dir "%ROOT%\FoundationPose\live_orbbec" --out-dir "%ROOT%\data\needle_inbox_raw" --prefix needle_inbox --start-camera --stop-camera-on-exit
if errorlevel 1 (
  echo.
  echo YOLO key capture failed. Install dependency with:
  echo python -m pip install pillow
  pause
)
