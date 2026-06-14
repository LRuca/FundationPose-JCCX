@echo off
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
python "%ROOT%\tools\needle_yolo_annotator.py" --images "%ROOT%\FoundationPose\live_orbbec" --dataset "%ROOT%\datasets\needle_seg"
if errorlevel 1 (
  echo.
  echo Needle annotator failed to start. Install dependency with:
  echo python -m pip install pillow
  pause
)
