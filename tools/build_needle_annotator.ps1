param(
  [string]$ProjectRoot = "C:\Users\lenovo\Desktop\JXCX",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
Set-Location $ProjectRoot

& $Python -m pip install --upgrade pillow pyinstaller
& $Python -m PyInstaller `
  --onefile `
  --windowed `
  --name NeedleAnnotator `
  --distpath "$ProjectRoot\dist" `
  --workpath "$ProjectRoot\build\needle_annotator" `
  --specpath "$ProjectRoot\build\needle_annotator" `
  "$ProjectRoot\tools\needle_yolo_annotator.py"

Write-Host "Built: $ProjectRoot\dist\NeedleAnnotator.exe"
