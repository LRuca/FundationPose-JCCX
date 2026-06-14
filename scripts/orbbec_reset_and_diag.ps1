param(
  [string]$ProjectRoot = "C:\Users\lenovo\Desktop\JXCX"
)

$ErrorActionPreference = "Continue"

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$SdkBin = Join-Path $ProjectRoot "OrbbecSDK_v1.10.16_win_x64\OrbbecSDK_v1.10.16\Example\bin"
$LiveDir = Join-Path $ProjectRoot "FoundationPose\live_orbbec"
$LogDir = Join-Path $ProjectRoot "logs\mug_pipeline"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$processNames = @(
  "SaveToDisk",
  "CommonUsages",
  "QuickStart",
  "MultiStream",
  "DepthViewer",
  "ColorViewer",
  "AlignFilterViewer",
  "SyncAlignViewer",
  "PointCloud"
)

"[$(Get-Date -Format o)] Reset Orbbec SDK example processes" | Set-Content -Encoding utf8 (Join-Path $LogDir "orbbec_reset.log")
foreach ($name in $processNames) {
  Get-Process -Name $name -ErrorAction SilentlyContinue |
    ForEach-Object {
      "Stopping $($_.ProcessName) PID $($_.Id)" | Add-Content -Encoding utf8 (Join-Path $LogDir "orbbec_reset.log")
      Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

Remove-Item -LiteralPath (Join-Path $LiveDir "color.png") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $LiveDir "depth.png") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $LiveDir "frame.json") -Force -ErrorAction SilentlyContinue

$out = Join-Path $LogDir "orbbec_commonusages_diag.out.txt"
$err = Join-Path $LogDir "orbbec_commonusages_diag.err.txt"
Remove-Item $out, $err -Force -ErrorAction SilentlyContinue

$common = Join-Path $SdkBin "CommonUsages.exe"
if (Test-Path $common) {
  "q" | & $common > $out 2> $err
}
else {
  "CommonUsages.exe not found: $common" | Set-Content -Encoding utf8 $err
}

$summary = Join-Path $LogDir "orbbec_reset_summary.txt"
$outText = if (Test-Path $out) { Get-Content $out -Raw } else { "" }
$errText = if (Test-Path $err) { Get-Content $err -Raw } else { "" }

$status = "unknown"
if ($outText -match "depth profile:" -and $outText -match "color profile:") {
  $status = "ok: color/depth profiles available"
}
elseif ($outText -match "0x80070005" -or $outText -match "拒绝访问") {
  $status = "blocked: Windows denied camera/device access or another app owns the device"
}
elseif ($outText -match "No required type sensor found! sensorType: OB_SENSOR_DEPTH" -or $errText -match "No required type sensor found! sensorType: OB_SENSOR_DEPTH") {
  $status = "blocked: Orbbec depth sensor is not available"
}

@"
status: $status
time: $(Get-Date -Format o)
out: $out
err: $err

Recommended if blocked:
1. Close Camera/browser/other Orbbec viewer apps.
2. In the GUI, Stop All, then Reset Orbbec again.
3. If still blocked, unplug and replug the Orbbec USB cable.
4. Check Windows Settings > Privacy & security > Camera > let desktop apps access camera.
"@ | Set-Content -Encoding utf8 $summary

Get-Content $summary
