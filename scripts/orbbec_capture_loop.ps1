param(
  [string]$SdkBin = "C:\Users\lenovo\Desktop\JXCX\OrbbecSDK_v1.10.16_win_x64\OrbbecSDK_v1.10.16\Example\bin",
  [string]$OutDir = "C:\Users\lenovo\Desktop\JXCX\FoundationPose\live_orbbec",
  [int]$IntervalMs = 800,
  [int]$TimeoutSec = 15,
  [int]$MaxInvalidDepthBeforePause = 5
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Stop-OrbbecSaveToDisk {
  Get-Process -Name "SaveToDisk" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Test-PngComplete {
  param([Parameter(Mandatory=$true)][string]$Path)
  if (-not (Test-Path $Path)) {
    return $false
  }
  $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
  if (-not $item -or $item.Length -lt 16) {
    return $false
  }
  try {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    try {
      if ($stream.Length -lt 16) {
        return $false
      }
      $stream.Seek(-12, [System.IO.SeekOrigin]::End) | Out-Null
      $tail = New-Object byte[] 12
      [void]$stream.Read($tail, 0, 12)
      $iend = [byte[]](0,0,0,0,73,69,78,68,174,66,96,130)
      for ($i = 0; $i -lt 12; $i++) {
        if ($tail[$i] -ne $iend[$i]) {
          return $false
        }
      }
      return $true
    }
    finally {
      $stream.Dispose()
    }
  }
  catch {
    return $false
  }
}

function Get-LatestCompletePng {
  param(
    [Parameter(Mandatory=$true)][string]$Directory,
    [Parameter(Mandatory=$true)][string]$Filter
  )
  $files = Get-ChildItem $Directory -Filter $Filter -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
  foreach ($file in $files) {
    if (Test-PngComplete -Path $file.FullName) {
      return $file
    }
  }
  return $null
}

function Copy-Atomic {
  param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Destination,
    [int]$Attempts = 5
  )
  $tmp = "$Destination.tmp"
  for ($i = 1; $i -le $Attempts; $i++) {
    try {
      if (-not (Test-PngComplete -Path $Source)) {
        Start-Sleep -Milliseconds 30
        continue
      }
      Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
      Copy-Item -LiteralPath $Source -Destination $tmp -Force -ErrorAction Stop
      Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
      Move-Item -LiteralPath $tmp -Destination $Destination -Force -ErrorAction Stop
      return $true
    }
    catch {
      Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
      Start-Sleep -Milliseconds (40 * $i)
    }
  }
  Write-Warning "Skipped unstable source frame: $Source"
  return $false
}

function Get-DepthStats {
  param([Parameter(Mandatory=$true)][string]$Path)
  $pythonExe = Join-Path $env:USERPROFILE ".conda\envs\posteval_dl\python.exe"
  if (-not (Test-Path $pythonExe)) {
    return @{ valid = -1; max = -1 }
  }
  try {
    $output = & $pythonExe -c "import cv2, numpy as np; d=cv2.imread(r'$Path',-1); print('-1 -1' if d is None else f'{int(np.count_nonzero(d))} {int(d.max())}')"
    $parts = "$output".Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    return @{ valid = [int]$parts[0]; max = [int]$parts[1] }
  }
  catch {
    return @{ valid = -1; max = -1 }
  }
}

function Write-AtomicText {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][string]$Text,
    [string]$Encoding = "utf8"
  )
  $tmp = "$Path.tmp"
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  $Text | Set-Content -Encoding $Encoding $tmp
  Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
  Move-Item -LiteralPath $tmp -Destination $Path -Force
}

$kPath = Join-Path $OutDir "cam_K.txt"
Write-AtomicText -Path $kPath -Encoding ascii -Text @"
357.028 0 321.213
0 357.028 181.147
0 0 1
"@

$saveExe = Join-Path $SdkBin "SaveToDisk.exe"
if (-not (Test-Path $saveExe)) {
  throw "SaveToDisk.exe not found: $saveExe"
}

Stop-OrbbecSaveToDisk
Write-Host "Writing live frames to $OutDir"
Write-Host "Press Ctrl+C to stop."

$frameIndex = 0
$invalidDepthCount = 0
while ($true) {
  Stop-OrbbecSaveToDisk
  Get-ChildItem $SdkBin -Filter "Color_*.png" -ErrorAction SilentlyContinue | Remove-Item -Force
  Get-ChildItem $SdkBin -Filter "Depth_*.png" -ErrorAction SilentlyContinue | Remove-Item -Force

  $stdout = Join-Path $OutDir "last_savetodisk.out.txt"
  $stderr = Join-Path $OutDir "last_savetodisk.err.txt"
  Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue

  $p = Start-Process -FilePath $saveExe `
    -WorkingDirectory $SdkBin `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    $color = Get-LatestCompletePng -Directory $SdkBin -Filter "Color_*.png"
    $depth = Get-LatestCompletePng -Directory $SdkBin -Filter "Depth_*.png"

    if ($color -and $depth) {
      break
    }
    Start-Sleep -Milliseconds 100
  }

  if (-not $p.HasExited) {
    Stop-Process -Id $p.Id -Force
  }

  $color = Get-LatestCompletePng -Directory $SdkBin -Filter "Color_*.png"
  $depth = Get-LatestCompletePng -Directory $SdkBin -Filter "Depth_*.png"

  if ($color) {
    $colorOut = Join-Path $OutDir "color.png"
    $depthOut = Join-Path $OutDir "depth.png"

    $depthValid = 0
    $depthMax = 0
    $depthSourceName = "none"

    if ($depth) {
      $depthStats = Get-DepthStats -Path $depth.FullName
      $depthValid = $depthStats.valid
      $depthMax = $depthStats.max
      $depthSourceName = $depth.Name

      if ($depthValid -eq 0) {
        $invalidDepthCount += 1
        Write-Warning "Depth all zero (count=$invalidDepthCount), publishing color only. source=$($depth.Name)"
        if ($invalidDepthCount -ge $MaxInvalidDepthBeforePause) {
          Write-Warning "Depth has been invalid for $invalidDepthCount captures. Pausing to let depth stream recover."
          Stop-OrbbecSaveToDisk
          Start-Sleep -Seconds 3
        }
        # still publish color even when depth is bad
      }
      elseif ($depthValid -gt 0) {
        $invalidDepthCount = 0
        # publish depth too
        $copiedDepth = Copy-Atomic -Source $depth.FullName -Destination $depthOut
        if (-not $copiedDepth) {
          Write-Warning "Failed to copy depth frame; publishing color only."
        }
      }
    }
    else {
      Write-Warning "No depth frame captured; publishing color only."
    }

    $copiedColor = Copy-Atomic -Source $color.FullName -Destination $colorOut
    if (-not $copiedColor) {
      Write-Warning "Failed to copy color frame; skipping frame."
      Start-Sleep -Milliseconds $IntervalMs
      continue
    }

    $frameIndex += 1
    $meta = [ordered]@{
      frame_index = $frameIndex
      color_source = $color.Name
      depth_source = $depthSourceName
      updated_at = (Get-Date).ToString("o")
      depth_unit = "millimeter_uint16"
      width = 640
      height = 360
      depth_valid_pixels = $depthValid
      depth_max_mm = $depthMax
    } | ConvertTo-Json
    Write-AtomicText -Path (Join-Path $OutDir "frame.json") -Encoding utf8 -Text $meta
    Write-Host "Frame $frameIndex -> color.png (depth_valid=$depthValid max_mm=$depthMax)"
  }
  else {
    Write-Warning "No complete color frame captured; see $stdout and $stderr"
  }

  Start-Sleep -Milliseconds $IntervalMs
}
