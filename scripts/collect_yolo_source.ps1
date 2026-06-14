param(
  [string]$LiveDir = "C:\Users\lenovo\Desktop\JXCX\FoundationPose\live_orbbec",
  [string]$OutDir = "C:\Users\lenovo\Desktop\JXCX\data\needle_raw",
  [int]$Count = 300,
  [int]$IntervalMs = 500,
  [string]$Prefix = "needle"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

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

function Copy-Atomic {
  param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Destination
  )
  $tmp = "$Destination.tmp"
  Copy-Item -LiteralPath $Source -Destination $tmp -Force
  Move-Item -LiteralPath $tmp -Destination $Destination -Force
}

$colorPath = Join-Path $LiveDir "color.png"
$framePath = Join-Path $LiveDir "frame.json"
$saved = 0
$lastFrameIndex = $null

Write-Host "Collecting YOLO source images from $colorPath"
Write-Host "Output directory: $OutDir"
Write-Host "Target count: $Count, interval: ${IntervalMs}ms"
Write-Host "Press Ctrl+C to stop early."

while ($saved -lt $Count) {
  if (-not (Test-PngComplete -Path $colorPath)) {
    Write-Warning "Waiting for a complete color.png in $LiveDir"
    Start-Sleep -Milliseconds $IntervalMs
    continue
  }

  $frameIndex = $null
  if (Test-Path $framePath) {
    try {
      $meta = Get-Content -LiteralPath $framePath -Raw | ConvertFrom-Json
      $frameIndex = [int]$meta.frame_index
    }
    catch {
      $frameIndex = $null
    }
  }

  if ($frameIndex -ne $null -and $frameIndex -eq $lastFrameIndex) {
    Start-Sleep -Milliseconds $IntervalMs
    continue
  }

  $saved += 1
  $lastFrameIndex = $frameIndex
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
  $framePart = if ($frameIndex -ne $null) { "{0:D6}" -f $frameIndex } else { "{0:D6}" -f $saved }
  $outFile = Join-Path $OutDir "$Prefix`_$stamp`_$framePart.png"
  Copy-Atomic -Source $colorPath -Destination $outFile
  Write-Host ("Saved {0}/{1}: {2}" -f $saved, $Count, $outFile)

  Start-Sleep -Milliseconds $IntervalMs
}

Write-Host "Done. Saved $saved images to $OutDir"
