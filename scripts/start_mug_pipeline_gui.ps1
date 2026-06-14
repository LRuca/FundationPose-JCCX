param(
  [string]$ProjectRoot = "C:\Users\lenovo\Desktop\JXCX"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase
Add-Type -AssemblyName System.Windows.Forms

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$LogDir = Join-Path $ProjectRoot "logs\mug_pipeline"
$LiveDir = Join-Path $ProjectRoot "FoundationPose\live_orbbec"
$DebugDir = Join-Path $ProjectRoot "FoundationPose\debug_orbbec_mug"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$StartupErrorLog = Join-Path $LogDir "gui_startup_error.log"
$PreviewErrorLog = Join-Path $LogDir "preview_error.log"
trap {
  New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
  $_ | Out-String | Set-Content -Encoding utf8 $StartupErrorLog
  [System.Windows.MessageBox]::Show($_.Exception.Message, "Mug pipeline GUI startup error") | Out-Null
  break
}

$script:Processes = @{}
$script:LogFiles = @{
  orbbec = Join-Path $LogDir "orbbec_capture.log"
  sam = Join-Path $LogDir "sam_mask.log"
  yolo = Join-Path $LogDir "yolo_bridge.log"
  pose = Join-Path $LogDir "foundationpose_mug.log"
}
$script:BaseLogFiles = $script:LogFiles.Clone()
$script:ChildProcessIds = @{}
$script:PreviewTimers = @()
$script:PreviewWindows = @()

function Clear-StartupLogs {
  Remove-Item -LiteralPath (Join-Path $LogDir "preview_error.log") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "gui_startup_error.log") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "orbbec_capture.log") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "orbbec_capture.log.err") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "sam_mask.log") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "sam_mask.log.err") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "yolo_bridge.log") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "yolo_bridge.log.err") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "foundationpose_mug.log") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LogDir "foundationpose_mug.log.err") -Force -ErrorAction SilentlyContinue
}
Clear-StartupLogs

function Convert-ToWslPath([string]$Path) {
  $full = [System.IO.Path]::GetFullPath($Path)
  $drive = $full.Substring(0,1).ToLowerInvariant()
  $rest = $full.Substring(2).Replace('\','/')
  return "/mnt/$drive$rest"
}

function Convert-ToBashSingleQuoted([string]$Text) {
  return "'" + $Text.Replace("'", "'\''") + "'"
}

function Start-ManagedProcess {
  param(
    [string]$Key,
    [string]$FilePath,
    [string]$Arguments,
    [string]$WorkingDirectory,
    [string]$LogFile
  )

  if ($script:Processes.ContainsKey($Key)) {
    $existing = $script:Processes[$Key]
    if ($existing -and -not $existing.HasExited) {
      return "Already running: $Key (PID $($existing.Id))"
    }
  }

  $baseName = [System.IO.Path]::GetFileNameWithoutExtension($LogFile)
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
  $runLogFile = Join-Path ([System.IO.Path]::GetDirectoryName($LogFile)) "$baseName`_$stamp.log"
  $script:LogFiles[$Key] = $runLogFile

  New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($runLogFile)) | Out-Null
  $errFile = "$runLogFile.err"
  "[$(Get-Date -Format o)] START $Key" | Set-Content -Encoding utf8 $runLogFile
  "Command: $FilePath $Arguments" | Add-Content -Encoding utf8 $runLogFile
  "[$(Get-Date -Format o)] STDERR $Key" | Set-Content -Encoding utf8 $errFile

  $p = Start-Process -FilePath $FilePath `
    -ArgumentList $Arguments `
    -WorkingDirectory $WorkingDirectory `
    -RedirectStandardOutput $runLogFile `
    -RedirectStandardError $errFile `
    -PassThru `
    -WindowStyle Hidden

  $script:Processes[$Key] = $p
  $script:ChildProcessIds[$Key] = $p.Id
  return "Started: $Key (PID $($p.Id))"
}

function Stop-ManagedProcess {
  param([string]$Key)
  if (-not $script:Processes.ContainsKey($Key)) {
    return "Not started: $Key"
  }
  $p = $script:Processes[$Key]
  if (-not $p -or $p.HasExited) {
    return "Not running: $Key"
  }
  Stop-Process -Id $p.Id -Force
  return "Stopped: $Key"
}

function Stop-OrbbecChildren {
  foreach ($name in @("SaveToDisk", "CommonUsages", "DepthViewer", "depth_viewer", "ColorViewer", "AlignFilterViewer", "SyncAlignViewer", "PointCloud")) {
    Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  }
  Get-CimInstance Win32_Process -Filter "name = 'powershell.exe' or name = 'pwsh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*orbbec_capture_loop.ps1*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Stop-KnownPipelineProcesses {
  foreach ($key in @("pose","sam","yolo","orbbec")) {
    Stop-ManagedProcess $key | Out-Null
  }
  foreach ($childPid in @($script:ChildProcessIds.Values)) {
    if ($childPid) {
      Stop-Process -Id $childPid -Force -ErrorAction SilentlyContinue
    }
  }
  Stop-OrbbecChildren
}

function Clear-RunArtifacts {
  Clear-StartupLogs
  Remove-Item -LiteralPath (Join-Path $LiveDir "mask_yolo.png") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LiveDir "mask_yolo.json") -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath (Join-Path $LiveDir "mask_sam_preview.png") -Force -ErrorAction SilentlyContinue
}

function Start-OrbbecProcess {
  Stop-OrbbecChildren
  return Start-ManagedProcess `
    -Key "orbbec" `
    -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$ProjectRoot\scripts\orbbec_capture_loop.ps1`" -IntervalMs 800 -MaxInvalidDepthBeforePause 5" `
    -WorkingDirectory $ProjectRoot `
    -LogFile $script:BaseLogFiles.orbbec
}

function Start-YoloProcess {
  $modelPath = Get-SelectedYoloModelPath
  $className = ($YoloClassName.Text.Trim())
  if (-not $className) { throw "Set a YOLO label/class name before starting YOLO." }
  $pythonExe = Join-Path $env:USERPROFILE ".conda\envs\posteval_dl\python.exe"
  if (-not (Test-Path $pythonExe)) {
    throw "Python environment for YOLO not found: $pythonExe"
  }
  $scriptPath = Join-Path $ProjectRoot "scripts\yolo_mug_mask_bridge.py"
  $imagePath = Join-Path $LiveDir "color.png"
  $maskPath = Join-Path $LiveDir "mask_yolo.png"
  $metaPath = Join-Path $LiveDir "mask_yolo.json"
  $logPath = Join-Path $LiveDir "yolo_mug_mask_bridge.log"
  $args = @(
    "`"$scriptPath`"",
    "--loop",
    "--image", "`"$imagePath`"",
    "--mask", "`"$maskPath`"",
    "--meta", "`"$metaPath`"",
    "--model", "`"$modelPath`"",
    "--class_name", "`"$className`"",
    "--conf", "0.25",
    "--imgsz", "960",
    "--min_area", "80",
    "--device", "0",
    "--log", "`"$logPath`""
  ) -join " "
  return Start-ManagedProcess `
    -Key "yolo" `
    -FilePath $pythonExe `
    -Arguments $args `
    -WorkingDirectory $ProjectRoot `
    -LogFile $script:BaseLogFiles.yolo
}

function Start-SamMaskProcess {
  param([string]$Bbox)
  if (-not $Bbox) {
    throw "Draw a bbox on the color image first."
  }
  $wslRoot = Convert-ToWslPath $ProjectRoot
  $cmd = "cd $wslRoot && /opt/conda/envs/yolo_mugseg/bin/python scripts/sam_first_frame_mask.py --image FoundationPose/live_orbbec/color.png --bbox '$Bbox' --mask FoundationPose/live_orbbec/mask_yolo.png --meta FoundationPose/live_orbbec/mask_yolo.json --preview FoundationPose/live_orbbec/mask_sam_preview.png --checkpoint downloads/sam/sam_vit_b_01ec64.pth --model_type vit_b --device cuda --min_area 80"
  return Start-ManagedProcess `
    -Key "sam" `
    -FilePath "$env:SystemRoot\System32\wsl.exe" `
    -Arguments "-d Ubuntu -- bash -lc `"$cmd`"" `
    -WorkingDirectory $ProjectRoot `
    -LogFile $script:BaseLogFiles.sam
}

function Get-ProcessStatusText {
  $lines = @()
  foreach ($key in @("orbbec","sam","yolo","pose")) {
    $status = "stopped"
    if ($script:Processes.ContainsKey($key)) {
      $p = $script:Processes[$key]
      if ($p -and -not $p.HasExited) {
        $status = "running PID $($p.Id)"
      }
      elseif ($p) {
        $status = "exited code $($p.ExitCode)"
      }
    }
    $lines += "${key}: $status"
  }

  $frame = Join-Path $LiveDir "frame.json"
  $mask = Join-Path $LiveDir "mask_yolo.png"
  $poseFiles = Join-Path $DebugDir "ob_in_cam"

  if (Test-Path $frame) {
    $lines += "frame.json: $((Get-Item $frame).LastWriteTime.ToString('HH:mm:ss'))"
  }
  else {
    $lines += "frame.json: missing"
  }

  if (Test-Path $mask) {
    $lines += "mask_yolo.png: $((Get-Item $mask).LastWriteTime.ToString('HH:mm:ss'))"
  }
  else {
    $lines += "mask_yolo.png: missing"
  }

  $depth = Join-Path $LiveDir "depth.png"
  if (Test-Path $depth) {
    try {
      $pythonExe = Join-Path $env:USERPROFILE ".conda\envs\posteval_dl\python.exe"
      if (Test-Path $pythonExe) {
        $depthStats = & $pythonExe -c "import cv2, numpy as np; d=cv2.imread(r'$depth',-1); print('missing' if d is None else f'valid={int(np.count_nonzero(d))} max_mm={int(d.max())}')"
        $lines += "depth.png: $depthStats"
      }
      else {
        $lines += "depth.png: $((Get-Item $depth).LastWriteTime.ToString('HH:mm:ss'))"
      }
    }
    catch {
      $lines += "depth.png: unreadable"
    }
  }
  else {
    $lines += "depth.png: missing"
  }

  if (Test-Path $poseFiles) {
    $count = (Get-ChildItem $poseFiles -Filter *.txt -ErrorAction SilentlyContinue | Measure-Object).Count
    $lines += "pose txt files: $count"
  }
  else {
    $lines += "pose txt files: 0"
  }

  return ($lines -join "`n")
}

function Test-LiveDepthReady {
  $depth = Join-Path $LiveDir "depth.png"
  if (-not (Test-Path $depth)) {
    throw "depth.png is missing. Start Orbbec and wait for live frames before starting Pose Tracker."
  }
  $pythonExe = Join-Path $env:USERPROFILE ".conda\envs\posteval_dl\python.exe"
  if (-not (Test-Path $pythonExe)) {
    return
  }
  $validText = & $pythonExe -c "import cv2, numpy as np; d=cv2.imread(r'$depth',-1); print(-1 if d is None else int(np.count_nonzero(d)))"
  $valid = [int]$validText
  if ($valid -le 0) {
    throw "depth.png has 0 valid depth pixels. FoundationPose needs metric depth. Stop All, use Reset Orbbec, then Start Orbbec again; also make sure the target is within the camera depth range and the depth sensor is not covered."
  }
}

function Read-RecentLogs {
  $chunks = @()
  foreach ($key in @("orbbec","sam","yolo","pose")) {
    $file = $script:LogFiles[$key]
    $chunks += "===== $key ====="
    if (Test-Path $file) {
      $chunks += (Get-Content $file -Tail 35 -ErrorAction SilentlyContinue)
    }
    else {
      $chunks += "no log yet"
    }
    $errFile = "$file.err"
    if (Test-Path $errFile) {
      $errLines = Get-Content $errFile -Tail 20 -ErrorAction SilentlyContinue
      if ($errLines.Count -gt 1) {
        $chunks += "----- ${key} stderr -----"
        $chunks += $errLines
      }
    }
  }
  return ($chunks -join "`n")
}

function Get-LatestTrackVis {
  $visDir = Join-Path $DebugDir "track_vis"
  if (-not (Test-Path $visDir)) {
    return $null
  }
  $latest = Get-ChildItem $visDir -Filter *.png -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if ($latest) {
    return $latest.FullName
  }
  return $null
}

function Load-BitmapNoLock {
  param([string]$Path)
  if (-not $Path -or -not (Test-Path $Path)) {
    return $null
  }
  try {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $stream = New-Object System.IO.MemoryStream(,$bytes)
    $bitmap = New-Object System.Windows.Media.Imaging.BitmapImage
    $bitmap.BeginInit()
    $bitmap.CacheOption = [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad
    $bitmap.StreamSource = $stream
    $bitmap.EndInit()
    $bitmap.Freeze()
    $stream.Dispose()
    return $bitmap
  }
  catch {
    return $null
  }
}

function Load-BitmapFresh {
  param([string]$Path)
  if (-not $Path -or -not (Test-Path $Path)) {
    return $null
  }
  try {
    $full = [System.IO.Path]::GetFullPath($Path)
    $stamp = [System.DateTimeOffset](Get-Item $full).LastWriteTimeUtc
    $uriText = "file:///$($full.Replace('\','/'))?v=$($stamp.ToUnixTimeMilliseconds())"
    $bitmap = New-Object System.Windows.Media.Imaging.BitmapImage
    $bitmap.BeginInit()
    $bitmap.CacheOption = [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad
    $bitmap.CreateOptions = [System.Windows.Media.Imaging.BitmapCreateOptions]::IgnoreImageCache
    $bitmap.UriSource = New-Object System.Uri($uriText)
    $bitmap.EndInit()
    $bitmap.Freeze()
    return $bitmap
  }
  catch {
    return Load-BitmapNoLock $Path
  }
}

function Update-PreviewWindows {
  foreach ($previewState in @($script:PreviewWindows)) {
    try {
      if (-not $previewState.window -or -not $previewState.window.IsVisible) {
        continue
      }
      $previewState.tick = [int]$previewState.tick + 1
      $colorPath = Join-Path $LiveDir "color.png"
      $maskPath = Join-Path $LiveDir "mask_yolo.png"
      $maskPreviewPath = Join-Path $LiveDir "mask_sam_preview.png"
      $posePath = Get-LatestTrackVis

      $colorBitmap = Load-BitmapFresh $colorPath
      if ($colorBitmap -and $previewState.images.color) { $previewState.images.color.Source = $colorBitmap }

      $maskDisplayPath = if (Test-Path $maskPreviewPath) { $maskPreviewPath } else { $maskPath }
      $maskBitmap = Load-BitmapFresh $maskDisplayPath
      if ($maskBitmap -and $previewState.images.mask) { $previewState.images.mask.Source = $maskBitmap }

      $poseBitmap = Load-BitmapFresh $posePath
      if ($poseBitmap -and $previewState.images.pose) { $previewState.images.pose.Source = $poseBitmap }

      $previewState.status.Text = "tick: $($previewState.tick) | color: $(if(Test-Path $colorPath){(Get-Item $colorPath).LastWriteTime.ToString('HH:mm:ss.fff')}else{'missing'}) | mask: $(if(Test-Path $maskPath){(Get-Item $maskPath).LastWriteTime.ToString('HH:mm:ss.fff')}else{'missing'}) | pose: $(if($posePath){[System.IO.Path]::GetFileName($posePath)}else{'missing'})"
    }
    catch {
      $_ | Out-String | Set-Content -Encoding utf8 $PreviewErrorLog
    }
  }
}

function Show-PreviewWindow {
  $preview = New-Object System.Windows.Window
  $preview.Title = "Mug Pipeline Preview"
  $preview.Width = 1180
  $preview.Height = 760
  $preview.WindowStartupLocation = "CenterScreen"

  $root = New-Object System.Windows.Controls.Grid
  $root.Margin = New-Object System.Windows.Thickness(10)
  $row0 = New-Object System.Windows.Controls.RowDefinition
  $row0.Height = [System.Windows.GridLength]::Auto
  $row1 = New-Object System.Windows.Controls.RowDefinition
  $row1.Height = New-Object System.Windows.GridLength(1, [System.Windows.GridUnitType]::Star)
  $root.RowDefinitions.Add($row0)
  $root.RowDefinitions.Add($row1)

  $status = New-Object System.Windows.Controls.TextBlock
  $status.FontFamily = "Consolas"
  $status.FontSize = 13
  $status.Margin = New-Object System.Windows.Thickness(0,0,0,8)
  [System.Windows.Controls.Grid]::SetRow($status, 0)
  $root.Children.Add($status) | Out-Null

  $imageGrid = New-Object System.Windows.Controls.Grid
  foreach ($i in 0..2) {
    $col = New-Object System.Windows.Controls.ColumnDefinition
    $col.Width = New-Object System.Windows.GridLength(1, [System.Windows.GridUnitType]::Star)
    $imageGrid.ColumnDefinitions.Add($col)
  }
  [System.Windows.Controls.Grid]::SetRow($imageGrid, 1)
  $root.Children.Add($imageGrid) | Out-Null

  function New-PreviewPanel {
    param([string]$Header, [int]$Column)
    $image = New-Object System.Windows.Controls.Image
    $image.Stretch = [System.Windows.Media.Stretch]::Uniform
    $border = New-Object System.Windows.Controls.Border
    $border.Background = [System.Windows.Media.Brushes]::Black
    $border.Child = $image
    $group = New-Object System.Windows.Controls.GroupBox
    $group.Header = $Header
    $group.Margin = New-Object System.Windows.Thickness(0,0,8,0)
    $group.Content = $border
    [System.Windows.Controls.Grid]::SetColumn($group, $Column)
    $imageGrid.Children.Add($group) | Out-Null
    return $image
  }

  $previewImages = @{
    color = New-PreviewPanel -Header "Color" -Column 0
    mask = New-PreviewPanel -Header "Mask" -Column 1
    pose = New-PreviewPanel -Header "Pose Visualization" -Column 2
  }
  $preview.Content = $root
  $previewState = [pscustomobject]@{
    window = $preview
    status = $status
    images = $previewImages
    tick = 0
  }
  $script:PreviewWindows += $previewState
  $preview.Show() | Out-Null
  Update-PreviewWindows
}

function Convert-CanvasPointToImagePixel {
  param(
    [System.Windows.Controls.Image]$Image,
    [System.Windows.Point]$Point
  )
  $bitmap = $Image.Source
  if (-not $bitmap) { return $null }
  $imageW = [double]$bitmap.PixelWidth
  $imageH = [double]$bitmap.PixelHeight
  $viewW = [double]$Image.ActualWidth
  $viewH = [double]$Image.ActualHeight
  if ($imageW -le 0 -or $imageH -le 0 -or $viewW -le 0 -or $viewH -le 0) { return $null }
  $scale = [Math]::Min($viewW / $imageW, $viewH / $imageH)
  $drawW = $imageW * $scale
  $drawH = $imageH * $scale
  $offsetX = ($viewW - $drawW) / 2.0
  $offsetY = ($viewH - $drawH) / 2.0
  $x = ($Point.X - $offsetX) / $scale
  $y = ($Point.Y - $offsetY) / $scale
  $x = [Math]::Max(0, [Math]::Min($imageW - 1, $x))
  $y = [Math]::Max(0, [Math]::Min($imageH - 1, $y))
  return New-Object System.Windows.Point($x, $y)
}

function Show-SamBboxWindow {
  $colorPath = Join-Path $LiveDir "color.png"
  if (-not (Test-Path $colorPath)) {
    [System.Windows.MessageBox]::Show("Start Orbbec first and wait for color.png.", "SAM bbox") | Out-Null
    return
  }

  $samWindow = New-Object System.Windows.Window
  $samWindow.Title = "SAM First-frame BBox"
  $samWindow.Width = 1040
  $samWindow.Height = 760
  $samWindow.WindowStartupLocation = "CenterScreen"

  $root = New-Object System.Windows.Controls.Grid
  $root.Margin = New-Object System.Windows.Thickness(10)
  $row0 = New-Object System.Windows.Controls.RowDefinition
  $row0.Height = [System.Windows.GridLength]::Auto
  $row1 = New-Object System.Windows.Controls.RowDefinition
  $row1.Height = New-Object System.Windows.GridLength(1, [System.Windows.GridUnitType]::Star)
  $root.RowDefinitions.Add($row0)
  $root.RowDefinitions.Add($row1)

  $toolbar = New-Object System.Windows.Controls.WrapPanel
  $toolbar.Margin = New-Object System.Windows.Thickness(0,0,0,8)
  [System.Windows.Controls.Grid]::SetRow($toolbar, 0)
  $root.Children.Add($toolbar) | Out-Null

  $bboxText = New-Object System.Windows.Controls.TextBox
  $bboxText.Width = 220
  $bboxText.Height = 30
  $bboxText.Margin = New-Object System.Windows.Thickness(0,0,8,0)
  $bboxText.FontFamily = "Consolas"
  $bboxText.IsReadOnly = $true
  $toolbar.Children.Add($bboxText) | Out-Null

  $refreshBtn = New-Object System.Windows.Controls.Button
  $refreshBtn.Width = 110
  $refreshBtn.Height = 30
  $refreshBtn.Margin = New-Object System.Windows.Thickness(0,0,8,0)
  $refreshBtn.Content = "Refresh Frame"
  $toolbar.Children.Add($refreshBtn) | Out-Null

  $clearBtn = New-Object System.Windows.Controls.Button
  $clearBtn.Width = 90
  $clearBtn.Height = 30
  $clearBtn.Margin = New-Object System.Windows.Thickness(0,0,8,0)
  $clearBtn.Content = "Clear"
  $toolbar.Children.Add($clearBtn) | Out-Null

  $generateBtn = New-Object System.Windows.Controls.Button
  $generateBtn.Width = 150
  $generateBtn.Height = 30
  $generateBtn.Margin = New-Object System.Windows.Thickness(0,0,8,0)
  $generateBtn.Content = "Generate SAM Mask"
  $toolbar.Children.Add($generateBtn) | Out-Null

  $status = New-Object System.Windows.Controls.TextBlock
  $status.Margin = New-Object System.Windows.Thickness(8,6,0,0)
  $status.Text = "Drag on the image to select the object."
  $toolbar.Children.Add($status) | Out-Null
  $script:SamBboxStatus = $status

  $imageHost = New-Object System.Windows.Controls.Grid
  $imageHost.Background = [System.Windows.Media.Brushes]::Black
  [System.Windows.Controls.Grid]::SetRow($imageHost, 1)
  $root.Children.Add($imageHost) | Out-Null

  $image = New-Object System.Windows.Controls.Image
  $image.Stretch = [System.Windows.Media.Stretch]::Uniform
  $imageHost.Children.Add($image) | Out-Null
  $script:SamBboxImage = $image

  $canvas = New-Object System.Windows.Controls.Canvas
  $canvas.Background = [System.Windows.Media.Brushes]::Transparent
  $imageHost.Children.Add($canvas) | Out-Null
  $script:SamBboxCanvas = $canvas

  $rect = New-Object System.Windows.Shapes.Rectangle
  $rect.Stroke = [System.Windows.Media.Brushes]::Lime
  $rect.StrokeThickness = 2
  $rect.Fill = New-Object System.Windows.Media.SolidColorBrush([System.Windows.Media.Color]::FromArgb(45, 0, 255, 0))
  $rect.Visibility = [System.Windows.Visibility]::Collapsed
  $canvas.Children.Add($rect) | Out-Null
  $script:SamBboxRect = $rect
  $script:SamBboxText = $bboxText

  $script:SamBboxState = @{
    dragging = $false
    start = $null
    bbox = $null
  }

  function Load-SamFrame {
    $bitmap = Load-BitmapFresh $colorPath
    if ($bitmap) {
      $script:SamBboxImage.Source = $bitmap
      $script:SamBboxStatus.Text = "Frame: $((Get-Item $colorPath).LastWriteTime.ToString('HH:mm:ss.fff'))"
    }
  }

  function script:Set-SamCanvasRect {
    param([System.Windows.Point]$A, [System.Windows.Point]$B)
    $left = [Math]::Min($A.X, $B.X)
    $top = [Math]::Min($A.Y, $B.Y)
    $width = [Math]::Abs($A.X - $B.X)
    $height = [Math]::Abs($A.Y - $B.Y)
    [System.Windows.Controls.Canvas]::SetLeft($script:SamBboxRect, $left)
    [System.Windows.Controls.Canvas]::SetTop($script:SamBboxRect, $top)
    $script:SamBboxRect.Width = $width
    $script:SamBboxRect.Height = $height
    $script:SamBboxRect.Visibility = [System.Windows.Visibility]::Visible
  }

  function script:Update-SamBboxFromPoints {
    param([System.Windows.Point]$A, [System.Windows.Point]$B)
    if (-not $script:SamBboxImage) { return }
    $p1 = Convert-CanvasPointToImagePixel -Image $script:SamBboxImage -Point $A
    $p2 = Convert-CanvasPointToImagePixel -Image $script:SamBboxImage -Point $B
    if (-not $p1 -or -not $p2) { return }
    $x1 = [int][Math]::Round([Math]::Min($p1.X, $p2.X))
    $y1 = [int][Math]::Round([Math]::Min($p1.Y, $p2.Y))
    $x2 = [int][Math]::Round([Math]::Max($p1.X, $p2.X))
    $y2 = [int][Math]::Round([Math]::Max($p1.Y, $p2.Y))
    if (($x2 - $x1) -lt 2 -or ($y2 - $y1) -lt 2) { return }
    $script:SamBboxState["bbox"] = "$x1,$y1,$x2,$y2"
    $script:SamBboxText.Text = $script:SamBboxState["bbox"]
  }

  $canvas.Add_MouseLeftButtonDown({
    param($sender, $eventArgs)
    if (-not $sender) { return }
    $script:SamBboxState["dragging"] = $true
    $script:SamBboxState["start"] = $eventArgs.GetPosition($sender)
    $sender.CaptureMouse() | Out-Null
    Set-SamCanvasRect -A $script:SamBboxState["start"] -B $script:SamBboxState["start"]
  })
  $canvas.Add_MouseMove({
    param($sender, $eventArgs)
    if (-not $sender) { return }
    if ($script:SamBboxState -and $script:SamBboxState["dragging"] -and $script:SamBboxState["start"]) {
      $current = $eventArgs.GetPosition($sender)
      Set-SamCanvasRect -A $script:SamBboxState["start"] -B $current
      Update-SamBboxFromPoints -A $script:SamBboxState["start"] -B $current
    }
  })
  $canvas.Add_MouseLeftButtonUp({
    param($sender, $eventArgs)
    if (-not $sender) { return }
    if ($script:SamBboxState -and $script:SamBboxState["dragging"] -and $script:SamBboxState["start"]) {
      $current = $eventArgs.GetPosition($sender)
      Set-SamCanvasRect -A $script:SamBboxState["start"] -B $current
      Update-SamBboxFromPoints -A $script:SamBboxState["start"] -B $current
    }
    if ($script:SamBboxState) {
      $script:SamBboxState["dragging"] = $false
    }
    $sender.ReleaseMouseCapture()
  })

  $refreshBtn.Add_Click({
    Load-SamFrame
  })
  $clearBtn.Add_Click({
    $script:SamBboxState["bbox"] = $null
    $script:SamBboxText.Text = ""
    $script:SamBboxRect.Visibility = [System.Windows.Visibility]::Collapsed
  })
  $generateBtn.Add_Click({
    try {
      $msg = Start-SamMaskProcess -Bbox $script:SamBboxState["bbox"]
      [System.Windows.MessageBox]::Show($msg, "SAM mask") | Out-Null
      $script:SamBboxStatus.Text = "SAM process started. Watch mask_yolo.png in Preview."
    }
    catch {
      [System.Windows.MessageBox]::Show($_.Exception.Message, "SAM mask error") | Out-Null
    }
    Update-Ui
  })

  $samWindow.Content = $root
  Load-SamFrame
  $samWindow.Show() | Out-Null
}

$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Object Pose Pipeline" Width="1080" Height="740" WindowStartupLocation="CenterScreen">
  <Grid Margin="12">
    <Grid.RowDefinitions>
      <RowDefinition Height="Auto"/>
      <RowDefinition Height="Auto"/>
      <RowDefinition Height="*"/>
      <RowDefinition Height="Auto"/>
    </Grid.RowDefinitions>

    <TextBlock Grid.Row="0" FontSize="18" FontWeight="SemiBold" Text="Object SAM First-frame Mask + FoundationPose"/>

    <StackPanel Grid.Row="1" Margin="0,12,0,12">
      <Grid Margin="0,0,0,8">
        <Grid.ColumnDefinitions>
          <ColumnDefinition Width="Auto"/>
          <ColumnDefinition Width="220"/>
          <ColumnDefinition Width="*"/>
          <ColumnDefinition Width="90"/>
        </Grid.ColumnDefinitions>
        <TextBlock Grid.Column="0" Width="70" VerticalAlignment="Center" Text="Mesh"/>
        <ComboBox Name="MeshPreset" Grid.Column="1" Height="30" Margin="0,0,8,0"/>
        <TextBox Name="MeshPath" Grid.Column="2" Height="30" Margin="0,0,8,0" FontFamily="Consolas"/>
        <Button Name="BrowseMesh" Grid.Column="3" Height="30" Content="Browse"/>
      </Grid>
      <Grid Margin="0,0,0,8">
        <Grid.ColumnDefinitions>
          <ColumnDefinition Width="Auto"/>
          <ColumnDefinition Width="210"/>
          <ColumnDefinition Width="*"/>
          <ColumnDefinition Width="90"/>
          <ColumnDefinition Width="Auto"/>
          <ColumnDefinition Width="140"/>
        </Grid.ColumnDefinitions>
        <TextBlock Grid.Column="0" Width="70" VerticalAlignment="Center" Text="YOLO"/>
        <ComboBox Name="YoloPreset" Grid.Column="1" Height="30" Margin="0,0,8,0"/>
        <TextBox Name="YoloModelPath" Grid.Column="2" Height="30" Margin="0,0,8,0" FontFamily="Consolas"/>
        <Button Name="BrowseYoloModel" Grid.Column="3" Height="30" Margin="0,0,8,0" Content="Browse"/>
        <TextBlock Grid.Column="4" VerticalAlignment="Center" Margin="8,0,6,0" Text="Label"/>
        <ComboBox Name="YoloClassName" Grid.Column="5" Height="30" IsEditable="True" Text="needle"/>
      </Grid>
      <WrapPanel>
        <Button Name="StartOrbbec" Width="140" Height="34" Margin="0,0,8,8" Content="Start Orbbec"/>
        <Button Name="OpenSamBBox" Width="150" Height="34" Margin="0,0,8,8" Content="SAM BBox Mask"/>
        <Button Name="StartYolo" Width="140" Height="34" Margin="0,0,8,8" Content="Start YOLO"/>
        <Button Name="StartPose" Width="160" Height="34" Margin="0,0,8,8" Content="Start Pose Tracker"/>
        <Button Name="StopAll" Width="120" Height="34" Margin="20,0,8,8" Content="Stop All"/>
        <Button Name="RestartAll" Width="150" Height="34" Margin="0,0,8,8" Content="Restart Capture+YOLO"/>
        <Button Name="ResetOrbbec" Width="130" Height="34" Margin="0,0,8,8" Content="Reset Orbbec"/>
        <Button Name="Refresh" Width="100" Height="34" Margin="0,0,8,8" Content="Refresh"/>
        <Button Name="OpenLogs" Width="100" Height="34" Margin="0,0,8,8" Content="Open Logs"/>
        <Button Name="OpenPreview" Width="120" Height="34" Margin="0,0,8,8" Content="Open Preview"/>
      </WrapPanel>
    </StackPanel>

    <Grid Grid.Row="2">
      <Grid.ColumnDefinitions>
        <ColumnDefinition Width="280"/>
        <ColumnDefinition Width="*"/>
      </Grid.ColumnDefinitions>
      <TextBox Name="StatusBox" Grid.Column="0" Margin="0,0,10,0" FontFamily="Consolas" FontSize="13"
               IsReadOnly="True" TextWrapping="Wrap" VerticalScrollBarVisibility="Auto"/>
      <TextBox Name="LogBox" Grid.Column="1" FontFamily="Consolas" FontSize="12"
               IsReadOnly="True" TextWrapping="NoWrap" VerticalScrollBarVisibility="Auto"
               HorizontalScrollBarVisibility="Auto"/>
    </Grid>

    <TextBlock Grid.Row="3" Margin="0,10,0,0" TextWrapping="Wrap"
               Text="Run order: choose Mesh and YOLO preset, Start Orbbec, then use Start YOLO or SAM BBox Mask to write mask_yolo.png before starting Pose Tracker."/>
  </Grid>
</Window>
"@

$reader = New-Object System.Xml.XmlNodeReader ([xml]$xaml)
$window = [Windows.Markup.XamlReader]::Load($reader)

$MeshPreset = $window.FindName("MeshPreset")
$MeshPath = $window.FindName("MeshPath")
$BrowseMesh = $window.FindName("BrowseMesh")
$YoloPreset = $window.FindName("YoloPreset")
$YoloModelPath = $window.FindName("YoloModelPath")
$BrowseYoloModel = $window.FindName("BrowseYoloModel")
$YoloClassName = $window.FindName("YoloClassName")
$StartOrbbec = $window.FindName("StartOrbbec")
$OpenSamBBox = $window.FindName("OpenSamBBox")
$StartYolo = $window.FindName("StartYolo")
$StartPose = $window.FindName("StartPose")
$StopAll = $window.FindName("StopAll")
$RestartAll = $window.FindName("RestartAll")
$ResetOrbbec = $window.FindName("ResetOrbbec")
$Refresh = $window.FindName("Refresh")
$OpenLogs = $window.FindName("OpenLogs")
$OpenPreview = $window.FindName("OpenPreview")
$StatusBox = $window.FindName("StatusBox")
$LogBox = $window.FindName("LogBox")

$script:MeshPresets = @(
  [pscustomobject]@{
    Name = "Needle structured v3"
    Path = Join-Path $ProjectRoot "model\fixed_unnamed_object_3\needle_structured_tail_reconstruction_v3.stl"
  },
  [pscustomobject]@{
    Name = "Mug YCB default"
    Path = Join-Path $ProjectRoot "FoundationPose\demo_data\ycb_mug\google_16k\textured.obj"
  }
)
foreach ($preset in $script:MeshPresets) {
  $item = New-Object System.Windows.Controls.ComboBoxItem
  $item.Content = $preset.Name
  $item.Tag = $preset.Path
  $MeshPreset.Items.Add($item) | Out-Null
}
if ($MeshPreset.Items.Count -gt 0) {
  $MeshPreset.SelectedIndex = 0
  $MeshPath.Text = $MeshPreset.SelectedItem.Tag
}

foreach ($label in @("needle_inbox", "needle", "cup")) {
  $YoloClassName.Items.Add($label) | Out-Null
}

$script:YoloPresets = @(
  [pscustomobject]@{
    Name = "Needle inbox combined (NEW)"
    Path = Join-Path $ProjectRoot "runs\needle_inbox_seg\yolov8n_seg_combined\weights\best.pt"
    ClassName = "needle_inbox"
  },
  [pscustomobject]@{
    Name = "Needle inbox pure"
    Path = Join-Path $ProjectRoot "runs\needle_inbox_seg\yolov8n_seg_inbox\weights\best.pt"
    ClassName = "needle_inbox"
  },
  [pscustomobject]@{
    Name = "Needle LWT trained"
    Path = Join-Path $ProjectRoot "runs\needle_lwt_seg\yolov8n_seg_lwt\weights\best.pt"
    ClassName = "needle"
  },
  [pscustomobject]@{
    Name = "Needle old trained"
    Path = Join-Path $ProjectRoot "runs\needle_seg\yolov8n_needle\weights\best.pt"
    ClassName = "needle"
  },
  [pscustomobject]@{
    Name = "COCO cup fallback"
    Path = Join-Path $ProjectRoot "yolov8n-seg.pt"
    ClassName = "cup"
  }
)

foreach ($preset in $script:YoloPresets) {
  if (Test-Path $preset.Path) {
    $item = New-Object System.Windows.Controls.ComboBoxItem
    $item.Content = $preset.Name
    $item.Tag = $preset
    $YoloPreset.Items.Add($item) | Out-Null
  }
}

function Apply-YoloPreset {
  param($Preset)
  if (-not $Preset) { return }
  $YoloModelPath.Text = $Preset.Path
  $YoloClassName.Text = $Preset.ClassName
}

if ($YoloPreset.Items.Count -gt 0) {
  $YoloPreset.SelectedIndex = 0
  Apply-YoloPreset $YoloPreset.SelectedItem.Tag
}
else {
  $YoloModelPath.Text = Join-Path $ProjectRoot "yolov8n-seg.pt"
  $YoloClassName.Text = "cup"
}

function Get-SelectedMeshPath {
  $path = $MeshPath.Text.Trim()
  if (-not $path) {
    throw "Choose a mesh file before starting pose tracking."
  }
  $full = [System.IO.Path]::GetFullPath($path)
  if (-not (Test-Path $full)) {
    throw "Mesh file does not exist: $full"
  }
  return $full
}

function Get-SelectedYoloModelPath {
  $path = $YoloModelPath.Text.Trim()
  if (-not $path) {
    throw "Choose a YOLO segmentation model before starting YOLO."
  }
  $full = [System.IO.Path]::GetFullPath($path)
  if (-not (Test-Path $full)) {
    throw "YOLO model file does not exist: $full"
  }
  return $full
}

function Update-Ui {
  $StatusBox.Text = Get-ProcessStatusText
  $now = Get-Date
  if (-not $script:LastLogRefresh -or (($now - $script:LastLogRefresh).TotalSeconds -ge 1.0)) {
    $LogBox.Text = Read-RecentLogs
    $LogBox.ScrollToEnd()
    $script:LastLogRefresh = $now
  }
  Update-PreviewWindows
}

$StartOrbbec.Add_Click({
  try {
    $msg = Start-OrbbecProcess
    [System.Windows.MessageBox]::Show($msg, "Orbbec") | Out-Null
  }
  catch {
    [System.Windows.MessageBox]::Show($_.Exception.Message, "Orbbec error") | Out-Null
  }
  Update-Ui
})

$StartYolo.Add_Click({
  try {
    $msg = Start-YoloProcess
    [System.Windows.MessageBox]::Show($msg, "YOLO") | Out-Null
  }
  catch {
    [System.Windows.MessageBox]::Show($_.Exception.Message, "YOLO error") | Out-Null
  }
  Update-Ui
})

$MeshPreset.Add_SelectionChanged({
  if ($MeshPreset.SelectedItem -and $MeshPreset.SelectedItem.Tag) {
    $MeshPath.Text = $MeshPreset.SelectedItem.Tag
  }
})

$YoloPreset.Add_SelectionChanged({
  if ($YoloPreset.SelectedItem -and $YoloPreset.SelectedItem.Tag) {
    Apply-YoloPreset $YoloPreset.SelectedItem.Tag
  }
})

$BrowseMesh.Add_Click({
  $dialog = New-Object System.Windows.Forms.OpenFileDialog
  $dialog.Title = "Choose FoundationPose mesh"
  $dialog.Filter = "Mesh files (*.obj;*.stl;*.ply)|*.obj;*.stl;*.ply|All files (*.*)|*.*"
  $dialog.InitialDirectory = Join-Path $ProjectRoot "model"
  if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $MeshPath.Text = $dialog.FileName
    $MeshPreset.SelectedIndex = -1
  }
})

$BrowseYoloModel.Add_Click({
  $dialog = New-Object System.Windows.Forms.OpenFileDialog
  $dialog.Title = "Choose YOLO segmentation model"
  $dialog.Filter = "YOLO model (*.pt)|*.pt|All files (*.*)|*.*"
  $dialog.InitialDirectory = $ProjectRoot
  if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $YoloModelPath.Text = $dialog.FileName
    $YoloPreset.SelectedIndex = -1
  }
})

$OpenSamBBox.Add_Click({
  Show-SamBboxWindow
  Update-Ui
})

$StartPose.Add_Click({
  try {
    Test-LiveDepthReady
    $wslRoot = Convert-ToWslPath $ProjectRoot
    $meshFile = Get-SelectedMeshPath
    $meshWsl = Convert-ToWslPath $meshFile
    $meshArg = Convert-ToBashSingleQuoted $meshWsl
    $cmd = "cd $wslRoot/FoundationPose && /opt/conda/envs/foundationpose/bin/python run_orbbec_mug_live.py --mesh_file $meshArg --max_frames 50"
    $msg = Start-ManagedProcess `
      -Key "pose" `
      -FilePath "$env:SystemRoot\System32\wsl.exe" `
      -Arguments "-d Ubuntu -- bash -lc `"$cmd`"" `
      -WorkingDirectory $ProjectRoot `
      -LogFile $script:LogFiles.pose
    [System.Windows.MessageBox]::Show(($msg + "`nMesh: $meshFile"), "Pose Tracker") | Out-Null
  }
  catch {
    [System.Windows.MessageBox]::Show($_.Exception.Message, "Pose error") | Out-Null
  }
  Update-Ui
})

$StopAll.Add_Click({
  Stop-KnownPipelineProcesses
  Update-Ui
})

$RestartAll.Add_Click({
  try {
    Stop-KnownPipelineProcesses
    Clear-RunArtifacts
    Start-Sleep -Milliseconds 500
    $m1 = Start-OrbbecProcess
    Start-Sleep -Seconds 2
    $m2 = Start-YoloProcess
    [System.Windows.MessageBox]::Show(($m1 + "`n" + $m2 + "`nStart Pose Tracker manually after mask_yolo.png updates."), "Restart Capture+YOLO") | Out-Null
  }
  catch {
    [System.Windows.MessageBox]::Show($_.Exception.Message, "Restart error") | Out-Null
  }
  Update-Ui
})

$ResetOrbbec.Add_Click({
  try {
    Stop-KnownPipelineProcesses
    $diag = & "$ProjectRoot\scripts\orbbec_reset_and_diag.ps1" -ProjectRoot $ProjectRoot
    [System.Windows.MessageBox]::Show(($diag -join "`n"), "Orbbec reset diagnostic") | Out-Null
  }
  catch {
    [System.Windows.MessageBox]::Show($_.Exception.Message, "Reset Orbbec error") | Out-Null
  }
  Update-Ui
})

$Refresh.Add_Click({ Update-Ui })
$OpenLogs.Add_Click({ Start-Process explorer.exe $LogDir })
$OpenPreview.Add_Click({ Show-PreviewWindow })

$timer = New-Object Windows.Threading.DispatcherTimer
$timer.Interval = [TimeSpan]::FromMilliseconds(200)
$timer.Add_Tick({ Update-Ui })
$timer.Start()

$window.Add_Closing({
  Stop-KnownPipelineProcesses
})

Update-Ui
$window.ShowDialog() | Out-Null
