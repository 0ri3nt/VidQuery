param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [int]$DelaySeconds = 3
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class VidQueryWindowCaptureNative
{
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int command);

    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder text, int count);
}
"@

$Process = Get-Process -Id $ProcessId -ErrorAction Stop
$Handle = [IntPtr]::Zero

for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    $Process.Refresh()
    $Handle = $Process.MainWindowHandle
    if ($Handle -ne [IntPtr]::Zero) {
        break
    }
    Start-Sleep -Milliseconds 250
}

if ($Handle -eq [IntPtr]::Zero) {
    # Windows Terminal may host the PowerShell process in a separate window.
    # The launched SQLite pane is foregrounded, so capture that host window.
    $Handle = [VidQueryWindowCaptureNative]::GetForegroundWindow()
    if ($Handle -eq [IntPtr]::Zero) {
        throw "Neither process $ProcessId nor its terminal host exposes a capturable window."
    }
}

[VidQueryWindowCaptureNative]::ShowWindow($Handle, 9) | Out-Null
[VidQueryWindowCaptureNative]::SetForegroundWindow($Handle) | Out-Null
Start-Sleep -Seconds $DelaySeconds

$Rectangle = New-Object VidQueryWindowCaptureNative+RECT
if (-not [VidQueryWindowCaptureNative]::GetWindowRect($Handle, [ref]$Rectangle)) {
    throw "Unable to read the window bounds for process $ProcessId."
}

$Width = $Rectangle.Right - $Rectangle.Left
$Height = $Rectangle.Bottom - $Rectangle.Top
if ($Width -le 0 -or $Height -le 0) {
    throw "The window bounds are invalid: ${Width}x${Height}."
}

$ResolvedOutputPath = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath
} else {
    Join-Path (Get-Location) $OutputPath
}
$OutputDirectory = Split-Path -Parent $ResolvedOutputPath
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$Bitmap = New-Object System.Drawing.Bitmap($Width, $Height)
$Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
$Graphics.CopyFromScreen(
    $Rectangle.Left,
    $Rectangle.Top,
    0,
    0,
    (New-Object System.Drawing.Size($Width, $Height)),
    [System.Drawing.CopyPixelOperation]::SourceCopy
)
$Bitmap.Save($ResolvedOutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$Graphics.Dispose()
$Bitmap.Dispose()

Write-Output "Captured actual process window: $ResolvedOutputPath"
