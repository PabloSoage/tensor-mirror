# Screenshot of the Vivado main window (only that window, through PrintWindow), once a
# GUI script such as fpga/render_device.tcl has written its marker file. Vivado stays open.
# The README's device view is this screenshot cropped to the device.
#   tools/capture_vivado.ps1 <marker file> <out.png> [seconds to settle]
param([string]$Marker, [string]$Out, [int]$Settle = 20)
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class W {
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint f);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
}
'@
[W]::SetProcessDPIAware() | Out-Null  # physical pixels, or a scaled display crops the window
while (-not (Test-Path $Marker)) { Start-Sleep -Seconds 5 }
$p = $null
while (-not $p) {
    $p = Get-Process | Where-Object { $_.MainWindowTitle -like '*Vivado*' } | Select-Object -First 1
    if (-not $p) { Start-Sleep -Seconds 2 }
}
$h = $p.MainWindowHandle
[W]::ShowWindow($h, 3) | Out-Null  # maximised
Start-Sleep -Seconds $Settle
$r = New-Object W+RECT
[W]::GetWindowRect($h, [ref]$r) | Out-Null
$bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$dc = $g.GetHdc()
[W]::PrintWindow($h, $dc, 2) | Out-Null  # PW_RENDERFULLCONTENT
$g.ReleaseHdc($dc)
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
"captured $($p.Id) $($bmp.Width)x$($bmp.Height) -> $Out"
