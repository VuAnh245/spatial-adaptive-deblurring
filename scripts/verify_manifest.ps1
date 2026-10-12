# Check extracted files against a MANIFEST.txt written by scripts/pack_results.py (Windows PowerShell 5.1).
#
#   powershell -ExecutionPolicy Bypass -File scripts\verify_manifest.ps1 -Manifest results\MANIFEST.txt -Root results
#
# MANIFEST lines: '#' header, then  relative/path<TAB>bytes<TAB>sha256.
# Prints one row per file (du / thieu / sai ma), then the totals; exit code 1 if any file is missing or wrong.
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$Root
)

$rows = @()
foreach ($line in Get-Content -LiteralPath $Manifest -Encoding UTF8) {
    if ($line -match '^\s*#' -or $line.Trim() -eq '') { continue }
    $parts = $line -split "`t"
    $rel, $size, $sha = $parts[0], [int64]$parts[1], $parts[2].ToLower()
    $path = Join-Path $Root ($rel -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path)) {
        $status = 'thieu'
    } else {
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
        if ($actual -eq $sha -and (Get-Item -LiteralPath $path).Length -eq $size) { $status = 'du' } else { $status = 'sai ma' }
    }
    $rows += [pscustomobject]@{ TrangThai = $status; MB = [math]::Round($size / 1MB, 2); File = $rel }
}

$rows | Sort-Object TrangThai, File | Format-Table -AutoSize
$ok = @($rows | Where-Object TrangThai -eq 'du').Count
$missing = @($rows | Where-Object TrangThai -eq 'thieu').Count
$bad = @($rows | Where-Object TrangThai -eq 'sai ma').Count
Write-Output ("Tong {0} file: du {1} | thieu {2} | sai ma {3}" -f $rows.Count, $ok, $missing, $bad)
if ($missing + $bad -gt 0) { exit 1 }
