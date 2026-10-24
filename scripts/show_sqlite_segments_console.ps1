param(
    [string]$DatabasePath = "data/app/vidquery.sqlite3",
    [string]$VideoName = "0f39OWEqJ24.mp4"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedDatabasePath = Join-Path $ProjectRoot $DatabasePath

if (-not (Test-Path -LiteralPath $ResolvedDatabasePath)) {
    throw "SQLite database not found: $ResolvedDatabasePath"
}

try {
    $WindowSize = New-Object System.Management.Automation.Host.Size(178, 44)
    $BufferSize = New-Object System.Management.Automation.Host.Size(178, 2000)
    $Host.UI.RawUI.BufferSize = $BufferSize
    $Host.UI.RawUI.WindowSize = $WindowSize
} catch {
    # Some terminal hosts do not allow programmatic resizing.
}

$Host.UI.RawUI.WindowTitle = "VidQuery - Actual SQLite Canonical Segments"
Clear-Host

$EscapedVideoName = $VideoName.Replace("'", "''")

$SummarySql = @"
SELECT
  v.display_name AS video,
  v.state AS state,
  COUNT(s.segment_id) AS stored_segments,
  SUM(CASE WHEN trim(s.transcript) <> '' THEN 1 ELSE 0 END) AS speech_segments
FROM videos AS v
JOIN segments AS s ON s.video_id = v.video_id
WHERE v.display_name = '$EscapedVideoName'
GROUP BY v.video_id;
"@

$SegmentsSql = @"
SELECT
  printf('%.1f-%.1fs', s.start_time, s.end_time) AS timestamp,
  substr(replace(s.transcript, char(10), ' '), 1, 42) AS transcript,
  substr(json_extract(s.payload_json, '$.entities'), 1, 34) AS entities,
  substr(json_extract(s.payload_json, '$.actions'), 1, 46) AS actions,
  json_array_length(json_extract(s.payload_json, '$.relationships')) AS relations
FROM segments AS s
JOIN videos AS v ON v.video_id = s.video_id
WHERE v.display_name = '$EscapedVideoName'
  AND trim(s.transcript) <> ''
  AND json_array_length(json_extract(s.payload_json, '$.entities')) > 0
ORDER BY s.start_time
LIMIT 12;
"@

Write-Host "VIDQUERY - ACTUAL SQLITE DATABASE" -ForegroundColor Green
Write-Host "Database: $ResolvedDatabasePath" -ForegroundColor DarkGray
Write-Host "Mode: read-only" -ForegroundColor DarkGray
Write-Host ""
Write-Host "sqlite> SELECT stored segment summary for '$VideoName';" -ForegroundColor Cyan
& sqlite3 -readonly -header -column $ResolvedDatabasePath $SummarySql

Write-Host ""
Write-Host "sqlite> SELECT timestamp, transcript, entities, actions, relationship count FROM segments;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 14 42 34 46 9" $ResolvedDatabasePath $SegmentsSql

Write-Host ""
Write-Host "Source table: segments (payload_json contains the complete canonical multimodal evidence)." -ForegroundColor DarkGray
Write-Host "This window is displaying direct sqlite3 output from the live VidQuery database." -ForegroundColor DarkGray
Write-Host ""
Read-Host "Press Enter after taking the screenshot"
