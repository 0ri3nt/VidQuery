param(
    [string]$DatabasePath = "data/app/vidquery.sqlite3",
    [string]$VideoName = "final_demo.mp4",
    [string]$OutputPath = "docs/screenshots/24-sqlite-segments.png"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedDatabasePath = Join-Path $ProjectRoot $DatabasePath
$ResolvedOutputPath = Join-Path $ProjectRoot $OutputPath

if (-not (Test-Path -LiteralPath $ResolvedDatabasePath)) {
    throw "SQLite database not found: $ResolvedDatabasePath"
}

if (-not (Get-Command sqlite3 -ErrorAction SilentlyContinue)) {
    throw "sqlite3 is required to capture the database evidence image."
}

$EscapedVideoName = $VideoName.Replace("'", "''")

$SummarySql = @"
SELECT
    COUNT(DISTINCT v.video_id) AS indexed_videos,
    COUNT(s.segment_id) AS stored_segments,
    SUM(CASE WHEN v.display_name = '$EscapedVideoName' THEN 1 ELSE 0 END) AS displayed_segments
FROM videos AS v
LEFT JOIN segments AS s ON s.video_id = v.video_id;
"@

$SegmentsSql = @"
SELECT
    v.display_name AS video,
    printf('%.1f - %.1f s', s.start_time, s.end_time) AS timestamp,
    substr(replace(s.transcript, char(10), ' '), 1, 70) AS transcript,
    json_extract(s.payload_json, '$.entities') AS entities,
    json_array_length(json_extract(s.payload_json, '$.actions')) AS actions,
    json_array_length(json_extract(s.payload_json, '$.relationships')) AS relations,
    json_array_length(json_extract(s.payload_json, '$.ocr_evidence')) AS ocr
FROM segments AS s
JOIN videos AS v ON v.video_id = s.video_id
WHERE v.display_name = '$EscapedVideoName'
ORDER BY s.start_time;
"@

function Invoke-SqliteJson {
    param([string]$Sql)

    $RawResult = & sqlite3 -readonly -json $ResolvedDatabasePath $Sql
    if ($LASTEXITCODE -ne 0) {
        throw "sqlite3 query failed."
    }
    if (-not $RawResult) {
        return @()
    }
    return @($RawResult | ConvertFrom-Json)
}

$SummaryRows = Invoke-SqliteJson -Sql $SummarySql
$SegmentRows = Invoke-SqliteJson -Sql $SegmentsSql

if ($SummaryRows.Count -eq 0) {
    throw "The SQLite summary query returned no data."
}
if ($SegmentRows.Count -eq 0) {
    throw "No canonical segments were found for '$VideoName'."
}

$OutputDirectory = Split-Path -Parent $ResolvedOutputPath
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

Add-Type -AssemblyName System.Drawing

$Width = 2000
$Height = 1180
$Bitmap = New-Object System.Drawing.Bitmap($Width, $Height)
$Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
$Graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$Graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$Graphics.Clear([System.Drawing.Color]::FromArgb(13, 17, 23))

$TitleFont = New-Object System.Drawing.Font("Segoe UI", 34, [System.Drawing.FontStyle]::Bold)
$SubtitleFont = New-Object System.Drawing.Font("Segoe UI", 17, [System.Drawing.FontStyle]::Regular)
$MetricFont = New-Object System.Drawing.Font("Segoe UI", 25, [System.Drawing.FontStyle]::Bold)
$MetricLabelFont = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Regular)
$HeaderFont = New-Object System.Drawing.Font("Consolas", 15, [System.Drawing.FontStyle]::Bold)
$CellFont = New-Object System.Drawing.Font("Consolas", 13, [System.Drawing.FontStyle]::Regular)
$QueryFont = New-Object System.Drawing.Font("Consolas", 13, [System.Drawing.FontStyle]::Regular)
$CaptionFont = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Italic)

$WhiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(240, 246, 252))
$MutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(139, 148, 158))
$GreenBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(63, 185, 80))
$BlueBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(88, 166, 255))
$OrangeBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(210, 153, 34))
$PanelBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(22, 27, 34))
$HeaderBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(33, 38, 45))
$RowAltBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(18, 23, 30))
$BorderPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(48, 54, 61), 1)
$AccentPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(57, 211, 83), 4)

$Graphics.DrawString("SQLite Canonical Segment Store", $TitleFont, $WhiteBrush, 55, 38)
$Graphics.DrawString(
    "Live evidence from data/app/vidquery.sqlite3 | read-only query | $VideoName",
    $SubtitleFont,
    $MutedBrush,
    58,
    92
)
$Graphics.DrawLine($AccentPen, 58, 132, 1940, 132)

$Summary = $SummaryRows[0]
$MetricDefinitions = @(
    @{ X = 58; Label = "INDEXED VIDEOS"; Value = [string]$Summary.indexed_videos; Brush = $BlueBrush },
    @{ X = 390; Label = "STORED SEGMENTS"; Value = [string]$Summary.stored_segments; Brush = $GreenBrush },
    @{ X = 722; Label = "DISPLAYED SEGMENTS"; Value = [string]$Summary.displayed_segments; Brush = $OrangeBrush }
)

foreach ($Metric in $MetricDefinitions) {
    $Graphics.FillRectangle($PanelBrush, $Metric.X, 158, 300, 108)
    $Graphics.DrawRectangle($BorderPen, $Metric.X, 158, 300, 108)
    $Graphics.DrawString($Metric.Label, $MetricLabelFont, $MutedBrush, $Metric.X + 20, 174)
    $Graphics.DrawString($Metric.Value, $MetricFont, $Metric.Brush, $Metric.X + 20, 205)
}

$Graphics.FillRectangle($PanelBrush, 1054, 158, 886, 108)
$Graphics.DrawRectangle($BorderPen, 1054, 158, 886, 108)
$Graphics.DrawString("SQL QUERY", $MetricLabelFont, $MutedBrush, 1075, 174)
$Graphics.DrawString(
    "SELECT timestamp, transcript, entities, action/relation/OCR counts`nFROM segments JOIN videos WHERE display_name = '$VideoName';",
    $QueryFont,
    $WhiteBrush,
    1075,
    202
)

$TableX = 58
$TableY = 302
$HeaderHeight = 58
$RowHeight = 132
$ColumnWidths = @(210, 185, 430, 690, 105, 120, 80)
$ColumnHeaders = @("VIDEO", "TIMESTAMP", "TRANSCRIPT", "ENTITIES", "ACTIONS", "RELATIONS", "OCR")
$TableWidth = ($ColumnWidths | Measure-Object -Sum).Sum

$Graphics.FillRectangle($HeaderBrush, $TableX, $TableY, $TableWidth, $HeaderHeight)
$Graphics.DrawRectangle($BorderPen, $TableX, $TableY, $TableWidth, $HeaderHeight)

$CurrentX = $TableX
for ($Index = 0; $Index -lt $ColumnHeaders.Count; $Index++) {
    $Graphics.DrawString($ColumnHeaders[$Index], $HeaderFont, $GreenBrush, $CurrentX + 12, $TableY + 17)
    if ($Index -gt 0) {
        $Graphics.DrawLine($BorderPen, $CurrentX, $TableY, $CurrentX, $TableY + $HeaderHeight + ($SegmentRows.Count * $RowHeight))
    }
    $CurrentX += $ColumnWidths[$Index]
}

$StringFormat = New-Object System.Drawing.StringFormat
$StringFormat.Trimming = [System.Drawing.StringTrimming]::EllipsisWord
$StringFormat.FormatFlags = [System.Drawing.StringFormatFlags]::LineLimit

for ($RowIndex = 0; $RowIndex -lt $SegmentRows.Count; $RowIndex++) {
    $Row = $SegmentRows[$RowIndex]
    $Y = $TableY + $HeaderHeight + ($RowIndex * $RowHeight)
    if (($RowIndex % 2) -eq 1) {
        $Graphics.FillRectangle($RowAltBrush, $TableX, $Y, $TableWidth, $RowHeight)
    } else {
        $Graphics.FillRectangle($PanelBrush, $TableX, $Y, $TableWidth, $RowHeight)
    }
    $Graphics.DrawRectangle($BorderPen, $TableX, $Y, $TableWidth, $RowHeight)

    $EntityText = if ($null -eq $Row.entities) { "[]" } else { [string]$Row.entities }
    $EntityText = $EntityText.Replace('["', '').Replace('"]', '').Replace('","', ', ')
    $TranscriptText = if ([string]::IsNullOrWhiteSpace([string]$Row.transcript)) {
        "No speech in this segment"
    } else {
        [string]$Row.transcript
    }

    $Values = @(
        [string]$Row.video,
        [string]$Row.timestamp,
        $TranscriptText,
        $EntityText,
        [string]$Row.actions,
        [string]$Row.relations,
        [string]$Row.ocr
    )

    $CurrentX = $TableX
    for ($ColumnIndex = 0; $ColumnIndex -lt $Values.Count; $ColumnIndex++) {
        $CellRectangle = New-Object System.Drawing.RectangleF(
            [single]($CurrentX + 12),
            [single]($Y + 14),
            [single]($ColumnWidths[$ColumnIndex] - 24),
            [single]($RowHeight - 28)
        )
        $CellBrush = if ($ColumnIndex -ge 4) { $BlueBrush } else { $WhiteBrush }
        $Graphics.DrawString($Values[$ColumnIndex], $CellFont, $CellBrush, $CellRectangle, $StringFormat)
        $CurrentX += $ColumnWidths[$ColumnIndex]
    }
}

$CaptionY = $TableY + $HeaderHeight + ($SegmentRows.Count * $RowHeight) + 30
$Graphics.DrawString(
    "Each row is a persisted timestamp interval containing aligned transcript, entities, actions, relationships and OCR evidence.",
    $CaptionFont,
    $MutedBrush,
    60,
    $CaptionY
)
$Graphics.DrawString(
    "SQLite remains VidQuery's authoritative local metadata and retrieval store.",
    $CaptionFont,
    $MutedBrush,
    60,
    $CaptionY + 30
)

$Bitmap.Save($ResolvedOutputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$Graphics.Dispose()
$Bitmap.Dispose()
$TitleFont.Dispose()
$SubtitleFont.Dispose()
$MetricFont.Dispose()
$MetricLabelFont.Dispose()
$HeaderFont.Dispose()
$CellFont.Dispose()
$QueryFont.Dispose()
$CaptionFont.Dispose()
$WhiteBrush.Dispose()
$MutedBrush.Dispose()
$GreenBrush.Dispose()
$BlueBrush.Dispose()
$OrangeBrush.Dispose()
$PanelBrush.Dispose()
$HeaderBrush.Dispose()
$RowAltBrush.Dispose()
$BorderPen.Dispose()
$AccentPen.Dispose()
$StringFormat.Dispose()

Write-Output "Saved SQLite evidence screenshot: $ResolvedOutputPath"
