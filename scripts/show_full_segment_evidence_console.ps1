param(
    [string]$DatabasePath = "data/app/vidquery.sqlite3",
    [string]$VideoName = "She How many dates have we been to.mp4",
    [double]$SegmentStart = 0.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedDatabasePath = Join-Path $ProjectRoot $DatabasePath

if (-not (Test-Path -LiteralPath $ResolvedDatabasePath)) {
    throw "SQLite database not found: $ResolvedDatabasePath"
}
if (-not (Get-Command sqlite3 -ErrorAction SilentlyContinue)) {
    throw "sqlite3 is required."
}

try {
    $WindowSize = New-Object System.Management.Automation.Host.Size(190, 48)
    $BufferSize = New-Object System.Management.Automation.Host.Size(190, 3000)
    $Host.UI.RawUI.BufferSize = $BufferSize
    $Host.UI.RawUI.WindowSize = $WindowSize
} catch {
    # Some terminal hosts do not support programmatic resizing.
}

$Host.UI.RawUI.WindowTitle = "VidQuery - Full SQLite Segment Evidence"
$EscapedVideoName = $VideoName.Replace("'", "''")

function Show-PageHeader {
    param(
        [string]$Page,
        [string]$Title
    )

    Clear-Host
    Write-Host "VIDQUERY - ACTUAL SQLITE CANONICAL SEGMENT STORE" -ForegroundColor Green
    Write-Host "Database: $ResolvedDatabasePath" -ForegroundColor DarkGray
    Write-Host "Video: $VideoName | Segment start: $SegmentStart seconds | Read-only" -ForegroundColor DarkGray
    Write-Host "Page ${Page}: $Title" -ForegroundColor Yellow
    Write-Host ""
}

$SegmentSummarySql = @"
SELECT
  printf('%.1f-%.1fs', s.start_time, s.end_time) AS timestamp,
  s.transcript,
  json_extract(s.payload_json, '$.entities') AS entities,
  json_extract(s.payload_json, '$.actions') AS actions,
  json_extract(s.payload_json, '$.speakers') AS speakers,
  json_array_length(json_extract(s.payload_json, '$.frame_ids')) AS frames,
  json_array_length(json_extract(s.payload_json, '$.detections')) AS detections,
  json_array_length(json_extract(s.payload_json, '$.relationships')) AS relationships,
  json_array_length(json_extract(s.payload_json, '$.ocr_evidence')) AS ocr
FROM segments AS s
JOIN videos AS v ON v.video_id = s.video_id
WHERE v.display_name = '$EscapedVideoName'
  AND s.start_time = $SegmentStart;
"@

$ModelMetadataSql = @"
SELECT 'YOLO detector' AS module,
       json_extract(s.payload_json, '$.processing_metadata.yolo_model') AS model,
       json_extract(s.payload_json, '$.processing_metadata.visual_detector_status') AS status,
       json_extract(s.payload_json, '$.processing_metadata.visual_detection_count') AS evidence
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT 'Whisper',
       json_extract(s.payload_json, '$.processing_metadata.whisper_model'),
       json_extract(s.payload_json, '$.processing_metadata.transcription_status'),
       json_extract(s.payload_json, '$.processing_metadata.transcript_segment_count')
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT 'Pyannote',
       json_extract(s.payload_json, '$.processing_metadata.pyannote_model'),
       json_extract(s.payload_json, '$.processing_metadata.diarization_status'),
       json_extract(s.payload_json, '$.processing_metadata.speaker_segment_count')
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT 'EasyOCR',
       json_extract(s.payload_json, '$.processing_metadata.ocr_source_method'),
       json_extract(s.payload_json, '$.processing_metadata.ocr_status'),
       json_extract(s.payload_json, '$.processing_metadata.ocr_evidence_count')
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT 'OpenCLIP appearance',
       json_extract(s.payload_json, '$.processing_metadata.appearance_embedding_model'),
       json_extract(s.payload_json, '$.processing_metadata.appearance_status'),
       json_extract(s.payload_json, '$.processing_metadata.appearance_track_count')
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT 'AVA action GNN',
       json_extract(s.payload_json, '$.processing_metadata.gnn_model_version'),
       json_extract(s.payload_json, '$.processing_metadata.learned_action_model'),
       json_array_length(json_extract(s.payload_json, '$.processing_metadata.gnn_action_instances'))
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT 'VidOR relation GNN',
       json_extract(s.payload_json, '$.processing_metadata.relationship_gnn_model_version'),
       json_extract(s.payload_json, '$.processing_metadata.relationship_gnn_status'),
       json_extract(s.payload_json, '$.processing_metadata.relationship_interval_count')
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT 'MiniLM semantic',
       json_extract(s.payload_json, '$.processing_metadata.sentence_embedding_model'),
       CASE json_extract(s.payload_json, '$.processing_metadata.sentence_embedding_cached') WHEN 1 THEN 'cached' ELSE 'not cached' END,
       json_extract(s.payload_json, '$.processing_metadata.sentence_embedding_dimensions')
FROM segments s JOIN videos v ON v.video_id=s.video_id
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart;
"@

$DetectionSql = @"
SELECT
  json_extract(d.value, '$.class_label') AS class,
  round(json_extract(d.value, '$.confidence'), 3) AS yolo_conf,
  printf('[%.0f,%.0f,%.0f,%.0f]',
    json_extract(d.value, '$.pixel_bbox.x1'),
    json_extract(d.value, '$.pixel_bbox.y1'),
    json_extract(d.value, '$.pixel_bbox.x2'),
    json_extract(d.value, '$.pixel_bbox.y2')) AS pixel_bbox,
  json_extract(d.value, '$.track_id') AS track_id,
  json_extract(d.value, '$.appearance_id') AS appearance_id
FROM segments s
JOIN videos v ON v.video_id=s.video_id,
json_each(s.payload_json, '$.detections') d
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
LIMIT 12;
"@

$AppearanceSql = @"
SELECT
  a.track_id,
  a.class_label AS class,
  a.appearance_embedding_model AS clip_model,
  json_array_length(a.embedding_json) AS dimensions,
  a.observation_count AS observations,
  json_extract(a.attributes_json, '$.upper_clothing_description.value') AS clothing,
  json_extract(a.attributes_json, '$.glasses.value') AS glasses,
  a.source_method
FROM entity_appearances a
JOIN videos v ON v.video_id=a.video_id
WHERE v.display_name='$EscapedVideoName'
ORDER BY a.track_id;
"@

$ActionSql = @"
SELECT
  action.key AS action,
  round(action.value, 3) AS confidence,
  json_extract(
    s.payload_json,
    '$.processing_metadata.action_sources."' || action.key || '"'
  ) AS provenance
FROM segments s
JOIN videos v ON v.video_id=s.video_id,
json_each(json_extract(s.payload_json, '$.processing_metadata.action_confidences')) action
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
ORDER BY action.value DESC
LIMIT 12;
"@

$RelationshipSql = @"
SELECT
  json_extract(r.value, '$.predicate') AS predicate,
  round(json_extract(r.value, '$.confidence'), 3) AS confidence,
  json_extract(r.value, '$.source_method') AS provenance,
  coalesce(json_extract(r.value, '$.model_version'), '-') AS model_version,
  json_extract(r.value, '$.source_track_id') AS source_track,
  json_extract(r.value, '$.target_track_id') AS target_track
FROM segments s
JOIN videos v ON v.video_id=s.video_id,
json_each(s.payload_json, '$.relationships') r
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
ORDER BY json_extract(r.value, '$.confidence') DESC
LIMIT 14;
"@

$SpeechOcrSql = @"
SELECT
  'transcript' AS evidence_type,
  json_extract(t.value, '$.text') AS evidence,
  json_extract(t.value, '$.speaker') AS source,
  json_extract(t.value, '$.start_time') || '-' || json_extract(t.value, '$.end_time') || 's' AS interval,
  coalesce(round(json_extract(t.value, '$.confidence'), 3), '-') AS confidence
FROM segments s
JOIN videos v ON v.video_id=s.video_id,
json_each(s.payload_json, '$.transcript_segments') t
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart
UNION ALL
SELECT
  'OCR',
  json_extract(o.value, '$.text'),
  json_extract(o.value, '$.source_method'),
  json_extract(o.value, '$.start_time') || '-' || json_extract(o.value, '$.end_time') || 's',
  round(json_extract(o.value, '$.confidence'), 3)
FROM segments s
JOIN videos v ON v.video_id=s.video_id,
json_each(s.payload_json, '$.ocr_evidence') o
WHERE v.display_name='$EscapedVideoName' AND s.start_time=$SegmentStart;
"@

Show-PageHeader -Page "1/3" -Title "Canonical segment and model metadata"
Write-Host "sqlite> SELECT complete canonical-segment summary FROM segments.payload_json;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 13 24 18 50 18 7 11 13 5" $ResolvedDatabasePath $SegmentSummarySql
Write-Host ""
Write-Host "sqlite> SELECT model status and evidence counts FROM processing_metadata;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 22 52 35 10" $ResolvedDatabasePath $ModelMetadataSql
Write-Host ""
Read-Host "Take screenshot 1, then press Enter for detection and CLIP storage"

Show-PageHeader -Page "2/3" -Title "YOLO detections linked to normalized OpenCLIP records"
Write-Host "sqlite> SELECT detections, bounding boxes, track IDs and appearance IDs FROM segment JSON;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 12 10 24 28 34" $ResolvedDatabasePath $DetectionSql
Write-Host ""
Write-Host "sqlite> SELECT linked 512-D OpenCLIP embeddings and attributes FROM entity_appearances;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 28 10 24 11 12 18 9 24" $ResolvedDatabasePath $AppearanceSql
Write-Host ""
Write-Host "CLIP vectors are normalized into entity_appearances and referenced by appearance_id from each segment detection." -ForegroundColor DarkGray
Write-Host "This avoids duplicating the same 512-dimensional track vector across every five-second segment." -ForegroundColor DarkGray
Write-Host ""
Read-Host "Take screenshot 2, then press Enter for action, relationship, speech and OCR evidence"

Show-PageHeader -Page "3/3" -Title "Learned actions, relationships, transcript, speaker and OCR evidence"
Write-Host "sqlite> SELECT AVA action confidence and provenance FROM processing_metadata;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 24 12 34" $ResolvedDatabasePath $ActionSql
Write-Host ""
Write-Host "sqlite> SELECT relationship confidence, provenance and model version FROM segment JSON;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 20 12 36 42 28 28" $ResolvedDatabasePath $RelationshipSql
Write-Host ""
Write-Host "sqlite> SELECT Whisper/Pyannote transcript and EasyOCR evidence FROM segment JSON;" -ForegroundColor Cyan
& sqlite3 -readonly -header -column -cmd ".width 15 54 20 18 12" $ResolvedDatabasePath $SpeechOcrSql
Write-Host ""
Read-Host "Take screenshot 3, then press Enter to close"
