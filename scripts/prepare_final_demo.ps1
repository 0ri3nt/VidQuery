param(
    [string]$SourceImage = "data/demo/source/meeting_scene.png",
    [string]$OutputVideo = "data/demo/final_demo.mp4",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = (Resolve-Path (Join-Path $projectRoot $SourceImage)).Path
$outputPath = Join-Path $projectRoot $OutputVideo
$workPath = Join-Path $projectRoot "data/demo/work"
New-Item -ItemType Directory -Force -Path $workPath | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $outputPath) | Out-Null

$pythonPath = (Get-Command $PythonExecutable -ErrorAction Stop).Source
$ffmpeg = & $pythonPath -c `
    "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if (-not (Test-Path $ffmpeg)) {
    throw "FFmpeg was not found. Install requirements.txt in the active environment."
}

$voices = @(
    @{ Name = "speaker_00"; Index = 1; Rate = 1; Text = "We discuss the software architecture." },
    @{ Name = "speaker_01"; Index = 2; Rate = 1; Text = "The laptop runs the API with SQLite metadata." },
    @{ Name = "speaker_00_deploy"; Index = 1; Rate = 1; Text = "Deployment uses Docker for the backend and Neo4j graph." },
    @{ Name = "speaker_01_board"; Index = 2; Rate = 1; Text = "This whiteboard shows the retrieval architecture." },
    @{ Name = "speaker_00_close"; Index = 1; Rate = 1; Text = "The system searches speech, objects, and relationships." }
)

foreach ($item in $voices) {
    $wavePath = Join-Path $workPath ($item.Name + ".wav")
    $voice = New-Object -ComObject SAPI.SpVoice
    $voice.Voice = $voice.GetVoices().Item($item.Index)
    $voice.Rate = $item.Rate
    $stream = New-Object -ComObject SAPI.SpFileStream
    # SAFT16kHz16BitMono is supported by the bundled Windows desktop voices.
    # Explicitly setting it also avoids inheriting a live-device format.
    $stream.Format.Type = 18
    $stream.Open($wavePath, 3, $false)
    $voice.AudioOutputStream = $stream
    [void]$voice.Speak($item.Text)
    $stream.Close()
}

$audioFilter = "[1:a]adelay=250|250[a1];[2:a]adelay=5250|5250[a2];" +
    "[3:a]adelay=10250|10250[a3];[4:a]adelay=15250|15250[a4];" +
    "[5:a]adelay=20250|20250[a5];[a1][a2][a3][a4][a5]amix=inputs=5:duration=longest[a]"

$videoFilter = "scale=1280:720:force_original_aspect_ratio=decrease," +
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p"

$ffmpegArguments = @(
    "-y", "-loop", "1", "-framerate", "25", "-i", $sourcePath,
    "-i", (Join-Path $workPath "speaker_00.wav"),
    "-i", (Join-Path $workPath "speaker_01.wav"),
    "-i", (Join-Path $workPath "speaker_00_deploy.wav"),
    "-i", (Join-Path $workPath "speaker_01_board.wav"),
    "-i", (Join-Path $workPath "speaker_00_close.wav"),
    "-filter_complex", $audioFilter,
    "-map", "0:v", "-map", "[a]", "-t", "25", "-r", "25",
    "-vf", $videoFilter,
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", $outputPath
)

& $ffmpeg @ffmpegArguments
if ($LASTEXITCODE -ne 0) {
    throw "FFmpeg failed with exit code $LASTEXITCODE"
}
Write-Output "Created controlled demo video: $outputPath"
