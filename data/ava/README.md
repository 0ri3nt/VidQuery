# AVA Data Layout

This project expects AVA assets in the following modular structure:

- data/ava/annotations
  - ava_train_v2.2.csv
  - ava_val_v2.2.csv
- data/ava/videos
  - *.mp4
  - ava_train_v2.2.txt (download list used by download.go)
- data/ava/audio
  - extracted wav files (optional)
- data/ava/extracted_frames
  - train/<video_id>/*.jpg
  - val/<video_id>/*.jpg
- data/ava/detections
  - train/<video_id>/*.json
  - val/<video_id>/*.json
- data/ava/scene_graphs
  - train/<video_id>/*.json
  - val/<video_id>/*.json
- data/ava/audio_outputs
  - whisper outputs
  - diarization outputs
- data/ava/visualizations
  - frame-level graph overlays
- data/ava/visualizations_video
  - video-level graph overlays
