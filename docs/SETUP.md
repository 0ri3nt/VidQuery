# Setup, model caches, and troubleshooting

This guide takes a fresh clone to a state where every model runs offline and
`GET /api/health` reports all components ready. It covers macOS, Linux, and
Windows.

## 1. Environment

Requirements: Python 3.11 or 3.12, FFmpeg/ffprobe on `PATH` (the
`imageio-ffmpeg` fallback also works), and several GB of free disk for models
and extracted media.

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,full]"
cp .env.example .env
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,full]"
Copy-Item .env.example .env
```

Install `torch`, `torchvision`, and `torchaudio` from the same release. A
mismatched `torchaudio` fails at import with an ABI error, which breaks
Pyannote.

## 2. Credentials

Edit `.env`. It is gitignored; never commit it.

| Variable | Needed for |
|---|---|
| `HUGGINGFACE_TOKEN` | Pyannote diarization. The account must have accepted the `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` terms. |
| `GROQ_API_KEY` | optional constrained query planner and grounded answers. Search works without it. |
| `NEO4J_PASSWORD` | only when `ENABLE_NEO4J=true` |

## 3. Model cache layout

Everything below lives under `data/app/`, which is gitignored. For the first
run, set `ALLOW_MODEL_DOWNLOADS=true` and `HF_HUB_OFFLINE=0`. After everything
is cached, switch back to `false` and `1`.

| Component | Setting | Expected local file or directory |
|---|---|---|
| YOLO | `YOLO_MODEL` | `data/app/models/yolo/yolov8n.pt` |
| Whisper | `WHISPER_MODEL=tiny`, `WHISPER_CACHE_DIR` | `data/app/models/whisper/tiny.pt` |
| Pyannote | `PYANNOTE_CACHE_DIR` | `data/app/models/pyannote/models--pyannote--speaker-diarization-3.1/` |
| MiniLM | `SENTENCE_EMBEDDING_CACHE_DIR` | `data/app/models/sentence_transformers/models--sentence-transformers--all-MiniLM-L6-v2/` |
| OpenCLIP | `APPEARANCE_CACHE_DIR` | `data/app/models/appearance/ViT-B-32.pt` |
| EasyOCR | `OCR_MODEL_DIR` | `data/app/models/easyocr/craft_mlt_25k.pth`, `english_g2.pth` |
| ResNet-18 backbone (AVA and VidOR features) | `TORCH_HOME` | `data/app/cache/torch/hub/checkpoints/resnet18-f37072fd.pth` |
| AVA action GNN | `GNN_CHECKPOINT`, `GNN_METADATA` | `data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.{pt,metadata.json}` |
| VidOR relation GNN | `RELATION_GNN_CHECKPOINT`, `RELATION_GNN_METADATA` | `data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.{pt,metadata.json}` |

The feature extractors set `TORCH_HOME` to `data/app/cache/torch` when it is
unset, and load the ResNet file from there directly when it exists, so no
network call is made. To fetch it manually:

```bash
mkdir -p data/app/cache/torch/hub/checkpoints
curl -fL -o data/app/cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  https://download.pytorch.org/models/resnet18-f37072fd.pth
```

The GNN checkpoints are trained artifacts, not downloads. Copy them to the paths
above; the metadata JSON must sit next to the `.pt` file. Checkpoints are
checksum-validated, and invalid ones are rejected visibly. Random weights are
never used. To enable the AVA GNN, set `ENABLE_GNN_ACTIONS=true`.

## 4. Verify

```bash
python -m vidquery serve --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/api/health | python -m json.tool
```

A fully configured run shows YOLO and Whisper as `available_configured`,
diarization available, and the GNN keys ending in `_enabled` or
`available_validated`. Any `unavailable_*` state names the missing piece.

Run the test suite:

```bash
python -m pytest -q
```

## 5. Reprocess after changing models

Model changes do not rewrite existing segments. Reprocess from the Library tab,
or run:

```bash
python -m vidquery refresh-library --video-id VIDEO_UUID
python -m vidquery refresh-library          # every registered video
```

Then confirm what was stored with
`python -m vidquery diagnose-query "..." --video-id VIDEO_UUID`
(see [RETRIEVAL_DIAGNOSTICS.md](RETRIEVAL_DIAGNOSTICS.md)).

## Troubleshooting

**`URLError: CERTIFICATE_VERIFY_FAILED` on macOS.** The python.org macOS build
does not use the system keychain. Either run
`/Applications/Python\ 3.12/Install\ Certificates.command`, or point Python at
certifi in your local `.env`:

```dotenv
SSL_CERT_FILE=/path/to/.venv/lib/python3.12/site-packages/certifi/cacert.pem
REQUESTS_CA_BUNDLE=/path/to/.venv/lib/python3.12/site-packages/certifi/cacert.pem
```

`python -c "import certifi; print(certifi.where())"` prints the path. If a
download still fails, fetch the file with `curl` into the cache path from the
table above.

**The GNN produces no actions or relationships.** The ResNet backbone is almost
always missing from `data/app/cache/torch`. Cache it, then reprocess.

**Whisper fails on Apple Silicon with a SparseMPS `NotImplementedError`.**
VidQuery already forces Whisper onto the CPU when the resolved device is MPS.
Other models still use MPS.

**A reprocess appears to fail or hang in the UI.** Diarization on CPU is the
slowest stage and can take several minutes for a short clip. A second click
while a job is running returns `409 Conflict`; this does not mean the job
failed. Poll `GET /api/videos/{id}/status`: the `stage`, `warnings`, and final
`READY` or `FAILED` state are authoritative.

**Model files appear in the repository root.** With downloads allowed and a
bare `YOLO_MODEL=yolov8n.pt`, Ultralytics downloads into the working directory.
Keep `YOLO_MODEL` pointed at `data/app/models/yolo/yolov8n.pt`.
