# YOLO detector audit: full pipeline rebuild of 50 validation videos per detector,
# identical det-ava8-vidor4 profile otherwise; evaluated with frozen queries/thresholds.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
ROOT=data/app/benchmark/yolo_audit_2026-10-26
RES=evaluation/results/yolo-audit-2026-10-26
MAN=$RES/validation50_manifest.json
W=$ROOT/weights
stamp () { echo "$1 $(date -u +%FT%TZ)" >> $RES/timeline.txt; }
run () { # name weights
  local N=$1 Y=$2 B=$ROOT/$1 R=$RES/$1
  mkdir -p $B $R
  withp () { "$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH=$B/index.sqlite3 \
    --set UPLOAD_DIR=$B/uploads --set GENERATED_DIR=$B/generated --set YOLO_MODEL=$Y "$@"; }
  stamp "index_${N}_start"
  withp --record $R/profile.json -- "$PY" -m scripts.index_videos_batch --manifest $MAN --splits validation \
    --video-dir data/source/charades/videos --progress $B/progress.jsonl > $B/index.log 2>&1
  echo "index $N rc=$? ready=$(grep -c '"READY"' $B/progress.jsonl) errors=$(grep -c '"ERROR"' $B/progress.jsonl)"
  stamp "index_${N}_end"
  withp -- "$PY" -m scripts.backfill_frame_embeddings > $R/backfill.log 2>&1
  for T in "T1-selected-soft-frame true true" "T0-default-strict-noframe false false"; do
    set -- $T
    [ -f $R/validation-$1.json ] && continue
    SOFT_CONSTRAINT_MATCHING=$2 ENABLE_FRAME_SEMANTIC_RETRIEVAL=$3 FRAME_SEMANTIC_WEIGHT=1.0 FRAME_SEMANTIC_FLOOR=0.18 FRAME_SEMANTIC_CEILING=0.32 \
      withp -- "$PY" -m scripts.evaluate_moment_benchmark --manifest $MAN --split validation --spans native segment expand:10 \
      --output $R/validation-$1.json > $R/validation-$1.log 2>&1
    echo "$N $1 rc=$?"
  done
  stamp "eval_${N}_end"
}
run yolov8n data/app/models/yolo/yolov8n.pt
run yolov8s data/app/models/yolo/yolov8s.pt
run yolov8m data/app/models/yolo/yolov8m.pt
run yolo11s $W/yolo11s.pt
run yolo11m $W/yolo11m.pt
run yolo26s $W/yolo26s.pt
stamp all_done
