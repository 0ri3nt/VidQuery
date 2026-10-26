# YOLOv8n vs YOLOv8s on all 150 validation videos: indexing only (evaluation runs later, same code for both).
# The 50 videos indexed on 2026-10-26 by the same profile/code are copied, not modified; 100 are added.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
OLD=data/app/benchmark/yolo_audit_2026-10-26
ROOT=data/app/benchmark/yolo_val150_2026-10-27
RES=evaluation/results/yolo-validation150-2026-10-27
stamp () { echo "$1 $(date -u +%FT%TZ)" >> $RES/timeline.txt; }
for N in yolov8n yolov8s; do
  B=$ROOT/$N; mkdir -p $B $RES/$N
  if [ ! -f $B/index.sqlite3 ]; then
    "$PY" -c "
import sqlite3; s=sqlite3.connect('file:$OLD/$N/index.sqlite3?mode=ro', uri=True); d=sqlite3.connect('$B/index.sqlite3'); s.backup(d); d.close(); s.close()"
    cp -r $OLD/$N/generated $B/generated; cp $OLD/$N/progress.jsonl $B/progress.jsonl
  fi
  rm -f $RES/$N/STOP
  "$PY" $RES/run_scripts/proc_logger.py $RES/$N/resources.jsonl $RES/$N/STOP &
  stamp "index_${N}_start"
  "$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH=$B/index.sqlite3 \
    --set UPLOAD_DIR=$B/uploads --set GENERATED_DIR=$B/generated --set YOLO_MODEL=data/app/models/yolo/$N.pt \
    --record $RES/$N/profile.json -- "$PY" -m scripts.index_videos_batch \
    --manifest evaluation/charades_sta_navmr_benchmark.json --splits validation \
    --video-dir data/source/charades/videos --progress $B/progress.jsonl > $B/index.log 2>&1
  echo "index $N rc=$? ready=$(grep -c '"READY"' $B/progress.jsonl) errors=$(grep -c '"ERROR"' $B/progress.jsonl)"
  stamp "index_${N}_end"
  "$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH=$B/index.sqlite3 \
    --set UPLOAD_DIR=$B/uploads --set GENERATED_DIR=$B/generated -- "$PY" -m scripts.backfill_frame_embeddings > $RES/$N/backfill.log 2>&1
  echo "backfill $N rc=$?"
  touch $RES/$N/STOP
  stamp "backfill_${N}_end"
done
stamp all_indexed
