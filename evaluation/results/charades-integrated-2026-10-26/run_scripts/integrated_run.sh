# Integrated Charades-STA/NA-VMR rebuild: 4 deterministic-Whisper profiles; frozen queries/thresholds.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
ROOT=data/app/benchmark/charades_integrated_2026-10-26
RES=evaluation/results/charades-integrated-2026-10-26
OLD=evaluation/results/charades-2026-10-25
mkdir -p $RES
stamp () { echo "$1 $(date -u +%FT%TZ)" >> $RES/timeline.txt; }
withp () { # profile index-dir cmd...
  local P=$1 B=$2; shift 2
  "$PY" -m scripts.run_with_profile --profile $P --set DATABASE_PATH=$B/index-snapshot.sqlite3 \
    --set UPLOAD_DIR=$B/uploads --set GENERATED_DIR=$B/generated -- "$@"
}
evals () { # profile index-dir result-dir split
  local P=$1 B=$2 R=$3 SPLIT=$4
  run1 () { # name soft frame priors
    [ -f $R/$SPLIT-$1.json ] && return
    SOFT_CONSTRAINT_MATCHING=$2 ENABLE_FRAME_SEMANTIC_RETRIEVAL=$3 FRAME_SEMANTIC_WEIGHT=1.0 FRAME_SEMANTIC_FLOOR=0.18 FRAME_SEMANTIC_CEILING=0.32 \
      withp $P $B "$PY" -m scripts.evaluate_moment_benchmark --split $SPLIT --spans native segment expand:10 \
      --reliability-priors $4 --output $R/$SPLIT-$1.json > $R/$SPLIT-$1.log 2>&1
    echo "$P $SPLIT $1 rc=$?"
  }
  run1 T1-selected-soft-frame true true default
  run1 T2-selected-p1-preserving-thresholds true true $OLD/abstention-thresholds-p1-preserving.json
  run1 T0-default-strict-noframe false false default
}
build () { # profile splits...
  local P=$1; shift
  local B=$ROOT/$P R=$RES/$P
  mkdir -p $B $R
  stamp "index_${P}_start"
  "$PY" -m scripts.run_with_profile --profile $P --set DATABASE_PATH=$B/index.sqlite3 \
    --set UPLOAD_DIR=$B/uploads --set GENERATED_DIR=$B/generated --record $R/profile.json -- \
    "$PY" -m scripts.index_videos_batch --manifest evaluation/charades_sta_navmr_benchmark.json \
    --splits "$@" --video-dir data/source/charades/videos --progress $B/progress.jsonl > $B/index.log 2>&1
  echo "index $P rc=$? ready=$(grep -c '"READY"' $B/progress.jsonl) errors=$(grep -c '"ERROR"' $B/progress.jsonl)"
  stamp "index_${P}_end"
  "$PY" -c "
import sqlite3
s=sqlite3.connect('$B/index.sqlite3'); d=sqlite3.connect('$B/index-snapshot.sqlite3'); s.backup(d); d.close(); s.close()"
  withp $P $B "$PY" -m scripts.backfill_frame_embeddings > $R/backfill.log 2>&1
  stamp "eval_${P}_start"
  for SPLIT in "$@"; do evals $P $B $R $SPLIT; done
  stamp "eval_${P}_end"
}
build det-ava8-vidor4 validation test
build det-ava5-vidor2 test
build det-ava8-vidor2 test
build det-ava5-vidor4 test
stamp all_done
