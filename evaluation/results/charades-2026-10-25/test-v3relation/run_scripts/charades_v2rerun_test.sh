source "C:/Users/Aryaman/AppData/Local/Temp/claude/c--Users-Aryaman-Desktop-work-capstone/f7fe97e8-5f76-41ae-a68d-2236af2f6e22/scratchpad/charades_v2rerun_env.sh"
"$PY" -c "
import sqlite3
src=sqlite3.connect('$B/charades-v2-rerun.sqlite3'); dst=sqlite3.connect('$B/charades-v2-rerun-snapshot-test.sqlite3'); src.backup(dst); dst.close(); src.close()"
export DATABASE_PATH=$B/charades-v2-rerun-snapshot-test.sqlite3
"$PY" -m scripts.backfill_frame_embeddings 2>&1 | tail -1
R=evaluation/results/charades-2026-10-25; T=$R/test-v2relation-rerun; mkdir -p $T
SPANS="native segment expand:10"
run () { # name soft frame priors   -- identical flags to the v2 test runs
  [ -f $T/$1.json ] && return
  SOFT_CONSTRAINT_MATCHING=$2 ENABLE_FRAME_SEMANTIC_RETRIEVAL=$3 FRAME_SEMANTIC_WEIGHT=1.0 FRAME_SEMANTIC_FLOOR=0.18 FRAME_SEMANTIC_CEILING=0.32 \
    "$PY" -m scripts.evaluate_moment_benchmark --split test --spans $SPANS --reliability-priors $4 --output $T/$1.json > $T/$1.log 2>&1
  echo "$1 rc=$?"
}
run T1-selected-soft-frame true true default
run T2-selected-p1-preserving-thresholds true true $R/abstention-thresholds-p1-preserving.json
run T0-default-strict-noframe false false default
