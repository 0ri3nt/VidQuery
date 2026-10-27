# Validation evaluations for the fixes (current code; verdict flag off unless set by the caller).
# usage: eval_run.sh NAME DATABASE GENERATED_DIR T0|T1 SPLIT [extra KEY=VALUE ...]
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
R=evaluation/results/query-quality-fixes2-2026-10-27
NAME=$1 DB=$2 GEN=$3 MODE=$4 SPLIT=$5; shift 5
OUT=$R/$SPLIT-$MODE-$NAME.json
[ -f $OUT ] && { echo "$NAME $MODE $SPLIT exists"; exit 0; }
if [ "$MODE" = T1 ]; then SOFT=true; FRAME=true; else SOFT=false; FRAME=false; fi
EXTRA=(); for kv in "$@"; do EXTRA+=(--set "$kv"); done
"$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH=$DB \
  --set UPLOAD_DIR=data/app/benchmark/query_audit_2026-10-27/uploads --set GENERATED_DIR=$GEN \
  --set SOFT_CONSTRAINT_MATCHING=$SOFT --set ENABLE_FRAME_SEMANTIC_RETRIEVAL=$FRAME \
  --set FRAME_SEMANTIC_WEIGHT=1.0 --set FRAME_SEMANTIC_FLOOR=0.18 --set FRAME_SEMANTIC_CEILING=0.32 \
  "${EXTRA[@]}" -- "$PY" -m scripts.evaluate_moment_benchmark --split $SPLIT \
  --reliability-priors ${PRIORS:-default} \
  --spans native segment expand:10 --output $OUT > ${OUT%.json}.log 2>&1
echo "$NAME $MODE $SPLIT rc=$?"
