# Scoring/routing variants on the validation split (T1 selected settings), audit copy of the integrated index.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
R=evaluation/results/query-quality-audit-2026-10-27
A=data/app/benchmark/query_audit_2026-10-27
G=data/app/benchmark/charades_integrated_2026-10-26/det-ava8-vidor4/generated
for V in baseline missing_penalty stopwords missing_penalty+stopwords no_transcripts missing_penalty+stopwords+no_transcripts missing_verdict_only; do
  OUT=$R/validation-T1-$V.json
  [ -f $OUT ] && continue
  "$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH=$A/index.sqlite3 \
    --set UPLOAD_DIR=$A/uploads --set GENERATED_DIR=$G --set SOFT_CONSTRAINT_MATCHING=true \
    --set ENABLE_FRAME_SEMANTIC_RETRIEVAL=true --set FRAME_SEMANTIC_WEIGHT=1.0 \
    --set FRAME_SEMANTIC_FLOOR=0.18 --set FRAME_SEMANTIC_CEILING=0.32 -- \
    "$PY" $R/run_scripts/scoring_audit.py evaluate --variant $V -- --split validation \
    --spans native segment expand:10 --output $OUT > $R/validation-T1-$V.log 2>&1
  echo "$V rc=$?"
done
