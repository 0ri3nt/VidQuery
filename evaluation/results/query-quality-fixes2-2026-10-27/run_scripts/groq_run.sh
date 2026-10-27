# Round-2 live Groq re-audit, paced under 6,000 tokens/minute and capped per run so the
# account's 200,000 tokens/day limit is not reached. Same libraries and settings as round 1.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
R=evaluation/results/query-quality-fixes2-2026-10-27
Q=evaluation/results/query-quality-audit-2026-10-27
DEMO="C:/Users/Aryaman/AppData/Local/Temp/claude/c--Users-Aryaman-Desktop-work-capstone/f7fe97e8-5f76-41ae-a68d-2236af2f6e22/scratchpad/live/library_a"
case "$1" in
production)
  # Production library copy with the production defaults from .env (no experimental flags).
  DATABASE_PATH=data/app/benchmark/query_audit_2026-10-27/production-copy.sqlite3 \
    "$PY" $R/run_scripts/groq_reaudit.py collect $Q/groq_queries_production.json \
    $R/groq-production-records.json --subset 4 --budget 50000 > $R/groq-production.log 2>&1 ;;
demo)
  "$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH="$DEMO/test.sqlite3" \
    --set UPLOAD_DIR="$DEMO/uploads" --set GENERATED_DIR="$DEMO/generated" --set SOFT_CONSTRAINT_MATCHING=true \
    --set ENABLE_FRAME_SEMANTIC_RETRIEVAL=true --set FRAME_SEMANTIC_WEIGHT=1.0 --set FRAME_SEMANTIC_FLOOR=0.18 \
    --set FRAME_SEMANTIC_CEILING=0.32 -- "$PY" $R/run_scripts/groq_reaudit.py collect \
    $Q/groq_queries_demo.json $R/groq-demo-records.json --subset 4 --budget 90000 > $R/groq-demo.log 2>&1 ;;
demo-rest)
  # The demo queries the first run could not answer before the daily cap (indices 10-27).
  "$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH="$DEMO/test.sqlite3" \
    --set UPLOAD_DIR="$DEMO/uploads" --set GENERATED_DIR="$DEMO/generated" --set SOFT_CONSTRAINT_MATCHING=true \
    --set ENABLE_FRAME_SEMANTIC_RETRIEVAL=true --set FRAME_SEMANTIC_WEIGHT=1.0 --set FRAME_SEMANTIC_FLOOR=0.18 \
    --set FRAME_SEMANTIC_CEILING=0.32 -- "$PY" $R/run_scripts/groq_reaudit.py collect \
    $Q/groq_queries_demo.json $R/groq-demo-records-rest.json --only $(seq 10 27) --subset 4 \
    --budget 90000 > $R/groq-demo-rest.log 2>&1 ;;
prompt)
  # Full-alias (old) prompt vs reduced prompt on queries Groq planned in the demo run.
  "$PY" $R/run_scripts/groq_reaudit.py prompt $R/groq-demo-records.json $R/groq-prompt-equivalence.json \
    --limit 6 > $R/groq-prompt.log 2>&1 ;;
esac
echo "$1 rc=$?"
