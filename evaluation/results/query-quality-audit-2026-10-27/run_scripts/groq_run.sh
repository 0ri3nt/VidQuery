# Groq audit collection, paced under the account's 8,000 tokens/minute limit.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
R=evaluation/results/query-quality-audit-2026-10-27
DEMO="C:/Users/Aryaman/AppData/Local/Temp/claude/c--Users-Aryaman-Desktop-work-capstone/f7fe97e8-5f76-41ae-a68d-2236af2f6e22/scratchpad/live/library_a"
"$PY" -m scripts.run_with_profile --profile det-ava8-vidor4 --set DATABASE_PATH="$DEMO/test.sqlite3" \
  --set UPLOAD_DIR="$DEMO/uploads" --set GENERATED_DIR="$DEMO/generated" --set SOFT_CONSTRAINT_MATCHING=true \
  --set ENABLE_FRAME_SEMANTIC_RETRIEVAL=true --set FRAME_SEMANTIC_WEIGHT=1.0 --set FRAME_SEMANTIC_FLOOR=0.18 \
  --set FRAME_SEMANTIC_CEILING=0.32 -- "$PY" $R/run_scripts/groq_audit.py collect --pause 40 \
  --queries $R/groq_queries_demo.json --output $R/groq-demo-records.json > $R/groq-demo.log 2>&1
echo "demo rc=$?"
# Production library copy with the production defaults from .env (no experimental flags).
DATABASE_PATH=data/app/benchmark/query_audit_2026-10-27/production-copy.sqlite3 \
  "$PY" $R/run_scripts/groq_audit.py collect --pause 40 --queries $R/groq_queries_production.json \
  --output $R/groq-production-records.json > $R/groq-production.log 2>&1
echo "production rc=$?"
