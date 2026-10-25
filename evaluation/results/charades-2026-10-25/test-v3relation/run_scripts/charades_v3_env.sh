cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
B=data/app/benchmark/charades_v3
export DATABASE_PATH=$B/charades-v3.sqlite3 UPLOAD_DIR=$B/uploads GENERATED_DIR=$B/generated
export ENABLE_NEO4J=false PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="/c/Program Files/Python312/python.exe"
# Relation model: v3-scale500 (pinned explicitly; identical to the promoted .env values).
export RELATION_GNN_CHECKPOINT=data/app/models/vidor_relation_scale500/vidor-relation-pair-visual-gat-v3-scale500.pt
export RELATION_GNN_METADATA=data/app/models/vidor_relation_scale500/vidor-relation-pair-visual-gat-v3-scale500.metadata.json
export RELATION_GNN_MODEL_VERSION=vidor-relation-pair-visual-gat-v3-scale500
