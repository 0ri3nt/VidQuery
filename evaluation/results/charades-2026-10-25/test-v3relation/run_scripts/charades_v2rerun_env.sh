cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
B=data/app/benchmark/charades_v2_rerun
export DATABASE_PATH=$B/charades-v2-rerun.sqlite3 UPLOAD_DIR=$B/uploads GENERATED_DIR=$B/generated
export ENABLE_NEO4J=false PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="/c/Program Files/Python312/python.exe"
# Control: original v2 relation model, fresh rebuild to measure run-to-run noise.
export RELATION_GNN_CHECKPOINT=data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt
export RELATION_GNN_METADATA=data/app/models/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.metadata.json
export RELATION_GNN_MODEL_VERSION=vidor-relation-pair-visual-gat-v2
