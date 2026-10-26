cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"; export PYTHONIOENCODING=utf-8
O=evaluation/results/vidor-scale2000-2026-10-26
"$PY" "C:/Users/Aryaman/AppData/Local/Temp/claude/c--Users-Aryaman-Desktop-work-capstone/f7fe97e8-5f76-41ae-a68d-2236af2f6e22/scratchpad/vidor_write_decision.py" || exit 1
echo "test_comparison_start $(date -u +%FT%TZ)" >> $O/timeline.txt
A=data/app/models
PAIRS=""; for p in gat_v2:gat_v4_scale2000 gat_v2:gat_v4_scale2000_warm gat_v3_scale500:gat_v4_scale2000 gat_v3_scale500:gat_v4_scale2000_warm mlp_v3_scale500:mlp_v4_scale2000 mlp_v3_scale500:mlp_v4_scale2000_warm gat_v4_scale2000:gat_v4_scale2000_warm mlp_v4_scale2000:mlp_v4_scale2000_warm gat_v4_scale2000:mlp_v4_scale2000 gat_v2:gat_v3_scale500; do PAIRS="$PAIRS --pair $p"; done
"$PY" -m scripts.compare_vidor_checkpoints --decision $O/validation-decision.json \
  --checkpoint gat_v2=$A/vidor_relation_pair_visual/vidor-relation-pair-visual-gat-v2.pt \
  --checkpoint mlp_v2=$A/vidor_relation_pair_visual/vidor-relation-pair-visual-mlp-v2.pt \
  --checkpoint gat_v3_scale500=$A/vidor_relation_scale500/vidor-relation-pair-visual-gat-v3-scale500.pt \
  --checkpoint mlp_v3_scale500=$A/vidor_relation_scale500/vidor-relation-pair-visual-mlp-v3-scale500.pt \
  --checkpoint gat_v4_scale2000=$A/vidor_relation_scale2000/vidor-relation-pair-visual-gat-v4-scale2000.pt \
  --checkpoint mlp_v4_scale2000=$A/vidor_relation_scale2000/vidor-relation-pair-visual-mlp-v4-scale2000.pt \
  --checkpoint gat_v4_scale2000_warm=$A/vidor_relation_scale2000_warm/vidor-relation-pair-visual-gat-v4-scale2000-warm.pt \
  --checkpoint mlp_v4_scale2000_warm=$A/vidor_relation_scale2000_warm/vidor-relation-pair-visual-mlp-v4-scale2000-warm.pt \
  $PAIRS --output $O/test-comparison.json > $O/test-comparison.log 2>&1
echo "compare rc=$?"; echo "test_comparison_end $(date -u +%FT%TZ)" >> $O/timeline.txt
