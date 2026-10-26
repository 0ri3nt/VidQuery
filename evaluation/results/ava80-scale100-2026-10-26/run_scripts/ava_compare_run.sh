cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"; export PYTHONIOENCODING=utf-8
O=evaluation/results/ava80-scale100-2026-10-26
"$PY" "C:/Users/Aryaman/AppData/Local/Temp/claude/c--Users-Aryaman-Desktop-work-capstone/f7fe97e8-5f76-41ae-a68d-2236af2f6e22/scratchpad/ava_write_decision.py" || exit 1
echo "test_comparison_start $(date -u +%FT%TZ)" >> $O/timeline.txt
"$PY" -m scripts.compare_ava_checkpoints --split evaluation/ava80_scale100_split.json --decision $O/validation-decision.json \
  --checkpoint legacy_gat_24=data/app/models/ava80_noleak/ava80-gat-v7-noleak-lr0.0003.pt \
  --checkpoint legacy_mlp_24=data/app/models/ava80_noleak/ava80-mlp-v7-noleak-lr0.0003.pt \
  --checkpoint stream_gat_24=data/app/models/ava80_shard_stream_check/ava80-gat-v7-noleak-shardstream-lr0.0003.pt \
  --checkpoint stream_mlp_24=data/app/models/ava80_scale100_baselines/ava80-mlp-v7-noleak-shardstream-s30-lr0.0003.pt \
  --checkpoint stream_gat_94=data/app/models/ava80_scale100/ava80-gat-v8-noleak-scale100-lr0.0003.pt \
  --checkpoint stream_mlp_94=data/app/models/ava80_scale100/ava80-mlp-v8-noleak-scale100-lr0.0003.pt \
  --checkpoint active_v5_leaky_reference=data/app/models/ava80_scaling/30/ablations/ava80-gat-v5-scale30-roi-frame.pt \
  --pair stream_gat_24:stream_gat_94 --pair stream_mlp_24:stream_mlp_94 \
  --pair legacy_gat_24:stream_gat_94 --pair legacy_gat_24:stream_mlp_94 --pair stream_gat_94:stream_mlp_94 \
  --pair legacy_mlp_24:stream_mlp_94 \
  --output $O/test-comparison.json > $O/test-comparison.log 2>&1
echo "compare rc=$?"; echo "test_comparison_end $(date -u +%FT%TZ)" >> $O/timeline.txt
