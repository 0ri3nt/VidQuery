# AVA -> 100 videos: expansion, split, training (test withheld). Resumable; each step skips if done.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
O=evaluation/results/ava80-scale100-2026-10-26
stamp () { echo "$1 $(date -u +%FT%TZ)" >> $O/timeline.txt; }
stamp expansion_start
"$PY" -m vidquery expand-ava-dataset --target-videos 100 --batch-size 2 --workers 2 \
  --shard-dir data/app/features/ava80_shards --no-legacy-cache --min-free-gb 8 \
  --temporary-root data/temp/ava_shard_scale100 \
  --state data/app/features/ava80_shards/expansion_state_scale100.json \
  --selection-manifest $O/expansion-selection.json > $O/expansion.log 2>&1
echo "expansion rc=$?"; stamp expansion_end
COUNT=$("$PY" -c "from pathlib import Path; from vidquery.ava_shards import completed_shard_video_ids as c; print(len(c(Path('data/app/features/ava80_shards'))))")
echo "training shards: $COUNT"
if [ "$COUNT" -lt 100 ]; then echo "STOP: target not reached; see $O/expansion.log"; exit 3; fi
[ -f evaluation/ava80_scale100_split.json ] || "$PY" -m scripts.build_ava_scaling_split \
  --extra-test 2PpxiG0WU18 NO2esmws190 uNT6HrrnqPU --output evaluation/ava80_scale100_split.json
train () { # output-dir model-version split extra-args...
  [ -f "$1/$2.pt" ] && return
  stamp "train_$2_start"
  "$PY" -m scripts.train_ava_from_shards --split "$3" --output-dir "$1" --model-version "$2" \
    --learning-rate 0.0003 --epochs 60 --patience 10 --batch-size 32 --seed 2026 "${@:4}" > "$O/train-$2.log" 2>&1
  echo "train $2 rc=$?"; stamp "train_$2_end"
}
train data/app/models/ava80_scale100_baselines ava80-mlp-v7-noleak-shardstream-s30-lr0.0003 \
  data/app/models/ava80_scaling/30/split_manifest.json --mlp-only
train data/app/models/ava80_scale100 ava80-gat-v8-noleak-scale100-lr0.0003 \
  evaluation/ava80_scale100_split.json --test-shard-dir data/app/features/ava80_eval_shards
train data/app/models/ava80_scale100 ava80-mlp-v8-noleak-scale100-lr0.0003 \
  evaluation/ava80_scale100_split.json --test-shard-dir data/app/features/ava80_eval_shards --mlp-only
stamp all_training_end
