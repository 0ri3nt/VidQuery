# VidOR -> 2,000 videos: staging, scratch + warm-start training (test withheld), graph invariant.
cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONPATH=.
O=evaluation/results/vidor-scale2000-2026-10-26
M=data/app/models/vidor_relation_scale2000
FR=data/source/vidor/selected-frames-2000
stamp () { echo "$1 $(date -u +%FT%TZ)" >> $O/timeline.txt; }
for N in 1 2 3 4 5 6 7 8; do
  [ -f "$O/stage-part$N.done" ] && continue
  for TRY in 1 2 3; do
    stamp "stage_part${N}_start_try$TRY"
    "$PY" scripts/stage_vidor_pair_visual_frames.py --part $N --maximum-selected-videos 2000 \
      --manifest $M/split_manifest.json --frame-root $FR > "$O/stage-part$N-try$TRY.log" 2>&1
    rc=$?; echo "stage part $N try $TRY rc=$rc"; stamp "stage_part${N}_end_try$TRY"
    [ $rc -eq 0 ] && break
    sleep 60
  done
  [ $rc -eq 0 ] && touch "$O/stage-part$N.done" || exit 4
done
"$PY" -c "
import json; m=json.load(open('$FR/split_manifest.json')); s=json.load(open('$M/split_manifest.json'))
print('staged', len(m['selected_videos']), 'of', len(s['selected_videos']), {k: len(m[k]) for k in ('train','validation','test')})
assert m['validation']==s['validation'] and m['test']==s['test'], 'validation/test incomplete after staging'"
[ $? -eq 0 ] || exit 5
if [ ! -f $M/vidor-relation-pair-visual-gat-v4-scale2000.pt ]; then
  stamp scratch_train_start
  "$PY" scripts/train_vidor_pair_visual.py --frame-root $FR --baseline-manifest $FR/split_manifest.json \
    --output-dir $M --version-tag v4-scale2000 --withhold-test > $O/train-scratch.log 2>&1
  echo "scratch rc=$?"; stamp scratch_train_end
fi
"$PY" -m scripts.verify_vidor_invariants graphs \
  --old-cache data/app/models/vidor_relation_scale500/vidor_relation_pair_visual_graphs.pt \
  --new-cache $M/vidor_relation_pair_visual_graphs.pt --output $O/graph-invariant-validation-test.json > $O/graph-invariant.log 2>&1
echo "graph invariant rc=$?"
W=data/app/models/vidor_relation_scale2000_warm
if [ ! -f $W/vidor-relation-pair-visual-gat-v4-scale2000-warm.pt ]; then
  stamp warm_train_start
  "$PY" scripts/train_vidor_pair_visual.py --frame-root $FR --baseline-manifest $FR/split_manifest.json \
    --output-dir $W --version-tag v4-scale2000-warm --withhold-test \
    --graph-cache $M/vidor_relation_pair_visual_graphs.pt \
    --init-gnn-checkpoint data/app/models/vidor_relation_scale500/vidor-relation-pair-visual-gat-v3-scale500.pt \
    --init-mlp-checkpoint data/app/models/vidor_relation_scale500/vidor-relation-pair-visual-mlp-v3-scale500.pt \
    > $O/train-warm.log 2>&1
  echo "warm rc=$?"; stamp warm_train_end
fi
