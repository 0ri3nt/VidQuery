cd /c/Users/Aryaman/Desktop/work/capstone/VidQuery
PY="/c/Program Files/Python312/python.exe"; export PYTHONIOENCODING=utf-8
RES=evaluation/results/charades-integrated-2026-10-26; OLD=evaluation/results/charades-2026-10-25
B=data/app/benchmark/charades_integrated_2026-10-26
for T in T1-selected-soft-frame T2-selected-p1-preserving-thresholds T0-default-strict-noframe; do
  OUT=$RES/comparison-test-$T.json; [ -f $OUT ] && continue
  "$PY" -m scripts.compare_moment_runs \
    --run orig_fallback_ava5_vidor2=$OLD/test/$T.json \
    --run rerun_fallback_ava5_vidor2=$OLD/test-v2relation-rerun/$T.json \
    --run rerun_fallback_ava5_vidor3=$OLD/test-v3relation/$T.json \
    --run det_ava5_vidor2=$RES/det-ava5-vidor2/test-$T.json \
    --run det_ava8_vidor2=$RES/det-ava8-vidor2/test-$T.json \
    --run det_ava5_vidor4=$RES/det-ava5-vidor4/test-$T.json \
    --run det_ava8_vidor4=$RES/det-ava8-vidor4/test-$T.json \
    --pair orig_fallback_ava5_vidor2:rerun_fallback_ava5_vidor2 \
    --pair orig_fallback_ava5_vidor2:det_ava5_vidor2 --pair rerun_fallback_ava5_vidor2:det_ava5_vidor2 \
    --pair det_ava5_vidor2:det_ava8_vidor2 --pair det_ava5_vidor4:det_ava8_vidor4 \
    --pair det_ava5_vidor2:det_ava5_vidor4 --pair det_ava8_vidor2:det_ava8_vidor4 \
    --pair det_ava5_vidor2:det_ava8_vidor4 --pair orig_fallback_ava5_vidor2:det_ava8_vidor4 \
    --pair rerun_fallback_ava5_vidor3:det_ava8_vidor4 \
    --output $OUT > /dev/null 2>$RES/comparison-$T.err; echo "compare $T rc=$?"
done
OUT=$RES/comparison-validation-T1.json
[ -f $OUT ] || "$PY" -m scripts.compare_moment_runs --run orig_fallback_ava5_vidor2=$OLD/validation-selected-C3.json \
  --run det_ava8_vidor4=$RES/det-ava8-vidor4/validation-T1-selected-soft-frame.json \
  --pair orig_fallback_ava5_vidor2:det_ava8_vidor4 --output $OUT > /dev/null 2>$RES/comparison-validation.err; echo "compare validation rc=$?"
mkdir -p $RES/index-evidence
ev () { [ -f $RES/index-evidence/$1.json ] || "$PY" -m scripts.compare_index_evidence $2 $3 --output $RES/index-evidence/$1.json > /dev/null; }
ev rerun-fallback_vs_det-ava5-vidor2 data/app/benchmark/charades_v2_rerun/charades-v2-rerun-snapshot-test.sqlite3 $B/det-ava5-vidor2/index-snapshot.sqlite3
ev det-ava5-vidor2_vs_det-ava8-vidor2 $B/det-ava5-vidor2/index-snapshot.sqlite3 $B/det-ava8-vidor2/index-snapshot.sqlite3
ev det-ava5-vidor2_vs_det-ava5-vidor4 $B/det-ava5-vidor2/index-snapshot.sqlite3 $B/det-ava5-vidor4/index-snapshot.sqlite3
ev det-ava5-vidor2_vs_det-ava8-vidor4 $B/det-ava5-vidor2/index-snapshot.sqlite3 $B/det-ava8-vidor4/index-snapshot.sqlite3
echo evidence done
