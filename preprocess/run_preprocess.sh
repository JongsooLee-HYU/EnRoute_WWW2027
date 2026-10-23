#!/usr/bin/env bash
set -euo pipefail
C=${C:?set C to NYC, TKY or CA}; PY=${PY:-python}; OUT=${OUT:-data}
LABELS=${LABELS:?set LABELS to the activity label directory}
P=$(cd "$(dirname "$0")" && pwd)
W=$OUT/$C/work; mkdir -p $W

if [ -z "${CSV:-}" ]; then
  $PY $P/preprocess.py --city $(echo $C | tr A-Z a-z) --raw ${RAW:?set RAW or CSV} --out_dir $W/csv
  CSV=$W/csv
fi

$PY $P/build_trips.py --csv_dir $CSV --out_dir $W/trips

$PY $P/build_dataset.py --csv_dir $CSV --trips_dir $W/trips --labels_dir $LABELS --out_dir $OUT/$C

[ -f $OUT/$C/user_lgcn_d64.npz ] || $PY $P/lightgcn.py --trips_dir $W/trips \
  --vocab $OUT/$C/main/poi_vocab.json --out $OUT/$C/user_lgcn_d64.npz

if [ ! -f $OUT/$C/route_car.npz ]; then
  $PY $P/osrm/build_matrix.py --coords ${COORDS:?set COORDS} --vocab $OUT/$C/main/poi_vocab.json \
    --url ${OSRM_URL:-http://localhost:5000} --profile driving --out $OUT/$C/route_car.npz
fi
