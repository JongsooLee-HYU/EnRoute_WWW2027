#!/usr/bin/env bash
set -euo pipefail
C=${C:?set C to NYC, TKY or CA}; GPU=${GPU:-0}; PY=${PY:-python}
cd "$(dirname "$0")"
DATA=${DATA:-data}
export CUDA_VISIBLE_DEVICES=$GPU
D=$DATA/$C; O=outputs/$C; mkdir -p $O

for f in main/poi_vocab.json pretrain/type_vocab.json route_car.npz user_lgcn_d64.npz \
         pretrain/{train,valid}.jsonl main/{train,valid,test}.jsonl main/memory_{train,valid,test}.jsonl; do
  [ -f $D/$f ] || { echo "missing $D/$f"; exit 1; }
done

$PY pretrain.py --data_dir $D/pretrain --output_dir $O/stage1

$PY train.py --stage 2 --data_dir $D/main --route_matrix $D/route_car.npz \
  --init_lora $O/stage1 --output_dir $O/stage2

$PY train.py --stage 3 --data_dir $D/main --route_matrix $D/route_car.npz \
  --init_from $O/stage2 --user_emb $D/user_lgcn_d64.npz --output_dir $O/stage3

$PY predict.py --data_dir $D/main --route_matrix $D/route_car.npz --user_emb $D/user_lgcn_d64.npz \
  --checkpoint_dir $O/stage3 --output_jsonl $O/test_predictions.jsonl
$PY evaluate.py $O/test_predictions.jsonl
