# EnRoute

## Setup
- Access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face (`huggingface-cli login`)
- Docker (OSRM travel-time matrix)

```bash
pip install -r requirements.txt
```

## 1. Data preprocessing

Download `datasets.zip` (raw NYC, TKY and CA data) from the LLM4POI repository (https://github.com/neolifer/LLM4POI.git) and unpack it:

```bash
unzip datasets.zip
for c in nyc tky ca; do mkdir -p raw/$c && unzip -j datasets/$c/raw.zip "raw/*" -d raw/$c; done
```

| dataset | `--raw` in step 1 |
|---|---|
| NYC | `raw/nyc` (`NYC_train.csv`, `NYC_val.csv`, `NYC_test.csv`) |
| TKY | `raw/tky/dataset_TSMC2014_TKY.txt` |
| CA | `raw/ca/dataset_gowalla_ca_ne.csv` |

`dataset_gowalla_ca_ne.csv` can also be rebuilt from the Gowalla dumps (`loc-gowalla_totalCheckins.txt`, `gowalla_spots_subset1.csv`, `us_state_polygon_json.json`):

```bash
python preprocess/make_gowalla_ca.py --raw_dir raw/ca
```

The steps below use NYC; replace the city name for the other datasets.

**Step 1. Chronological split** (TKY/CA: infrequent users and POIs removed, 24h trajectory split, 8:1:1 by time; NYC: the publicly released split)
```bash
python preprocess/preprocess.py --city nyc --raw raw/nyc --out_dir data/NYC/work/csv
```

**Step 2. Trips** (origin, waypoints, destination; up to 5 previous trips / 30 check-ins as history)
```bash
python preprocess/build_trips.py --csv_dir data/NYC/work/csv --out_dir data/NYC/work/trips
```

**Step 3. Activity labels** (GPT-5-mini, 12 types in `preprocess/taxonomy.json`)
```bash
export OPENAI_API_KEY=...
for s in train valid; do
  python preprocess/label_activities.py --dataset NYC --input_jsonl data/NYC/work/trips/$s.jsonl \
    --output_jsonl labels/NYC/labels_${s}_multi.jsonl
done
```

**Step 4. Training data** (prompts, waypoint labels, copy memory, activity labels)
```bash
python preprocess/build_dataset.py --csv_dir data/NYC/work/csv --trips_dir data/NYC/work/trips \
  --labels_dir labels/NYC --out_dir data/NYC
```

**Step 5. LightGCN user embeddings** (train user–POI graph)
```bash
python preprocess/lightgcn.py --trips_dir data/NYC/work/trips --vocab data/NYC/main/poi_vocab.json \
  --out data/NYC/user_lgcn_d64.npz
```

**Step 6. OSRM car travel-time matrix**
```bash
bash preprocess/osrm/manage_osrm.sh build nyc car
bash preprocess/osrm/manage_osrm.sh start nyc car
python preprocess/osrm/build_matrix.py --coords preprocess/coords/nyc_poi_coords.json \
  --vocab data/NYC/main/poi_vocab.json --url http://localhost:5000 --profile driving \
  --out data/NYC/route_car.npz
bash preprocess/osrm/manage_osrm.sh stop nyc car
```

Steps 1, 2, 4, 5 and 6 can be run together once the activity labels exist and the OSRM server is up:
```bash
C=NYC RAW=raw/nyc LABELS=labels/NYC COORDS=preprocess/coords/nyc_poi_coords.json \
  bash preprocess/run_preprocess.sh
```

Resulting layout:
```
data/NYC/
  main/        train.jsonl valid.jsonl test.jsonl memory_{train,valid,test}.jsonl poi_vocab.json
  pretrain/    train.jsonl valid.jsonl type_vocab.json
  route_car.npz
  user_lgcn_d64.npz
```

## 2. Training and inference

```bash
C=NYC GPU=0 bash run.sh
```

`run.sh` runs the three training stages, predicts the test split and prints Recall@1/5/20 and NDCG@20. Checkpoints and predictions are written to `outputs/NYC/`.

The same steps one by one:
```bash
D=data/NYC; O=outputs/NYC

# Stage 1: activity pretraining
python pretrain.py --data_dir $D/pretrain --output_dir $O/stage1

# Stage 2: relevance head, temporal copy memory, detour scoring
python train.py --stage 2 --data_dir $D/main --route_matrix $D/route_car.npz \
  --init_lora $O/stage1 --output_dir $O/stage2

# Stage 3: user adapter on the frozen Stage-2 model
python train.py --stage 3 --data_dir $D/main --route_matrix $D/route_car.npz \
  --init_from $O/stage2 --user_emb $D/user_lgcn_d64.npz --output_dir $O/stage3

# Inference and evaluation
python predict.py --data_dir $D/main --route_matrix $D/route_car.npz --user_emb $D/user_lgcn_d64.npz \
  --checkpoint_dir $O/stage3 --output_jsonl $O/test_predictions.jsonl
python evaluate.py $O/test_predictions.jsonl
```

Every stage trains for 3 epochs (4 trips per step, 4 accumulation steps) and validates every 200 updates. Stage 1 keeps the checkpoint with the lowest validation loss; Stages 2 and 3 keep the one with the highest validation NDCG@20.
