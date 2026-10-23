#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data import (PoiVocab, WaypointCollator, WaypointDataset, load_route_matrix,
                  load_user_embeddings)
from model import EnRoute


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--route_matrix", required=True)
    ap.add_argument("--user_emb", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    vocab = PoiVocab.load(data_dir / "poi_vocab.json")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    route_T = load_route_matrix(args.route_matrix)
    user2row, user_E = load_user_embeddings(args.user_emb)
    model = EnRoute.load(args.checkpoint_dir, args.base_model, route_T, user_E, device)
    print(f"Loaded {args.checkpoint_dir}: {model.config()}")

    ds = WaypointDataset(data_dir / f"{args.split}.jsonl", tok, args.max_length, vocab, user2row)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=WaypointCollator(tok.pad_token_id, len(vocab)))
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(out_path.suffix + ".args.json").write_text(
        json.dumps({**vars(args), "n_poi": len(vocab)}, indent=2))

    n = 0
    with out_path.open("w") as fout, torch.no_grad():
        for batch in tqdm(loader, desc=args.split):
            logits, g = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                              batch["memory"].to(device), batch["se"].to(device),
                              batch["user_idx"].to(device))
            top = logits.topk(args.top_n, dim=-1)
            for idxs, scores, gate, gold in zip(top.indices.tolist(), top.values.tolist(),
                                                g.squeeze(1).tolist(), batch["gold"]):
                row = ds.rows[n]
                fout.write(json.dumps({
                    "query_id": str(row.get("trajectory_id")),
                    "trajectory_id": row.get("trajectory_id"),
                    "user_id": row.get("user_id"),
                    "scope": "trajectory", "step_index": None,
                    "gold": gold,
                    "ranked_predictions": vocab.decode(idxs),
                    "scores": [round(s, 4) for s in scores],
                    "gate": round(gate, 4)}) + "\n")
                n += 1
    print(f"Wrote {n} queries -> {out_path}")


if __name__ == "__main__":
    main()
