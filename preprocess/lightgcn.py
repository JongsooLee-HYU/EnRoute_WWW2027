#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips_dir", required=True, help="build_trips.py output (train.jsonl is used)")
    ap.add_argument("--vocab", required=True, help="poi_vocab.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    vocab = json.loads(Path(a.vocab).read_text())["poi_ids"]
    poi2i = {str(p): i for i, p in enumerate(vocab)}
    inter = defaultdict(set)
    for line in open(Path(a.trips_dir) / "train.jsonl"):
        r = json.loads(line)
        for p in [r["start_poi"], *r["gold_waypoint_poi_ids"], r["end_poi"]]:
            if str(p) in poi2i:
                inter[str(r["user_id"])].add(poi2i[str(p)])
    users = sorted(inter, key=lambda x: int(x) if x.isdigit() else 0)
    u2i = {u: i for i, u in enumerate(users)}
    nU, nI = len(users), len(vocab)
    pairs = np.array([(u2i[u], p) for u in users for p in sorted(inter[u])], np.int64)
    print(f"{nU} users, {nI} POIs, {len(pairs)} interactions, dim {a.dim}, L={a.layers}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = torch.from_numpy(pairs[:, 0])
    c = torch.from_numpy(pairs[:, 1]) + nU
    idx = torch.cat([torch.stack([r, c]), torch.stack([c, r])], 1)
    deg = torch.zeros(nU + nI).index_add_(0, idx[0], torch.ones(idx.shape[1]))
    w = (deg[idx[0]] * deg[idx[1]]).rsqrt()
    A = torch.sparse_coo_tensor(idx, w, (nU + nI, nU + nI)).coalesce().to(dev)
    E0 = torch.nn.Parameter(torch.randn(nU + nI, a.dim, device=dev) * 0.1)
    opt = torch.optim.Adam([E0], lr=a.lr)
    P = torch.from_numpy(pairs).to(dev)
    n = len(P)

    def propagate():
        es, e = [E0], E0
        for _ in range(a.layers):
            e = torch.sparse.mm(A, e)
            es.append(e)
        return torch.stack(es).mean(0)

    for ep in range(a.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, a.batch_size):
            b = P[perm[i:i + a.batch_size]]
            neg = torch.randint(0, nI, (len(b),), device=dev)
            E = propagate()
            U, V = E[:nU], E[nU:]
            pu, pi, ni = U[b[:, 0]], V[b[:, 1]], V[neg]
            x = (pu * pi).sum(1) - (pu * ni).sum(1)
            reg = a.reg * (E0[b[:, 0]].pow(2).sum(1) + E0[nU + b[:, 1]].pow(2).sum(1)
                           + E0[nU + neg].pow(2).sum(1)).mean() / 2
            loss = -F.logsigmoid(x).mean() + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep} BPR loss {tot / n:.4f}", flush=True)

    with torch.no_grad():
        E = propagate()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, user_ids=np.array(users), U=E[:nU].cpu().numpy().astype(np.float32),
                        V=E[nU:].cpu().numpy().astype(np.float32))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
