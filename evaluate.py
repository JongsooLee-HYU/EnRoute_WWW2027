#!/usr/bin/env python3
import json
import sys

from metrics import summarize

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
m = summarize(rows, (1, 5, 20), 20)
for k in ("Recall@1", "Recall@5", "Recall@20", "NDCG@20"):
    print(f"{k} {m[k]:.4f}")
print(f"({m['n_trips']} test trips)")
