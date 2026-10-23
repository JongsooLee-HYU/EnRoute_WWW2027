from __future__ import annotations

import math
from typing import List, Sequence, Set


def dedupe(seq: Sequence) -> List[str]:
    out, seen = [], set()
    for x in seq:
        s = str(x)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def recall_at_k(ranked: List[str], gold: Set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def ndcg(ranked: List[str], gold: Set[str], cutoff: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:cutoff], start=1) if p in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), cutoff) + 1))
    return dcg / idcg


def summarize(rows: List[dict], ks: Sequence[int] = (1, 5, 20), cutoff: int = 20) -> dict:
    n = len(rows)
    out = {f"Recall@{k}": 0.0 for k in ks}
    out[f"NDCG@{cutoff}"] = 0.0
    for r in rows:
        gold, ranked = set(dedupe(r["gold"])), dedupe(r["ranked_predictions"])
        for k in ks:
            out[f"Recall@{k}"] += recall_at_k(ranked, gold, k) / n
        out[f"NDCG@{cutoff}"] += ndcg(ranked, gold, cutoff) / n
    out["n_trips"] = n
    return out
