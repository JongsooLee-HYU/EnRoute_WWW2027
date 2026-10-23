#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

BLOCK = 200


def fetch_block(url, profile, coords, src_idx, dst_idx, timeout=120, retries=3):
    pts = [coords[i] for i in src_idx] + [coords[j] for j in dst_idx]
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in pts)
    params = {
        "sources": ";".join(str(k) for k in range(len(src_idx))),
        "destinations": ";".join(str(len(src_idx) + k) for k in range(len(dst_idx))),
        "annotations": "duration,distance",
    }
    for attempt in range(retries):
        try:
            r = requests.get(f"{url}/table/v1/{profile}/{coord_str}",
                             params=params, timeout=timeout)
            payload = r.json()
            if payload.get("code") == "Ok":
                return payload
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"/table failed for block {src_idx[0]}..{src_idx[-1]} x "
                       f"{dst_idx[0]}..{dst_idx[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--profile", required=True, help="driving | foot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    vocab = json.loads(Path(args.vocab).read_text())["poi_ids"]
    coord_map = json.loads(Path(args.coords).read_text())
    n = len(vocab)

    have = [i for i, p in enumerate(vocab) if str(p) in coord_map]
    coords = {i: (coord_map[str(vocab[i])]["lat"], coord_map[str(vocab[i])]["lon"])
              for i in have}
    print(f"{n} POIs, {len(have)} with coordinates ({n - len(have)} missing -> NaN)")

    dur = np.full((n, n), np.nan, dtype=np.float32)
    dist = np.full((n, n), np.nan, dtype=np.float32)

    blocks = [have[k:k + BLOCK] for k in range(0, len(have), BLOCK)]
    jobs = [(bi, bj) for bi in range(len(blocks)) for bj in range(len(blocks))]
    print(f"{len(blocks)}x{len(blocks)} blocks = {len(jobs)} /table requests")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_block, args.url, args.profile, coords,
                          blocks[bi], blocks[bj]): (bi, bj) for bi, bj in jobs}
        for fut in as_completed(futs):
            bi, bj = futs[fut]
            payload = fut.result()
            src, dst = blocks[bi], blocks[bj]
            d = np.array(payload["durations"], dtype=np.float64)
            m = np.array(payload["distances"], dtype=np.float64)
            dur[np.ix_(src, dst)] = (d / 60.0).astype(np.float32)
            dist[np.ix_(src, dst)] = m.astype(np.float32)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} blocks | {time.time() - t0:.0f}s",
                      flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, duration_min=dur, distance_m=dist,
                        poi_ids=np.array(vocab))
    finite = np.isfinite(dur).mean()
    print(f"saved {out} | finite cells {finite:.4f} | "
          f"median duration {np.nanmedian(dur):.1f} min")


if __name__ == "__main__":
    main()
