#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from build_trips import load_trajectories

SPLITS = ("train", "valid", "test")

_ENTRY = re.compile(r"^- \[([\d\-: ]+)\] POI (\d+) \((.*)\)\s*$", re.M)
_CURRENT = re.compile(r"^- (Start|End):\s*\[([\d\-: ]+)\] POI (\d+) \((.*)\)\s*$", re.M)
_TRAJ_SPLIT = re.compile(r"Trajectory \d+:")


def parse_history_trips(prompt: str) -> List[List[Tuple[str, str, str]]]:
    head = prompt[prompt.find("Historical trajectories"): prompt.find("Current trip")]
    trips = ([m.groups() for m in _ENTRY.finditer(block)] for block in _TRAJ_SPLIT.split(head)[1:])
    return [t for t in trips if len(t) >= 2]


def build_question(row: dict, max_poi_id: int) -> str:
    user = row["user_id"]
    cur = {m.group(1): (m.group(2), m.group(3), m.group(4)) for m in _CURRENT.finditer(row["prompt"])}
    start, end = cur["Start"], cur["End"]
    lines = [
        f"<question>: The following data is a trip of user {user} with a known "
        f"start POI and a known destination POI:",
        f"At {start[0]}, user {user} starts at POI id {start[1]} which is a {start[2]}.",
        f"User {user} will arrive at POI id {end[1]} which is a {end[2]}.",
    ]
    trips = parse_history_trips(row["prompt"])
    if not trips:
        lines.append(f"There is no historical data for user {user}.")
    else:
        lines.append(f"There is historical data for user {user}. Each past trip is shown "
                     f"with its start POI, its intermediate waypoints, and its destination POI:")
        for i, trip in enumerate(trips, start=1):
            first, last, middle = trip[0], trip[-1], trip[1:-1]
            lines.append(f"Trip {i}:")
            lines.append(f"  started at POI id {first[1]} ({first[2]}) on {first[0]}")
            lines.append("  waypoints: " + (", ".join(f"POI id {p} ({c}) on {ts}" for ts, p, c in middle)
                                            if middle else "none"))
            lines.append(f"  arrived at POI id {last[1]} ({last[2]}) on {last[0]}")
    lines.append(f"Given the data, which POI ids will user {user} visit as intermediate "
                 f"waypoints between the start POI and the destination POI? Note that POI id "
                 f"is an integer in the range from 0 to {max_poi_id}.")
    return "\n".join(lines)


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def copy_memory(trips_by_split, p2i: Dict[str, int]):
    visits = defaultdict(list)
    for split in SPLITS:
        for t in trips_by_split[split]:
            visits[t.user_id].extend((_dt(v.time_str), v.poi_id) for v in t.visits)
    out = {}
    for split in SPLITS:
        rows = {}
        for t in trips_by_split[split]:
            t1 = _dt(t.visits[0].time_str)
            cnt, last = defaultdict(int), {}
            for ts, p in visits[t.user_id]:
                if ts < t1:
                    cnt[p] += 1
                    last[p] = max(ts, last.get(p, ts))
            mem = sorted((p2i[p], p) for p in cnt)
            rows[t.tid] = ([i for i, _ in mem], [cnt[p] for _, p in mem],
                           [round((t1 - last[p]).total_seconds() / 86400.0, 6) for _, p in mem])
        out[split] = rows
    return out


def load_activity_labels(path: Path) -> Dict[str, List[str]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("error") and r.get("trajectory_id") is not None:
            out[str(r["trajectory_id"])] = list(r.get("types") or [])
    return out


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True, help="{train,valid,test}.csv from preprocess.py")
    ap.add_argument("--trips_dir", required=True, help="{train,valid,test}.jsonl from build_trips.py")
    ap.add_argument("--labels_dir", required=True, help="labels_{split}_multi.jsonl from label_activities.py")
    ap.add_argument("--taxonomy", default=str(Path(__file__).resolve().parent / "taxonomy.json"))
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    csv_dir, trips_dir, out = Path(args.csv_dir), Path(args.trips_dir), Path(args.out_dir)

    rows = {s: [json.loads(l) for l in open(trips_dir / f"{s}.jsonl")] for s in SPLITS}
    ids = {str(p) for s in SPLITS for r in rows[s]
           for p in [r["start_poi"], r["end_poi"], *r["gold_waypoint_poi_ids"]]}
    vocab = sorted(ids, key=int)
    p2i = {p: i for i, p in enumerate(vocab)}
    (out / "main").mkdir(parents=True, exist_ok=True)
    (out / "main" / "poi_vocab.json").write_text(json.dumps({"poi_ids": vocab}))
    max_poi_id = max(int(p) for p in vocab)

    trips = {"train": load_trajectories(csv_dir / "train.csv"),
             "valid": load_trajectories(csv_dir / "valid.csv"),
             "test": load_trajectories(csv_dir / "test.csv")}
    memory = copy_memory(trips, p2i)

    types = [t["id"] for t in json.loads(Path(args.taxonomy).read_text())["types"]]
    t2i = {t: i for i, t in enumerate(types)}
    (out / "pretrain").mkdir(parents=True, exist_ok=True)
    (out / "pretrain" / "type_vocab.json").write_text(json.dumps({"types": types}))

    for split in SPLITS:
        main_rows, mem_rows, pre_rows = [], [], []
        labels = load_activity_labels(Path(args.labels_dir) / f"labels_{split}_multi.jsonl")
        for r in rows[split]:
            tid = str(r["trajectory_id"])
            label_idx = sorted({p2i[str(g)] for g in r["gold_waypoint_poi_ids"]})
            q = build_question(r, max_poi_id)
            main_rows.append({"trajectory_id": tid, "user_id": r["user_id"], "question": q,
                              "label_indices": label_idx, "gold_poi_ids": [vocab[i] for i in label_idx]})
            idx, cnt, dt = memory[split][tid]
            mem_rows.append({"trajectory_id": tid, "idx": idx, "cnt": cnt, "dt": dt})
            if split != "test":
                act = labels.get(tid)
                type_idx = sorted({t2i[t] for t in act if t in t2i}) if act is not None else []
                pre_rows.append({"trajectory_id": tid, "user_id": r["user_id"], "question": q,
                                 "type_indices": type_idx, "types": [types[i] for i in type_idx],
                                 "has_type": act is not None})
        write_jsonl(out / "main" / f"{split}.jsonl", main_rows)
        write_jsonl(out / "main" / f"memory_{split}.jsonl", mem_rows)
        msg = f"{split}: {len(main_rows)} trips | avg |M_u| {sum(len(m['idx']) for m in mem_rows) / len(mem_rows):.1f}"
        if split != "test":
            write_jsonl(out / "pretrain" / f"{split}.jsonl", pre_rows)
            msg += f" | activity-labelled {sum(r['has_type'] for r in pre_rows)}/{len(pre_rows)}"
        print(msg)
    print(f"POI vocabulary {len(vocab)} | activity types {len(types)} -> {out}")


if __name__ == "__main__":
    main()
