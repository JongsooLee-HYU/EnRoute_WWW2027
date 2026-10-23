#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

MIN_TRAJ_LEN = 3
MAX_HISTORY_TRAJ = 5
MAX_HISTORY_VISITS = 30
INSTRUCTION = (
    "You are a waypoint recommendation model.\n"
    "Given the user's historical trajectories and the current trip, "
    "predict the intermediate waypoint POI IDs between the start POI and the end POI.\n"
    "Output JSON only:\n"
    '{"waypoint_poi_ids": [...]}'
)


@dataclass
class Visit:
    poi_id: str
    category: str
    ts: int
    time_str: str


@dataclass
class Trajectory:
    tid: str
    user_id: str
    visits: List[Visit]

    @property
    def start_ts(self) -> int:
        return self.visits[0].ts


def load_trajectories(path: Path) -> List[Trajectory]:
    df = pd.read_csv(path)
    for col in ("PoiId", "UserId", "pseudo_session_trajectory_id"):
        df[col] = df[col].astype(str)
    df = df.sort_values(["UserId", "pseudo_session_trajectory_id", "UTCTimeOffsetEpoch", "check_ins_id"])
    trajs = []
    for tid, g in df.groupby("pseudo_session_trajectory_id", sort=False):
        g = g.sort_values(["UTCTimeOffsetEpoch", "check_ins_id"])
        visits = [Visit(str(r.PoiId), str(r.PoiCategoryName), int(r.UTCTimeOffsetEpoch), str(r.UTCTimeOffset))
                  for r in g.itertuples(index=False)]
        if len(visits) >= MIN_TRAJ_LEN:
            trajs.append(Trajectory(str(tid), str(g.iloc[0]["UserId"]), visits))
    trajs.sort(key=lambda t: (t.user_id, t.start_ts, t.tid))
    return trajs


def format_visit(v: Visit) -> str:
    return f"- [{v.time_str}] POI {v.poi_id} ({v.category})"


def history_text(prior: List[Trajectory]) -> str:
    remaining, segments = MAX_HISTORY_VISITS, []
    for t in reversed(prior):
        if remaining <= 0:
            break
        chosen = t.visits[-remaining:]
        segments.append(chosen)
        remaining -= len(chosen)
    if not segments:
        return "No prior trajectory history is available for this user."
    segments.reverse()
    return "\n\n".join("\n".join([f"Trajectory {i}:"] + [format_visit(v) for v in seg])
                       for i, seg in enumerate(segments, start=1))


def build_rows(trips: List[Trajectory], history: Dict[str, List[Trajectory]]) -> List[dict]:
    rows = []
    for t in trips:
        key = (t.start_ts, t.tid)
        prior = [h for h in history.get(t.user_id, []) if (h.start_ts, h.tid) < key][-MAX_HISTORY_TRAJ:]
        start = format_visit(t.visits[0]).replace("- ", "- Start: ", 1)
        end = format_visit(t.visits[-1]).replace("- ", "- End:   ", 1)
        prompt = (f"Instruction:\n{INSTRUCTION}\n\n"
                  f"Historical trajectories of user {t.user_id}:\n\n{history_text(prior)}\n\n"
                  f"Current trip of user {t.user_id}:\n{start}\n{end}")
        rows.append({"trajectory_id": t.tid, "user_id": t.user_id,
                     "start_poi": t.visits[0].poi_id, "end_poi": t.visits[-1].poi_id,
                     "gold_waypoint_poi_ids": [v.poi_id for v in t.visits[1:-1]],
                     "prompt": prompt})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    csv_dir, out = Path(args.csv_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trips = {s: load_trajectories(csv_dir / f"{s}.csv") for s in ("train", "valid", "test")}
    history = defaultdict(list)
    for split_trips in trips.values():
        for t in split_trips:
            history[t.user_id].append(t)
    for h in history.values():
        h.sort(key=lambda t: (t.start_ts, t.tid))
    for split in ("train", "valid", "test"):
        rows = build_rows(trips[split], history)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} trips -> {out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
