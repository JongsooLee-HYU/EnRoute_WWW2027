#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

CONTEXT = {
    "NYC": "These are Foursquare check-in trajectories from New York City, 2012-2013.",
    "TKY": "These are Foursquare check-in trajectories from Tokyo, Japan, 2012-2013.",
    "CA": "These are Gowalla check-in trajectories from California, USA, 2009-2010.",
}

_ENTRY = re.compile(r"^- \[([\d\-: ]+)\] POI (\d+) \((.*)\)\s*$", re.M)
_CURRENT = re.compile(r"^- (Start|End):\s*\[([\d\-: ]+)\] POI (\d+) \((.*)\)\s*$", re.M)


def poi_categories(paths: List[Path]) -> Dict[str, str]:
    cat: Dict[str, str] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            p = json.loads(line)["prompt"]
            for m in _ENTRY.finditer(p):
                cat.setdefault(m.group(2), m.group(3))
            for m in _CURRENT.finditer(p):
                cat.setdefault(m.group(3), m.group(4))
    return cat


def render_trip(row: dict, cat: Dict[str, str]) -> str:
    cur = {m.group(1): (m.group(2), m.group(3), m.group(4)) for m in _CURRENT.finditer(row["prompt"])}
    start, end = cur["Start"], cur["End"]
    gold = [str(g) for g in row["gold_waypoint_poi_ids"]]
    stops = [("start", start[0], start[1], start[2])]
    stops += [("waypoint", None, g, cat.get(g, "Unknown")) for g in gold]
    stops.append(("destination", end[0], end[1], end[2]))
    t0 = datetime.strptime(start[0], "%Y-%m-%d %H:%M:%S")
    t1 = datetime.strptime(end[0], "%Y-%m-%d %H:%M:%S")
    lines = [f"Trip of user {row['user_id']}: starts {t0.strftime('%A')} {start[0]}, ends {end[0]} "
             f"(duration {(t1 - t0).total_seconds() / 3600.0:.1f} hours, {len(gold)} intermediate waypoints)."]
    for i, (role, ts, pid, c) in enumerate(stops, start=1):
        lines.append(f"  {i}. [{role}] POI id {pid} ({c})" + (f" at {ts}" if ts else ""))
    return "\n".join(lines)


def build_prompt(trip: str, types: List[str], context: str) -> str:
    return (f"{context}\nYour task is to classify the trip below into the following categories: "
            f"{', '.join(types)}.\n\n"
            "Assign every type that the trip clearly involves — usually one to three. Do not assign "
            "a type on weak evidence; if only one applies, return only that one.\n"
            "Use only ids from the list above. If nothing fits, return an empty list.\n\n"
            f"{trip}\n\n"
            "Respond with JSON only, in exactly this shape:\n"
            '{"types": ["TYPE_A", "TYPE_B"], "reason": "<one short sentence>"}')


def parse_response(raw: str, valid: set) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        types = obj.get("types", obj.get("type", [])) or []
        types = [types] if isinstance(types, str) else types
        types = [str(t).strip().upper() for t in types]
        reason = str(obj.get("reason", ""))[:300]
    else:
        types = [t for t in valid if re.search(rf"\b{re.escape(t)}\b", text.upper())]
        reason = ""
    return {"types": [t for t in dict.fromkeys(types) if t in valid], "reason": reason,
            "empty_ok": isinstance(obj, dict) and not types}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(CONTEXT))
    ap.add_argument("--taxonomy", default=str(Path(__file__).resolve().parent / "taxonomy.json"))
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--max_output_tokens", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    types = [t["id"] for t in json.loads(Path(args.taxonomy).read_text())["types"]]
    in_path, out_path = Path(args.input_jsonl), Path(args.output_jsonl)
    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("gold_waypoint_poi_ids")]
    cat = poi_categories(sorted(in_path.parent.glob("*.jsonl")))

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error"):
                done.add(r["trajectory_id"])
    jobs = [(r, build_prompt(render_trip(r, cat), types, CONTEXT[args.dataset]))
            for r in rows if r["trajectory_id"] not in done]
    print(f"{len(rows)} trips | {len(jobs)} to label")
    if args.dry_run:
        if jobs:
            print(jobs[0][1])
        return

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    valid = set(types)

    def label(prompt: str) -> dict:
        err = None
        for attempt in range(args.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=args.model, messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=args.max_output_tokens,
                    response_format={"type": "json_object"})
                raw = resp.choices[0].message.content or ""
                res = parse_response(raw, valid)
                if res["types"] or res["empty_ok"]:
                    return {"types": res["types"], "reason": res["reason"]}
                err = f"no valid type in response: {raw[:200]!r}"
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(min(2 ** attempt + random.random(), 30))
        return {"types": [], "reason": "", "error": err}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock, n_err = threading.Lock(), 0
    with out_path.open("a") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(label, p): r for r, p in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            r, res = futures[fut], fut.result()
            n_err += "error" in res
            with lock:
                f.write(json.dumps({"trajectory_id": r["trajectory_id"], "user_id": r["user_id"], **res},
                                   ensure_ascii=False) + "\n")
                f.flush()
            if i % 500 == 0 or i == len(futures):
                print(f"{i}/{len(futures)} | errors {n_err}", flush=True)


if __name__ == "__main__":
    main()
