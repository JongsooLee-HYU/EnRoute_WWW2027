#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_GAP_MIN = 1440
MIN_FREQ = 9
CATEGORY_NAME = r"'name':\s*'([^']*)'"
KEEP_COLS = ["check_ins_id", "UTCTimeOffset", "UTCTimeOffsetEpoch", "pseudo_session_trajectory_id",
             "UserId", "Latitude", "Longitude", "PoiId", "PoiCategoryId", "PoiCategoryName"]


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["UTCTimeOffsetEpoch"] = (df["UTCTimeOffset"].dt.tz_localize(None)
                                .astype("int64") // 10**9).astype(str)
    df["UTCTimeOffsetWeekday"] = df["UTCTimeOffset"].dt.weekday
    df["UTCTimeOffsetHour"] = df["UTCTimeOffset"].dt.hour
    df["UserRank"] = df.groupby("UserId")["UTCTimeOffset"].rank(method="first")
    return df


def read_tky(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="latin-1", header=None)
    df.columns = ["UserId", "PoiId", "PoiCategoryId", "PoiCategoryName", "Latitude", "Longitude",
                  "TimezoneOffset", "UTCTime"]
    df["UTCTime"] = pd.to_datetime(df["UTCTime"], format="%a %b %d %H:%M:%S +0000 %Y")
    df["UTCTimeOffset"] = df["UTCTime"] + pd.to_timedelta(df["TimezoneOffset"], unit="m")
    return add_time_columns(df)


def read_ca(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTime"], format="%Y-%m-%dT%H:%M:%SZ")
    cell = df["PoiCategoryId"].astype(str)
    df["PoiCategoryName"] = cell.str.extract(CATEGORY_NAME, expand=False).fillna(cell)
    return add_time_columns(df)


def do_filter(df: pd.DataFrame) -> pd.DataFrame:
    poi_count = df.groupby("PoiId")["UserId"].count()
    df = df[df["PoiId"].isin(poi_count[poi_count > MIN_FREQ].index)]
    user_count = df.groupby("UserId")["PoiId"].count()
    return df[df["UserId"].isin(user_count[user_count > MIN_FREQ].index)]


def split_train_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["UserId", "UTCTimeOffset"], ascending=True)
    df["SplitTag"] = "train"
    n = df.shape[0]
    df = df.sort_values(by="UTCTimeOffset", ascending=True)
    df.loc[df.index[int(n * 0.8):int(n * 0.9)], "SplitTag"] = "validation"
    df.loc[df.index[int(n * 0.9):], "SplitTag"] = "test"
    df["UserRank"] = df.groupby("UserId")["UTCTimeOffset"].rank(method="first")
    df = df.sort_values(by=["UserId", "UTCTimeOffset"], ascending=True)
    t, u = df["UTCTimeOffset"], df["UserId"]
    far_prev = t.diff(1).abs().dt.total_seconds().gt(86400)
    far_next = t.diff(-1).abs().dt.total_seconds().gt(86400)
    same_prev, same_next = u.eq(u.shift(1)), u.eq(u.shift(-1))
    isolated = ((far_prev & far_next & same_prev & same_next)
                | (far_next & same_next & ~same_prev)
                | (far_prev & same_prev & ~same_next))
    return df[~isolated.to_numpy()]


def label_encode(fit: pd.Series, values: pd.Series, one_based: bool) -> np.ndarray:
    classes = np.unique(fit.to_numpy())
    pos = np.searchsorted(classes, values.to_numpy())
    pos_c = np.clip(pos, 0, len(classes) - 1)
    seen = classes[pos_c] == values.to_numpy()
    if one_based:
        return np.where(seen, pos_c + 1, 0)
    return np.where(seen, pos_c, len(classes))


def drop_single_checkin_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    size = df.groupby("pseudo_session_trajectory_id")["UTCTimeOffset"].transform("count")
    df.loc[size == 1, "SplitTag"] = "ignore"
    return df


def generate_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["UserId", "UTCTimeOffset"], ascending=True)
    gap_min = df["UTCTimeOffset"].diff().dt.total_seconds() / 60
    new_traj = df["UserId"].ne(df["UserId"].shift()) | gap_min.gt(SESSION_GAP_MIN)
    new_traj.iloc[0] = False
    train = df[df["SplitTag"] == "train"]
    for col in ("PoiId", "PoiCategoryId", "UserId"):
        df[col] = label_encode(train[col], df[col], one_based=True)
    df["check_ins_id"] = df["UTCTimeOffset"].rank(ascending=True, method="first") - 1
    df["pseudo_session_trajectory_id"] = new_traj.astype(int).cumsum().to_numpy()
    return drop_single_checkin_trajectories(df)


def preprocess_tky_ca(city: str, raw: str) -> pd.DataFrame:
    df = read_tky(raw) if city == "tky" else read_ca(raw)
    rounds = 2 if city == "ca" else 1
    for _ in range(rounds):
        df = split_train_test(do_filter(df))
    return generate_id(df)


def preprocess_nyc(raw_dir: str) -> pd.DataFrame:
    parts = []
    for name, tag in (("train", "train"), ("val", "validation"), ("test", "test")):
        p = pd.read_csv(Path(raw_dir) / f"NYC_{name}.csv")
        p["SplitTag"] = tag
        parts.append(p)
    df = pd.concat(parts)
    df.columns = ["UserId", "PoiId", "PoiCategoryId", "PoiCategoryCode", "PoiCategoryName", "Latitude", "Longitude",
                  "TimezoneOffset", "UTCTime", "UTCTimeOffset", "UTCTimeOffsetWeekday", "UTCTimeOffsetNormInDayTime",
                  "pseudo_session_trajectory_id", "UTCTimeOffsetNormDayShift", "UTCTimeOffsetNormRelativeTime", "SplitTag"]
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"].str[:19], format="%Y-%m-%d %H:%M:%S")
    df = add_time_columns(df)
    df = df.sort_values(by=["UserId", "UTCTimeOffset"], ascending=True)
    df["check_ins_id"] = df["UTCTimeOffset"].rank(ascending=True, method="first") - 1
    train = df[df["SplitTag"] == "train"]
    for col in ("PoiId", "PoiCategoryId", "UserId"):
        df[col] = label_encode(train[col], df[col], one_based=False)
    return drop_single_checkin_trajectories(df)


def poi_coordinates(df: pd.DataFrame) -> dict:
    out = {}
    for pid, g in df.groupby("PoiId"):
        (lat, lon), _ = Counter(zip(g["Latitude"].round(7), g["Longitude"].round(7))).most_common(1)[0]
        out[str(pid)] = {"lat": float(lat), "lon": float(lon)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["nyc", "tky", "ca"], required=True)
    ap.add_argument("--raw", required=True, help="raw file (tky, ca) or directory with NYC_{train,val,test}.csv (nyc)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = preprocess_nyc(args.raw) if args.city == "nyc" else preprocess_tky_ca(args.city, args.raw)
    df["UTCTimeOffset"] = df["UTCTimeOffset"].dt.strftime("%Y-%m-%d %H:%M:%S")
    train = df[df["SplitTag"] == "train"]
    seen_users, seen_pois = set(train["UserId"]), set(train["PoiId"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tag, name in (("train", "train"), ("validation", "valid"), ("test", "test")):
        part = df[df["SplitTag"] == tag]
        if tag != "train":
            part = part[part["UserId"].isin(seen_users) & part["PoiId"].isin(seen_pois)]
        part[KEEP_COLS].to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(part):,} check-ins | {part.UserId.nunique():,} users | {part.PoiId.nunique():,} POIs "
              f"| {part.pseudo_session_trajectory_id.nunique():,} trajectories")
    coords = poi_coordinates(df[df["SplitTag"] != "ignore"])
    (out / "poi_coords.json").write_text(json.dumps(coords))
    print(f"coordinates for {len(coords):,} POIs -> {out / 'poi_coords.json'}")


if __name__ == "__main__":
    main()
