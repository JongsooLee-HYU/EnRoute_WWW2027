#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import shapely


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    checkins = pd.read_csv(raw / "loc-gowalla_totalCheckins.txt", sep="\t", header=None)
    checkins.columns = ["userid", "datetime", "checkins_lat", "checkins_lng", "id"]
    spots = pd.read_csv(raw / "gowalla_spots_subset1.csv")
    df = checkins.merge(spots, on="id")
    first_seen = pd.Series(range(len(checkins)), index=checkins["id"]).groupby(level=0).min()
    df = df.iloc[df["id"].map(first_seen).argsort(kind="stable")]

    states = {}
    for feature in json.loads((raw / "us_state_polygon_json.json").read_text())["features"]:
        name = feature["properties"]["name"].lower()
        if name in ("california", "nevada"):
            states[name] = shapely.polygons(feature["geometry"]["coordinates"][0])
    lng, lat = df["checkins_lng"].to_numpy(), df["checkins_lat"].to_numpy()
    inside = shapely.intersects_xy(states["nevada"], lng, lat) | shapely.intersects_xy(states["california"], lng, lat)
    df = df[inside]

    out = df[["userid", "id", "spot_categories", "checkins_lat", "checkins_lng", "datetime"]]
    out.columns = ["UserId", "PoiId", "PoiCategoryId", "Latitude", "Longitude", "UTCTime"]
    out.to_csv(raw / "dataset_gowalla_ca_ne.csv", index=False)
    print(f"{len(out):,} check-ins in CA/NV -> {raw / 'dataset_gowalla_ca_ne.csv'}")


if __name__ == "__main__":
    main()
