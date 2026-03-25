from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class SplitTrajectory:
    cid: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    edge_path: List[str]
    node_path: List[int]
    travel_time_min: float
    path_key: str
    od_key: str
    depart_slot: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _roadid_to_nodes(roadid: str) -> Optional[Tuple[int, int]]:
    parts = str(roadid).split("_")
    if len(parts) != 2:
        return None
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None


def _to_depart_slot(ts: pd.Timestamp, freq_min: int) -> str:
    floored = ts.floor(f"{int(freq_min)}min")
    return str(floored)


def _finalize_segment(
    cid: str,
    points: List[Tuple[pd.Timestamp, str]],
    freq_min: int,
) -> Optional[SplitTrajectory]:
    if len(points) < 2:
        return None
    edge_path = [str(e) for _, e in points]
    node_path: List[int] = []
    for i, e in enumerate(edge_path):
        uv = _roadid_to_nodes(e)
        if uv is None:
            return None
        u, v = uv
        if i == 0:
            node_path.append(u)
        node_path.append(v)
    start_time = points[0][0]
    end_time = points[-1][0]
    travel_time_min = (end_time - start_time).total_seconds() / 60.0
    if travel_time_min <= 0:
        return None
    path_key = "|".join(edge_path)
    od_key = f"{node_path[0]}-{node_path[-1]}"
    return SplitTrajectory(
        cid=cid,
        start_time=start_time,
        end_time=end_time,
        edge_path=edge_path,
        node_path=node_path,
        travel_time_min=float(travel_time_min),
        path_key=path_key,
        od_key=od_key,
        depart_slot=_to_depart_slot(start_time, freq_min),
    )


def split_vehicle_trajectory_no_loop(
    points: List[Tuple[pd.Timestamp, str]],
    cid: str,
    freq_min: int,
) -> List[SplitTrajectory]:
    out: List[SplitTrajectory] = []
    seg: List[Tuple[pd.Timestamp, str]] = []
    used_edges: set[str] = set()
    for ts, edge in points:
        edge_s = str(edge)
        if not seg:
            seg = [(ts, edge_s)]
            used_edges = {edge_s}
            continue
        if edge_s in used_edges:
            item = _finalize_segment(cid=cid, points=seg, freq_min=freq_min)
            if item is not None:
                out.append(item)
            seg = [(ts, edge_s)]
            used_edges = {edge_s}
            continue
        seg.append((ts, edge_s))
        used_edges.add(edge_s)
    item = _finalize_segment(cid=cid, points=seg, freq_min=freq_min)
    if item is not None:
        out.append(item)
    return out


def build_split_trajectory_table(
    fcd_csv: str,
    out_csv: str,
    freq_min: int = 5,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    need_cols = ["cid", "time", "roadid"]
    rows: List[Dict[str, object]] = []
    pending: Dict[str, List[Tuple[pd.Timestamp, str]]] = {}

    reader = pd.read_csv(fcd_csv, usecols=need_cols, chunksize=chunksize)
    for chunk in reader:
        chunk["cid"] = chunk["cid"].astype(str)
        chunk["roadid"] = chunk["roadid"].astype(str)
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time", "cid", "roadid"])
        chunk = chunk.sort_values(["cid", "time"])

        for veh, g in chunk.groupby("cid", sort=False):
            buf = pending.get(veh, [])
            pts = buf + list(zip(g["time"].tolist(), g["roadid"].tolist()))
            pts.sort(key=lambda x: x[0])
            if len(pts) <= 1:
                pending[veh] = pts
                continue
            # Keep last point as carry-over for next chunk continuity.
            ready = pts[:-1]
            pending[veh] = [pts[-1]]
            segs = split_vehicle_trajectory_no_loop(ready, cid=veh, freq_min=freq_min)
            for s in segs:
                rows.append(
                    {
                        "cid": s.cid,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "travel_time_min": s.travel_time_min,
                        "edge_path": json.dumps(s.edge_path, ensure_ascii=False),
                        "node_path": json.dumps(s.node_path, ensure_ascii=False),
                        "path_key": s.path_key,
                        "od_key": s.od_key,
                        "depart_slot": s.depart_slot,
                    }
                )

    # Flush tail buffers.
    for veh, pts in pending.items():
        segs = split_vehicle_trajectory_no_loop(pts, cid=veh, freq_min=freq_min)
        for s in segs:
            rows.append(
                {
                    "cid": s.cid,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "travel_time_min": s.travel_time_min,
                    "edge_path": json.dumps(s.edge_path, ensure_ascii=False),
                    "node_path": json.dumps(s.node_path, ensure_ascii=False),
                    "path_key": s.path_key,
                    "od_key": s.od_key,
                    "depart_slot": s.depart_slot,
                }
            )

    df = pd.DataFrame(rows)
    _ensure_dir(os.path.dirname(out_csv) or ".")
    df.to_csv(out_csv, index=False)
    return df


def parse_path_field(v: object) -> List[int]:
    if isinstance(v, list):
        return [int(x) for x in v]
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        arr = ast.literal_eval(s)
        if isinstance(arr, list):
            return [int(x) for x in arr]
    except Exception:
        pass
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcd_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    df = build_split_trajectory_table(
        fcd_csv=args.fcd_csv,
        out_csv=args.out_csv,
        freq_min=args.freq_min,
        chunksize=args.chunksize,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": int(len(df)),
                "out_csv": args.out_csv,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
