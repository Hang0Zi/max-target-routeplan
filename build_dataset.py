from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils_geo import build_edge_feature_matrix
from utils_time import floor_to_slot, parse_ts, slot_index_in_day, slots_per_day, ts_to_slot_str


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _validate_required_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Got: {list(df.columns)}")


def _load_network(network_csv: str) -> pd.DataFrame:
    df = pd.read_csv(network_csv)
    # Minimal required columns (others are ignored)
    _validate_required_cols(df, ["cid", "from_node", "to_node", "geometry"], "network_csv")
    df["cid"] = df["cid"].astype(str)
    # Some CSVs store node ids as floats like "4393.0"
    df["from_node"] = df["from_node"].astype(float).astype(int)
    df["to_node"] = df["to_node"].astype(float).astype(int)
    df["geometry"] = df["geometry"].astype(str)
    return df


def _load_loop(loop_csv: str, freq_min: int, network_cids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(loop_csv)
    _validate_required_cols(df, [" ROAD_ID", "FTIME", "TTIME"], "loop.csv")

    df["ROAD_ID"] = df[" ROAD_ID"].astype(str)
    df["FTIME"] = pd.to_datetime(df["FTIME"])
    df["TTIME"] = pd.to_datetime(df["TTIME"])

    df["slot_ts"] = df["FTIME"].apply(lambda x: floor_to_slot(x, freq_min))
    df["cid"] = df["ROAD_ID"]
    df["tt_min"] = (df["TTIME"] - df["FTIME"]).dt.total_seconds() / 60.0

    # Remove invalid travel times
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["tt_min"])
    df = df[df["tt_min"] > 0]

    # Keep only edges present in the network
    df = df[df["cid"].isin(network_cids)]
    if len(df) == 0:
        raise ValueError("No loop records matched any network `cid`. Check ROAD_ID/cid alignment.")
    return df[["slot_ts", "cid", "tt_min"]]


def _build_context_features(slot_ts_list, freq_min):
    """构建上下文特征（星期几、小时、分钟等）"""
    import pandas as pd
    
    features = []
    for ts in slot_ts_list:
        # 修复：转换为 pandas Timestamp
        try:
            # 如果是 numpy.datetime64，先转换为字符串再创建 Timestamp
            if hasattr(ts, 'isoformat'):
                pd_ts = pd.Timestamp(ts)
            else:
                pd_ts = pd.Timestamp(str(ts))
        except:
            pd_ts = pd.Timestamp(ts)
        
        weekday = int(pd_ts.dayofweek)  # 0..6 (Mon..Sun)
        hour = pd_ts.hour
        minute = pd_ts.minute
        time_of_day = (hour * 60 + minute) / (24 * 60)
        
        features.append([weekday, hour, minute, time_of_day])
    
    d_ctx = len(features[0]) if features else 0
    return np.array(features, dtype=np.float32), d_ctx


def build_artifacts(
    loop_csv: str,
    network_csv: str,
    out_dir: str,
    freq_min: int = 5,
    agg: str = "mean",
) -> None:
    _ensure_dir(out_dir)

    network_df = _load_network(network_csv)
    network_cids = set(network_df["cid"].astype(str).tolist())

    edge_features, cid_to_edge_idx = build_edge_feature_matrix(network_df)
    M = int(edge_features.shape[0])
    edge_idx_to_info = []
    for _, r in network_df.iterrows():
        cid = str(r["cid"])
        eidx = cid_to_edge_idx[cid]
        edge_idx_to_info.append(
            {
                "from_node": int(r["from_node"]),
                "to_node": int(r["to_node"]),
                "cid": cid,
            }
        )
    # Ensure sorted by edge idx
    edge_idx_to_info = sorted(edge_idx_to_info, key=lambda x: cid_to_edge_idx[x["cid"]])

    loop_df = _load_loop(loop_csv, freq_min, network_cids)
    # Deterministic ordering of slots by time
    slot_ts_list = sorted(loop_df["slot_ts"].unique())
    N = len(slot_ts_list)
    slot_ts_to_idx = {pd.Timestamp(ts): i for i, ts in enumerate(slot_ts_list)}

    if N == 0:
        raise ValueError("No valid slots were found from loop.csv.")

    context_features, d_ctx = _build_context_features(slot_ts_list, freq_min)
    d_edge = int(edge_features.shape[1])

    # Build tt_matrix and mask
    tt_matrix = np.full((N, M), np.nan, dtype=np.float32)
    mask = np.zeros((N, M), dtype=np.bool_)

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    grouped = loop_df.groupby(["slot_ts", "cid"])["tt_min"]
    if agg == "mean":
        agg_series = grouped.mean()
    else:
        agg_series = grouped.median()

    for (slot_ts, cid), v in agg_series.items():
        si = slot_ts_to_idx[pd.Timestamp(slot_ts)]
        ei = cid_to_edge_idx[str(cid)]
        tt_matrix[si, ei] = float(v)
        mask[si, ei] = True

    # Save artifacts
    np.save(os.path.join(out_dir, "edge_features.npy"), edge_features)
    np.save(os.path.join(out_dir, "context_features.npy"), context_features.astype(np.float32))
    np.save(os.path.join(out_dir, "tt_matrix.npy"), tt_matrix)
    np.save(os.path.join(out_dir, "mask.npy"), mask)

    # slot_timestamps.csv for lookup by timestamp -> slot_idx (global)
    # 保存时间槽信息
    slot_rows = []
    for i, ts in enumerate(slot_ts_list):
        # 修复：将 numpy.datetime64 转换为字符串
        if hasattr(ts, 'isoformat'):
            ts_str = ts.isoformat()
        else:
            # 处理 numpy.datetime64
            ts_str = str(ts)
        slot_rows.append({"slot_idx": int(i), "slot_ts": ts_str})

    slot_df = pd.DataFrame(slot_rows)
    slot_df.to_csv(os.path.join(out_dir, "slot_timestamps.csv"), index=False)

    edge_index = {
        "cid_to_edge_idx": cid_to_edge_idx,
        "edge_idx_to_info": edge_idx_to_info,
        "M": M,
    }
    with open(os.path.join(out_dir, "edge_index.json"), "w", encoding="utf-8") as f:
        json.dump(edge_index, f, ensure_ascii=False, indent=2)

    meta = {
        "freq_min": int(freq_min),
        "slots_per_day": int(slots_per_day(freq_min)),
        "N_slots_total": int(N),
        "M": int(M),
        "d_ctx": int(d_ctx),
        "d_edge": int(d_edge),
        "slot_ts_min": str(min(slot_ts_list)),
        "slot_ts_max": str(max(slot_ts_list)),
        "loop_csv_rows": int(len(loop_df)),
        "agg": agg,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Tiny sanity check prints for CLI
    observed_ratio = float(mask.mean())
    print(f"[build_dataset] slots={N}, edges={M}, d_ctx={d_ctx}, d_edge={d_edge}, observed_ratio={observed_ratio:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop_csv", type=str, required=True)
    ap.add_argument("--network_csv", type=str, required=True)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "median"])
    args = ap.parse_args()

    build_artifacts(
        loop_csv=args.loop_csv,
        network_csv=args.network_csv,
        out_dir=args.out_dir,
        freq_min=args.freq_min,
        agg=args.agg,
    )


if __name__ == "__main__":
    main()

