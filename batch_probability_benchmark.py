from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from sample_diffusion import sample_slot
from A_star import shortest_path_astar
from Dijkstra import shortest_path_dijkstra
from evaluation import EmpiricalMatcher
from otap_plan_anytime import plan_max_otap_anytime_lagrangian
from trajectory_split import build_split_trajectory_table
from utils_time import ts_to_slot_str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_node_path(v: object) -> List[int]:
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
        return []
    return []


def _nodes_to_edges(nodes: List[int]) -> List[str]:
    if len(nodes) < 2:
        return []
    return [f"{int(nodes[i])}_{int(nodes[i + 1])}" for i in range(len(nodes) - 1)]


def _build_graph(network_csv: str) -> nx.DiGraph:
    df = pd.read_csv(network_csv, usecols=["cid", "from_node", "to_node"])
    df["cid"] = df["cid"].astype(str)
    df["from_node"] = df["from_node"].astype(float).astype(int)
    df["to_node"] = df["to_node"].astype(float).astype(int)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(int(r["from_node"]), int(r["to_node"]), cid=str(r["cid"]))
    return G


def _load_edge_index(data_dir: str) -> Dict[str, int]:
    with open(os.path.join(data_dir, "edge_index.json"), "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): int(v) for k, v in obj["cid_to_edge_idx"].items()}


def _load_tau_for_departure(samples_root: str, depart_time: pd.Timestamp, freq_min: int) -> np.ndarray:
    slot_dir = os.path.join(samples_root, ts_to_slot_str(depart_time, freq_min))
    tau_path = os.path.join(slot_dir, "tau_samples.npy")
    if not os.path.exists(tau_path):
        raise FileNotFoundError(f"Missing tau_samples for slot: {tau_path}")
    tau = np.load(tau_path)
    if tau.ndim != 2:
        raise ValueError(f"tau_samples must be [Omega, M], got {tau.shape}")
    return tau.astype(np.float32, copy=False)


def _floor_depart_slot(dep: pd.Timestamp, freq_min: int) -> pd.Timestamp:
    # Keep consistent naming with ts_to_slot_str + matcher depart_slot format.
    return pd.Timestamp(dep).floor(f"{int(freq_min)}min")


def _load_or_predict_tau_for_depart_slot(
    *,
    samples_root: str,
    depart_slot_ts: pd.Timestamp,
    freq_min: int,
    auto_sample: bool,
    checkpoint: str | None,
    num_samples: int,
    seed: int,
    device: str,
    data_dir: str,
) -> np.ndarray:
    slot_dir = os.path.join(samples_root, ts_to_slot_str(depart_slot_ts, freq_min))
    tau_path = os.path.join(slot_dir, "tau_samples.npy")
    if os.path.exists(tau_path):
        tau = np.load(tau_path)
        if tau.ndim != 2:
            raise ValueError(f"tau_samples must be [Omega, M], got {tau.shape}")
        return tau.astype(np.float32, copy=False)

    if not auto_sample:
        raise FileNotFoundError(f"Missing tau_samples for slot: {tau_path}")
    if not checkpoint:
        raise ValueError("Missing --checkpoint: required when --auto_sample is enabled.")

    # Dynamically predict and save the missing slot samples.
    sample_slot(
        checkpoint=checkpoint,
        data_dir=data_dir,
        slot=str(depart_slot_ts),
        out_dir=slot_dir,
        freq_min=freq_min,
        num_samples=num_samples,
        seed=seed,
        device=device,
    )
    if not os.path.exists(tau_path):
        raise RuntimeError(f"Auto-sampling finished but tau_samples not found: {tau_path}")
    tau = np.load(tau_path)
    if tau.ndim != 2:
        raise ValueError(f"tau_samples must be [Omega, M], got {tau.shape}")
    return tau.astype(np.float32, copy=False)


def run_batch(
    fcd_csv: str,
    split_csv: str,
    requests_csv: str,
    network_csv: str,
    data_dir: str,
    samples_root: str,
    out_csv: str,
    freq_min: int = 5,
    min_samples: int = 8,
    auto_sample: bool = False,
    checkpoint: str | None = None,
    num_samples: int = 200,
    seed: int = 0,
    device: str = "cuda",
    missing_slot_policy: str = "fail",
    beam: int = 200,
    max_hops: int = 40,
) -> pd.DataFrame:
    if not os.path.exists(split_csv):
        _ensure_dir(os.path.dirname(split_csv) or ".")
        build_split_trajectory_table(fcd_csv=fcd_csv, out_csv=split_csv, freq_min=freq_min)

    split_df = pd.read_csv(split_csv)
    matcher = EmpiricalMatcher.from_split_df(split_df, min_samples=min_samples)

    req = pd.read_csv(requests_csv)
    need_cols = {"request_id", "origin", "destination", "departure_time", "T_max"}
    if not need_cols.issubset(set(req.columns)):
        raise ValueError(f"requests_csv missing columns: {need_cols}")
    req["departure_time"] = pd.to_datetime(req["departure_time"])
    req["origin"] = req["origin"].astype(int)
    req["destination"] = req["destination"].astype(int)
    req["request_id"] = req["request_id"].astype(int)
    req["T_max"] = pd.to_numeric(req["T_max"], errors="coerce")

    # our path: computed on-the-fly by OTAP (do not require `traj` column).

    G = _build_graph(network_csv)
    cid_to_edge_idx = _load_edge_index(data_dir)

    # Preload (and optionally auto-predict) tau_samples for all departure slots.
    req["depart_slot_ts"] = req["departure_time"].apply(lambda x: _floor_depart_slot(pd.Timestamp(x), freq_min))
    slot_groups = req.groupby("depart_slot_ts", sort=False)
    tau_cache: Dict[pd.Timestamp, Optional[np.ndarray]] = {}
    for depart_slot_ts, _ in slot_groups:
        # First: if tau_samples already exist, load them directly.
        slot_dir_probe = os.path.join(samples_root, ts_to_slot_str(depart_slot_ts, freq_min))
        tau_path_probe = os.path.join(slot_dir_probe, "tau_samples.npy")
        if os.path.exists(tau_path_probe):
            tau = np.load(tau_path_probe)
            if tau.ndim != 2:
                raise ValueError(f"tau_samples must be [Omega, M], got {tau.shape}")
            tau_cache[depart_slot_ts] = tau.astype(np.float32, copy=False)
            continue

        if not auto_sample:
            if missing_slot_policy == "skip":
                tau_cache[depart_slot_ts] = None
                continue
            raise FileNotFoundError(
                f"Missing tau_samples for slot: {os.path.join(samples_root, ts_to_slot_str(depart_slot_ts, freq_min), 'tau_samples.npy')}"
            )

        tau_cache[depart_slot_ts] = _load_or_predict_tau_for_depart_slot(
            samples_root=samples_root,
            depart_slot_ts=depart_slot_ts,
            freq_min=freq_min,
            auto_sample=auto_sample,
            checkpoint=checkpoint,
            num_samples=num_samples,
            seed=seed,
            device=device,
            data_dir=data_dir,
        )

    rows: List[Dict[str, object]] = []
    for _, r in req.iterrows():
        rid = int(r["request_id"])
        o = int(r["origin"])
        d = int(r["destination"])
        dep = pd.Timestamp(r["departure_time"])
        tmax = float(r["T_max"])
        depart_slot_ts = pd.Timestamp(r["depart_slot_ts"])
        depart_slot = str(depart_slot_ts)
        od_key = f"{o}-{d}"

        tau = tau_cache.get(depart_slot_ts, None)
        omega_count = int(tau.shape[0]) if tau is not None else int(num_samples)
        our_nodes, _, _ = plan_max_otap_anytime_lagrangian(
            G=G,
            cid_to_edge_idx=cid_to_edge_idx,
            origin=o,
            destination=d,
            depart_time=dep,
            Tb_min=tmax,
            freq_min=freq_min,
            samples_root=samples_root,
            data_dir=data_dir,
            checkpoint=checkpoint,
            auto_sample=auto_sample,
            num_samples=num_samples,
            seed=seed,
            device=device,
            omega_count=omega_count,
            lagrange_iters=2,
            mu_clip=10.0,
            mu_init_noise=0.0,
            mu_label_phi=30,
            mu_label_gamma=3000,
            y_label_phi=50,
            y_label_gamma=5000,
            max_hops=max_hops,
            beam_time_slack_min=tmax,
        )
        dijkstra_nodes: List[int] = []
        astar_nodes: List[int] = []
        if tau is not None:
            try:
                dijkstra_nodes = shortest_path_dijkstra(G, o, d, cid_to_edge_idx, tau)
            except Exception:
                dijkstra_nodes = []
            try:
                astar_nodes = shortest_path_astar(G, o, d, cid_to_edge_idx, tau)
            except Exception:
                astar_nodes = []

        model_paths = {
            "our": our_nodes,
            "dijkstra": dijkstra_nodes,
            "astar": astar_nodes,
        }
        result_row: Dict[str, object] = {
            "request_id": rid,
            "origin": o,
            "destination": d,
            "departure_time": str(dep),
            "T_max": tmax,
        }
        for name, nodes in model_paths.items():
            edges = _nodes_to_edges(nodes)
            p, src, n = matcher.query_on_time_prob(
                edge_path=edges,
                od_key=od_key,
                depart_slot=depart_slot,
                t_target_min=tmax,
            )
            result_row[f"{name}_node_path"] = json.dumps(nodes, ensure_ascii=False)
            result_row[f"{name}_edge_path"] = json.dumps(edges, ensure_ascii=False)
            result_row[f"{name}_on_time_prob_empirical"] = float(p)
            result_row[f"{name}_prob_source"] = src
            result_row[f"{name}_matched_samples"] = int(n)

        probs = {
            "our": float(result_row["our_on_time_prob_empirical"]),
            "dijkstra": float(result_row["dijkstra_on_time_prob_empirical"]),
            "astar": float(result_row["astar_on_time_prob_empirical"]),
        }
        best_model = max(probs.items(), key=lambda x: x[1])[0]
        result_row["best_model"] = best_model
        rows.append(result_row)

    out = pd.DataFrame(rows)
    _ensure_dir(os.path.dirname(out_csv) or ".")
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcd_csv", type=str, required=True)
    ap.add_argument("--split_csv", type=str, required=True)
    ap.add_argument("--requests_csv", type=str, required=True)
    ap.add_argument("--network_csv", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--samples_root", type=str, required=True)
    ap.add_argument("--auto_sample", action="store_true", help="If missing tau_samples, run sample_slot and save it.")
    ap.add_argument("--checkpoint", type=str, default=None, help="Diffusion model checkpoint (required for --auto_sample).")
    ap.add_argument("--num_samples", type=int, default=200, help="Omega count when auto-sampling a missing slot.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for auto-sampling.")
    ap.add_argument("--device", type=str, default="cuda", help="Device for auto-sampling.")
    ap.add_argument(
        "--missing_slot_policy",
        type=str,
        default="fail",
        choices=["fail", "skip", "nearest"],
        help="When auto-sampling and a requested slot_ts is not present in artifacts/slot_timestamps.csv.",
    )
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--min_samples", type=int, default=8)
    ap.add_argument("--beam", type=int, default=200, help="Beam width for OTAP planning.")
    ap.add_argument("--max_hops", type=int, default=40, help="Max hops for OTAP planning.")
    args = ap.parse_args()

    out = run_batch(
        fcd_csv=args.fcd_csv,
        split_csv=args.split_csv,
        requests_csv=args.requests_csv,
        network_csv=args.network_csv,
        data_dir=args.data_dir,
        samples_root=args.samples_root,
        out_csv=args.out_csv,
        freq_min=args.freq_min,
        min_samples=args.min_samples,
        auto_sample=bool(args.auto_sample),
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
        missing_slot_policy=args.missing_slot_policy,
        beam=args.beam,
        max_hops=args.max_hops,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": int(len(out)),
                "out_csv": args.out_csv,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
