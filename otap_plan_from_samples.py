from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sample_diffusion import sample_slot
from utils_time import floor_to_slot, parse_ts, ts_to_slot_str


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_edge_index(data_dir: str) -> Dict[str, Any]:
    p = os.path.join(data_dir, "edge_index.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing edge_index.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_from_network(network_csv: str) -> nx.DiGraph:
    df = pd.read_csv(network_csv)
    need = {"cid", "from_node", "to_node"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"network_csv missing columns {need}; got {list(df.columns)}")
    df["cid"] = df["cid"].astype(str)
    df["from_node"] = df["from_node"].astype(float).astype(int)
    df["to_node"] = df["to_node"].astype(float).astype(int)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(int(r["from_node"]), int(r["to_node"]), cid=str(r["cid"]))
    return G


def nodes_to_cids(G: nx.DiGraph, path_nodes: List[int]) -> List[str]:
    cids: List[str] = []
    for i in range(len(path_nodes) - 1):
        u = int(path_nodes[i])
        v = int(path_nodes[i + 1])
        cids.append(str(G[u][v]["cid"]))
    return cids


def _slot_df(data_dir: str) -> pd.DataFrame:
    slot_csv = os.path.join(data_dir, "slot_timestamps.csv")
    df = pd.read_csv(slot_csv)
    if not {"slot_idx", "slot_ts"}.issubset(set(df.columns)):
        raise ValueError(f"slot_timestamps.csv must contain slot_idx/slot_ts, got {list(df.columns)}")
    df["slot_ts"] = pd.to_datetime(df["slot_ts"])
    return df


def _needed_slots(slot_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[pd.Timestamp]:
    s = pd.to_datetime(start_ts)
    e = pd.to_datetime(end_ts)
    out = slot_df[(slot_df["slot_ts"] >= s) & (slot_df["slot_ts"] <= e)]["slot_ts"].tolist()
    return sorted(out)


def _load_tau(slot_dir: str) -> np.ndarray:
    p = os.path.join(slot_dir, "tau_samples.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    tau = np.load(p)
    if tau.ndim != 2:
        raise ValueError(f"tau_samples.npy must be [Ω,M], got {tau.shape}")
    return tau.astype(np.float32, copy=False)


def _tt_from_cache(
    tau_cache: Dict[int, np.ndarray],
    slot_idx_vec: np.ndarray,  # [Ω]
    edge_idx: int,
    Omega: int,
) -> np.ndarray:
    """
    Each scenario row omega uses:
      tt[omega] = tau_cache[ slot_idx_vec[omega] ][ omega, edge_idx ]
    """
    tt = np.full((Omega,), np.inf, dtype=np.float32)
    uniq = np.unique(slot_idx_vec)
    for s in uniq:
        s_int = int(s)
        tau_s = tau_cache.get(s_int)
        if tau_s is None:
            continue
        idx = np.where(slot_idx_vec == s)[0]
        tt[idx] = tau_s[idx, edge_idx]
    return tt


def beam_search_otap_time_dependent(
    G: nx.DiGraph,
    origin: int,
    destination: int,
    tau_cache: Dict[int, np.ndarray],  # slot_idx_global -> [Ω,M]
    cid_to_edge_idx: Dict[str, int],
    depart_slot_idx_global: int,
    Omega: int,
    Tb_min: float,
    freq_min: int,
    beam: int,
    max_hops: int,
    min_slot_idx_global: int,
    max_slot_idx_global: int,
) -> Tuple[List[int], float, np.ndarray]:
    init_arr = np.zeros((Omega,), dtype=np.float32)
    init_slot_vec = np.full((Omega,), int(depart_slot_idx_global), dtype=np.int32)

    frontier: List[Tuple[List[int], np.ndarray, np.ndarray]] = [([int(origin)], init_arr, init_slot_vec)]

    best_path: List[int] | None = None
    best_prob = -1.0
    best_arr: np.ndarray | None = None

    for depth in range(max_hops):
        new_frontier: List[Tuple[List[int], np.ndarray, np.ndarray]] = []

        for path_nodes, arr_min, slot_idx_vec in frontier:
            u = int(path_nodes[-1])
            if u == int(destination):
                prob = float(np.mean(arr_min <= Tb_min))
                if prob > best_prob:
                    best_prob = prob
                    best_path = path_nodes
                    best_arr = arr_min
                continue

            for v in G.successors(u):
                v = int(v)
                if v in path_nodes:
                    continue
                cid = str(G[u][v]["cid"])
                eidx = cid_to_edge_idx.get(cid, None)
                if eidx is None:
                    continue

                tt = _tt_from_cache(tau_cache, slot_idx_vec, int(eidx), Omega)  # [Ω]
                arr2 = arr_min + tt

                if np.quantile(arr2, 0.1) > Tb_min and depth > 2:
                    continue

                # Paper: k' = k + tau_{ijk}^ω (converted to slots)
                dt_slots = np.ceil(tt / float(freq_min)).astype(np.int32)
                slot2 = slot_idx_vec + dt_slots

                out_mask = (slot2 < min_slot_idx_global) | (slot2 > max_slot_idx_global)
                if np.any(out_mask):
                    slot2 = slot2.copy()
                    arr2 = arr2.copy()
                    slot2[out_mask] = max_slot_idx_global + 1
                    arr2[out_mask] = np.inf

                new_frontier.append((path_nodes + [v], arr2, slot2))

        if not new_frontier:
            break

        scored: List[Tuple[float, float, List[int], np.ndarray, np.ndarray]] = []
        for pn, arr, slot2 in new_frontier:
            prob = float(np.mean(arr <= Tb_min))
            exp_t = float(np.mean(arr)) if np.isfinite(arr).any() else float("inf")
            scored.append((prob, -exp_t, pn, arr, slot2))

        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        frontier = [(pn, arr, slot2) for _, _, pn, arr, slot2 in scored[:beam]]

    if best_path is None or best_arr is None:
        return [], 0.0, np.array([], dtype=np.float32)
    return best_path, float(best_prob), best_arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network_csv", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True, help="Artifacts dir from build_dataset.py")
    ap.add_argument("--samples_root", type=str, required=True, help="slot dirs: <slot_ts_str>/tau_samples.npy")
    ap.add_argument("--origin", type=int, required=True)
    ap.add_argument("--destination", type=int, required=True)
    ap.add_argument("--depart_time", type=str, required=True)
    ap.add_argument("--Tb_min", type=float, required=True)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--max_hops", type=int, default=40)
    ap.add_argument("--max_arrival_time_min", type=float, default=None, help="Slot sampling end. Default=Tb_min")
    ap.add_argument("--checkpoint", type=str, default=None, help="If provided with --auto_sample, missing slots are sampled.")
    ap.add_argument("--auto_sample", action="store_true")
    ap.add_argument("--num_samples", type=int, default=200, help="Omega for auto-sampling.")
    ap.add_argument("--seed", type=int, default=0, help="torch manual seed for auto-sampling.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()
    G = build_graph_from_network(args.network_csv)
    edge_index = load_edge_index(args.data_dir)
    cid_to_edge_idx = {str(k): int(v) for k, v in edge_index["cid_to_edge_idx"].items()}

    depart_time = pd.to_datetime(parse_ts(args.depart_time))
    depart_slot_ts = floor_to_slot(depart_time, args.freq_min)

    if args.max_arrival_time_min is None:
        max_arrival_time_min = float(args.Tb_min)
    else:
        max_arrival_time_min = float(args.max_arrival_time_min)
    end_time = depart_time + pd.Timedelta(minutes=max_arrival_time_min)
    end_slot_ts = floor_to_slot(end_time, args.freq_min)

    slot_df = _slot_df(args.data_dir)
    needed_slot_ts = _needed_slots(slot_df, depart_slot_ts, end_slot_ts)
    if not needed_slot_ts:
        raise ValueError("No slot timestamps found in the requested time window.")

    slot_ts_to_idx = {pd.Timestamp(r["slot_ts"]): int(r["slot_idx"]) for _, r in slot_df.iterrows()}
    min_slot_idx_global = int(slot_df["slot_idx"].min())
    max_slot_idx_global = int(slot_df["slot_idx"].max())
    depart_slot_idx_global = int(slot_ts_to_idx[pd.Timestamp(depart_slot_ts)])

    tau_cache: Dict[int, np.ndarray] = {}
    Omega: int | None = None

    for s_ts in needed_slot_ts:
        slot_idx_global = int(slot_ts_to_idx[pd.Timestamp(s_ts)])
        slot_dir = os.path.join(args.samples_root, ts_to_slot_str(s_ts, args.freq_min))
        tau_path = os.path.join(slot_dir, "tau_samples.npy")
        if not os.path.exists(tau_path):
            if not args.auto_sample:
                raise FileNotFoundError(f"Missing tau_samples: {tau_path}")
            if args.checkpoint is None:
                raise ValueError("--auto_sample requires --checkpoint")
            _ensure_dir(slot_dir)
            sample_slot(
                checkpoint=args.checkpoint,
                data_dir=args.data_dir,
                slot=str(s_ts),
                out_dir=slot_dir,
                freq_min=args.freq_min,
                num_samples=args.num_samples,
                seed=args.seed,
                device=args.device,
            )

        tau_s = _load_tau(slot_dir)
        if Omega is None:
            Omega = int(tau_s.shape[0])
        elif int(tau_s.shape[0]) != Omega:
            raise ValueError(f"Omega mismatch across loaded slots: got {tau_s.shape[0]} vs {Omega}")
        tau_cache[slot_idx_global] = tau_s

    if Omega is None:
        raise RuntimeError("No tau_samples loaded.")

    best_nodes, best_prob, best_arr = beam_search_otap_time_dependent(
        G=G,
        origin=args.origin,
        destination=args.destination,
        tau_cache=tau_cache,
        cid_to_edge_idx=cid_to_edge_idx,
        depart_slot_idx_global=depart_slot_idx_global,
        Omega=int(Omega),
        Tb_min=float(args.Tb_min),
        freq_min=int(args.freq_min),
        beam=int(args.beam),
        max_hops=int(args.max_hops),
        min_slot_idx_global=min_slot_idx_global,
        max_slot_idx_global=max_slot_idx_global,
    )

    runtime_sec = time.time() - t0
    if not best_nodes:
        out = {
            "best_path_nodes": [],
            "best_path_edges": [],
            "on_time_prob": 0.0,
            "arrival_time_quantiles_min": {},
            "runtime_sec": float(runtime_sec),
            "status": "no_path_found",
        }
    else:
        best_edges = nodes_to_cids(G, best_nodes)
        qs = {
            "p50": float(np.quantile(best_arr, 0.50)),
            "p80": float(np.quantile(best_arr, 0.80)),
            "p95": float(np.quantile(best_arr, 0.95)),
        }
        out = {
            "best_path_nodes": [int(x) for x in best_nodes],
            "best_path_edges": best_edges,
            "on_time_prob": float(best_prob),
            "arrival_time_quantiles_min": qs,
            "runtime_sec": float(runtime_sec),
            "status": "ok",
        }

    _ensure_dir(os.path.dirname(args.out_json))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)

# from __future__ import annotations  # legacy duplicated code (disabled)

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sample_diffusion import sample_slot
from utils_time import floor_to_slot, parse_ts, ts_to_slot_str


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_edge_index(data_dir: str) -> Dict[str, Any]:
    p = os.path.join(data_dir, "edge_index.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing edge_index.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_from_network(network_csv: str) -> nx.DiGraph:
    df = pd.read_csv(network_csv)
    need = {"cid", "from_node", "to_node"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"network_csv missing columns {need}; got {list(df.columns)}")
    df["cid"] = df["cid"].astype(str)
    df["from_node"] = df["from_node"].astype(float).astype(int)
    df["to_node"] = df["to_node"].astype(float).astype(int)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(int(r["from_node"]), int(r["to_node"]), cid=str(r["cid"]))
    return G


def nodes_to_cids(G: nx.DiGraph, path_nodes: List[int]) -> List[str]:
    cids: List[str] = []
    for i in range(len(path_nodes) - 1):
        u = int(path_nodes[i])
        v = int(path_nodes[i + 1])
        cids.append(str(G[u][v]["cid"]))
    return cids


def _slot_df(data_dir: str) -> pd.DataFrame:
    slot_csv = os.path.join(data_dir, "slot_timestamps.csv")
    df = pd.read_csv(slot_csv)
    if not {"slot_idx", "slot_ts"}.issubset(set(df.columns)):
        raise ValueError(f"slot_timestamps.csv must contain slot_idx/slot_ts, got {list(df.columns)}")
    df["slot_ts"] = pd.to_datetime(df["slot_ts"])
    return df


def _needed_slots(slot_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[pd.Timestamp]:
    s = pd.to_datetime(start_ts)
    e = pd.to_datetime(end_ts)
    out = slot_df[(slot_df["slot_ts"] >= s) & (slot_df["slot_ts"] <= e)]["slot_ts"].tolist()
    return sorted(out)


def _load_tau(slot_dir: str) -> np.ndarray:
    p = os.path.join(slot_dir, "tau_samples.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    tau = np.load(p)
    if tau.ndim != 2:
        raise ValueError(f"tau_samples.npy must be [Ω,M], got {tau.shape}")
    return tau.astype(np.float32, copy=False)


def _tt_from_cache(
    tau_cache: Dict[int, np.ndarray],
    slot_idx_vec: np.ndarray,  # [Ω] int32 (global slot idx)
    edge_idx: int,
    Omega: int,
) -> np.ndarray:
    tt = np.full((Omega,), np.inf, dtype=np.float32)
    uniq_slots = np.unique(slot_idx_vec)
    for s in uniq_slots:
        s_int = int(s)
        tau_s = tau_cache.get(s_int, None)
        if tau_s is None:
            continue
        widx = np.where(slot_idx_vec == s)[0]
        tt[widx] = tau_s[widx, edge_idx]
    return tt


def beam_search_otap_time_dependent(
    G: nx.DiGraph,
    origin: int,
    destination: int,
    tau_cache: Dict[int, np.ndarray],
    cid_to_edge_idx: Dict[str, int],
    depart_slot_idx_global: int,
    Omega: int,
    Tb_min: float,
    freq_min: int,
    beam: int,
    max_hops: int,
    min_slot_idx_global: int,
    max_slot_idx_global: int,
) -> Tuple[List[int], float, np.ndarray]:
    init_arr = np.zeros((Omega,), dtype=np.float32)
    init_slot_vec = np.full((Omega,), int(depart_slot_idx_global), dtype=np.int32)

    frontier: List[Tuple[List[int], np.ndarray, np.ndarray]] = [([int(origin)], init_arr, init_slot_vec)]

    best_path: List[int] | None = None
    best_prob = -1.0
    best_arr: np.ndarray | None = None

    for depth in range(max_hops):
        new_frontier: List[Tuple[List[int], np.ndarray, np.ndarray]] = []
        for path_nodes, arr_min, slot_idx_vec in frontier:
            u = int(path_nodes[-1])
            if u == int(destination):
                prob = float(np.mean(arr_min <= Tb_min))
                if prob > best_prob:
                    best_prob = prob
                    best_path = path_nodes
                    best_arr = arr_min
                continue

            for v in G.successors(u):
                v = int(v)
                if v in path_nodes:
                    continue

                cid = str(G[u][v]["cid"])
                eidx = cid_to_edge_idx.get(cid, None)
                if eidx is None:
                    continue

                tt = _tt_from_cache(tau_cache, slot_idx_vec, int(eidx), Omega)
                arr2 = arr_min + tt

                # prune hopeless partial paths
                if np.quantile(arr2, 0.1) > Tb_min and depth > 2:
                    continue

                dt_slots = np.ceil(tt / float(freq_min)).astype(np.int32)
                slot2 = slot_idx_vec + dt_slots

                out_mask = (slot2 < min_slot_idx_global) | (slot2 > max_slot_idx_global)
                if np.any(out_mask):
                    slot2 = slot2.copy()
                    arr2 = arr2.copy()
                    slot2[out_mask] = max_slot_idx_global + 1
                    arr2[out_mask] = np.inf

                new_frontier.append((path_nodes + [v], arr2, slot2))

        if not new_frontier:
            break

        scored: List[Tuple[float, float, List[int], np.ndarray, np.ndarray]] = []
        for pn, arr, slot2 in new_frontier:
            prob = float(np.mean(arr <= Tb_min))
            exp_t = float(np.mean(arr)) if np.isfinite(arr).any() else float("inf")
            scored.append((prob, -exp_t, pn, arr, slot2))
        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        frontier = [(pn, arr, slot2) for _, _, pn, arr, slot2 in scored[:beam]]

    if best_path is None or best_arr is None:
        return [], 0.0, np.array([], dtype=np.float32)
    return best_path, float(best_prob), best_arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network_csv", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True, help="Artifacts dir produced by build_dataset.py")
    ap.add_argument("--samples_root", type=str, required=True, help="slot dirs: <slot_ts_str>/tau_samples.npy")
    ap.add_argument("--origin", type=int, required=True)
    ap.add_argument("--destination", type=int, required=True)
    ap.add_argument("--depart_time", type=str, required=True)
    ap.add_argument("--Tb_min", type=float, required=True)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--max_hops", type=int, default=40)
    ap.add_argument("--max_arrival_time_min", type=float, default=None)
    ap.add_argument("--checkpoint", type=str, default=None, help="If provided with --auto_sample, generate missing slots.")
    ap.add_argument("--auto_sample", action="store_true")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()

    G = build_graph_from_network(args.network_csv)
    edge_index = load_edge_index(args.data_dir)
    cid_to_edge_idx = {str(k): int(v) for k, v in edge_index["cid_to_edge_idx"].items()}

    depart_time = pd.to_datetime(parse_ts(args.depart_time))
    depart_slot_ts = floor_to_slot(depart_time, args.freq_min)
    slot_df = _slot_df(args.data_dir)

    if args.max_arrival_time_min is None:
        max_arrival_time_min = float(args.Tb_min)
    else:
        max_arrival_time_min = float(args.max_arrival_time_min)
    end_time = depart_time + pd.Timedelta(minutes=max_arrival_time_min)
    end_slot_ts = floor_to_slot(end_time, args.freq_min)

    needed_slot_ts = _needed_slots(slot_df, depart_slot_ts, end_slot_ts)
    if not needed_slot_ts:
        raise ValueError("No slot timestamps found within the requested time window.")

    slot_ts_to_idx = {pd.Timestamp(r["slot_ts"]): int(r["slot_idx"]) for _, r in slot_df.iterrows()}
    min_slot_idx_global = int(slot_df["slot_idx"].min())
    max_slot_idx_global = int(slot_df["slot_idx"].max())
    depart_slot_idx_global = int(slot_ts_to_idx[pd.Timestamp(depart_slot_ts)])

    # Preload tau_samples for all needed slots
    tau_cache: Dict[int, np.ndarray] = {}
    Omega: int | None = None

    for s_ts in needed_slot_ts:
        slot_idx_global = int(slot_ts_to_idx[pd.Timestamp(s_ts)])
        slot_dir = os.path.join(args.samples_root, ts_to_slot_str(s_ts, args.freq_min))

        tau_path = os.path.join(slot_dir, "tau_samples.npy")
        if not os.path.exists(tau_path):
            if not args.auto_sample:
                raise FileNotFoundError(f"Missing tau_samples for slot {s_ts}: {tau_path}")
            if args.checkpoint is None:
                raise ValueError("--auto_sample requires --checkpoint")
            _ensure_dir(slot_dir)
            sample_slot(
                checkpoint=args.checkpoint,
                data_dir=args.data_dir,
                slot=str(s_ts),
                out_dir=slot_dir,
                freq_min=args.freq_min,
                num_samples=args.num_samples,
                seed=args.seed,
                device=args.device,
            )

        tau_s = _load_tau(slot_dir)
        if Omega is None:
            Omega = int(tau_s.shape[0])
        else:
            if int(tau_s.shape[0]) != Omega:
                raise ValueError(f"Omega mismatch across slots: got {tau_s.shape[0]} vs {Omega}")
        tau_cache[slot_idx_global] = tau_s

    if Omega is None:
        raise RuntimeError("Failed to load any tau_samples.")

    best_nodes, best_prob, best_arr = beam_search_otap_time_dependent(
        G=G,
        origin=args.origin,
        destination=args.destination,
        tau_cache=tau_cache,
        cid_to_edge_idx=cid_to_edge_idx,
        depart_slot_idx_global=depart_slot_idx_global,
        Omega=Omega,
        Tb_min=float(args.Tb_min),
        freq_min=int(args.freq_min),
        beam=int(args.beam),
        max_hops=int(args.max_hops),
        min_slot_idx_global=min_slot_idx_global,
        max_slot_idx_global=max_slot_idx_global,
    )

    runtime_sec = time.time() - t0
    if not best_nodes:
        out = {
            "best_path_nodes": [],
            "best_path_edges": [],
            "on_time_prob": 0.0,
            "arrival_time_quantiles_min": {},
            "runtime_sec": float(runtime_sec),
            "status": "no_path_found",
        }
    else:
        best_edges = nodes_to_cids(G, best_nodes)
        qs = {
            "p50": float(np.quantile(best_arr, 0.50)),
            "p80": float(np.quantile(best_arr, 0.80)),
            "p95": float(np.quantile(best_arr, 0.95)),
        }
        out = {
            "best_path_nodes": [int(x) for x in best_nodes],
            "best_path_edges": best_edges,
            "on_time_prob": float(best_prob),
            "arrival_time_quantiles_min": qs,
            "runtime_sec": float(runtime_sec),
            "status": "ok",
        }

    _ensure_dir(os.path.dirname(args.out_json))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
    raise SystemExit(0)

# from __future__ import annotations  # legacy duplicated code (disabled)

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sample_diffusion import sample_slot
from utils_time import add_minutes_to_slot_index, floor_to_slot, parse_ts, slot_index_in_day, ts_to_slot_str


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_edge_index(data_dir: str) -> Dict[str, Any]:
    p = os.path.join(data_dir, "edge_index.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing edge_index.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_from_network(network_csv: str) -> nx.DiGraph:
    df = pd.read_csv(network_csv)
    need = {"cid", "from_node", "to_node"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"network_csv missing columns {need}; got {list(df.columns)}")
    df["cid"] = df["cid"].astype(str)
    df["from_node"] = df["from_node"].astype(int)
    df["to_node"] = df["to_node"].astype(int)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(int(r["from_node"]), int(r["to_node"]), cid=str(r["cid"]))
    return G


def nodes_to_cids(G: nx.DiGraph, path_nodes: List[int]) -> List[str]:
    cids: List[str] = []
    for i in range(len(path_nodes) - 1):
        u = int(path_nodes[i])
        v = int(path_nodes[i + 1])
        cids.append(str(G[u][v]["cid"]))
    return cids


def _slot_idx_map_from_artifacts(data_dir: str) -> pd.DataFrame:
    slot_csv = os.path.join(data_dir, "slot_timestamps.csv")
    df = pd.read_csv(slot_csv)
    if not {"slot_idx", "slot_ts"}.issubset(set(df.columns)):
        raise ValueError(f"slot_timestamps.csv must contain slot_idx/slot_ts, got {list(df.columns)}")
    df["slot_ts"] = pd.to_datetime(df["slot_ts"])
    return df


def _slot_ts_list_between(
    slot_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> List[pd.Timestamp]:
    # We assume slot_ts are already contiguous in 5-min steps in the dataset.
    start_ts2 = pd.to_datetime(start_ts)
    end_ts2 = pd.to_datetime(end_ts)
    out = slot_df[(slot_df["slot_ts"] >= start_ts2) & (slot_df["slot_ts"] <= end_ts2)]["slot_ts"].tolist()
    return sorted(out)


def _load_tau_samples_for_slot(slot_dir: str) -> np.ndarray:
    p = os.path.join(slot_dir, "tau_samples.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    tau = np.load(p)  # [Ω,M]
    if tau.ndim != 2:
        raise ValueError(f"tau_samples.npy must be [Ω,M], got shape {tau.shape}")
    return tau.astype(np.float32, copy=False)


def _get_tt_vector_from_cache(
    tau_cache: Dict[int, np.ndarray],
    slot_idx_vec: np.ndarray,
    edge_idx: int,
    Omega: int,
) -> np.ndarray:
    """
    slot_idx_vec: [Ω] int32 global slot indices for each scenario row omega.
    Return tt: [Ω] minutes for this edge for each omega, reading tau_samples[omega_idx, edge_idx]
    from each scenario slot.
    """
    tt = np.full((Omega,), np.inf, dtype=np.float32)
    # group by slot idx for fewer numpy indexing operations
    unique_slots = np.unique(slot_idx_vec)
    for s in unique_slots:
        idx = np.where(slot_idx_vec == s)[0]
        s_int = int(s)
        tau_s = tau_cache.get(s_int, None)
        if tau_s is None:
            # missing slot sample => keep inf
            continue
        # omega row selection uses the original omega indices (idx)
        tt[idx] = tau_s[idx, edge_idx]
    return tt


def beam_search_otap_time_dependent(
    G: nx.DiGraph,
    origin: int,
    destination: int,
    tau_cache: Dict[int, np.ndarray],  # slot_idx_global -> [Ω,M]
    cid_to_edge_idx: Dict[str, int],
    depart_slot_idx_global: int,
    Omega: int,
    depart_time: pd.Timestamp,  # for logging only
    Tb_min: float,
    freq_min: int,
    beam: int = 200,
    max_hops: int = 40,
    min_slot_idx_global: int = 0,
    max_slot_idx_global: int = 10**9,
) -> Tuple[List[int], float, np.ndarray]:
    init_arr = np.zeros((Omega,), dtype=np.float32)
    init_slot = np.full((Omega,), int(depart_slot_idx_global), dtype=np.int32)
    frontier: List[Tuple[List[int], np.ndarray, np.ndarray]] = [([int(origin)], init_arr, init_slot)]

    best_path: List[int] | None = None
    best_prob = -1.0
    best_arr: np.ndarray | None = None

    for depth in range(max_hops):
        new_frontier: List[Tuple[List[int], np.ndarray, np.ndarray]] = []
        for path_nodes, arr_min, slot_idx_vec in frontier:
            u = int(path_nodes[-1])
            if u == int(destination):
                prob = float(np.mean(arr_min <= Tb_min))
                if prob > best_prob:
                    best_prob = prob
                    best_path = path_nodes
                    best_arr = arr_min
                continue

            # Expand outgoing edges
            for v in G.successors(u):
                v = int(v)
                if v in path_nodes:
                    continue  # FIFO + paper assumes loop-free optimal path
                cid = str(G[u][v]["cid"])
                eidx = cid_to_edge_idx.get(cid, None)
                if eidx is None:
                    continue

                tt = _get_tt_vector_from_cache(tau_cache, slot_idx_vec, int(eidx), Omega)

                # If a scenario cannot be evaluated (inf tt), it will be pruned by deadline checks.
                arr2 = arr_min + tt

                # Hard prune: once time already clearly exceeds, skip
                if np.quantile(arr2, 0.1) > Tb_min and depth > 2:
                    continue

                # Paper slot update:
                #   k' = k + tau_{ijk}^ω  (converted to slots)
                # Here: slot length = freq_min, so slot increment = ceil(tt/freq_min)
                slot2 = slot_idx_vec + np.ceil(tt / float(freq_min)).astype(np.int32)

                # Clamp out-of-range scenarios to impossible
                out_mask = (slot2 < min_slot_idx_global) | (slot2 > max_slot_idx_global)
                if np.any(out_mask):
                    slot2 = slot2.copy()
                    arr2 = arr2.copy()
                    slot2[out_mask] = max_slot_idx_global + 1
                    arr2[out_mask] = np.inf

                new_frontier.append((path_nodes + [v], arr2, slot2))

        if not new_frontier:
            break

        # Rank by OTAP then by expected arrival time (lower is better)
        scored: List[Tuple[float, float, List[int], np.ndarray, np.ndarray]] = []
        for pn, arr, slot2 in new_frontier:
            prob = float(np.mean(arr <= Tb_min))
            exp_t = float(np.mean(arr)) if np.isfinite(arr).any() else float("inf")
            scored.append((prob, -exp_t, pn, arr, slot2))

        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        top = scored[:beam]
        frontier = [(pn, arr, slot2) for _, _, pn, arr, slot2 in top]

    if best_path is None or best_arr is None:
        return [], 0.0, np.array([], dtype=np.float32)
    return best_path, float(best_prob), best_arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network_csv", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True, help="Artifacts dir from build_dataset.")
    ap.add_argument("--samples_root", type=str, required=True, help="slot dirs: <slot_ts_str>/tau_samples.npy")
    ap.add_argument("--origin", type=int, required=True)
    ap.add_argument("--destination", type=int, required=True)
    ap.add_argument("--depart_time", type=str, required=True)
    ap.add_argument("--Tb_min", type=float, required=True)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--max_hops", type=int, default=40)
    ap.add_argument("--max_arrival_time_min", type=float, default=None, help="Slot sample range end.")
    ap.add_argument("--checkpoint", type=str, default=None, help="If provided, auto-sample missing slots.")
    ap.add_argument("--auto_sample", action="store_true")
    ap.add_argument("--num_samples", type=int, default=200, help="Omega for auto-sampling.")
    ap.add_argument("--seed", type=int, default=0, help="torch manual seed for auto-sampling.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()

    G = build_graph_from_network(args.network_csv)
    edge_index = load_edge_index(args.data_dir)
    cid_to_edge_idx = {str(k): int(v) for k, v in edge_index["cid_to_edge_idx"].items()}

    depart_time = parse_ts(args.depart_time)
    depart_time = pd.to_datetime(depart_time)

    depart_slot_ts = floor_to_slot(depart_time, args.freq_min)
    slot_df = _slot_idx_map_from_artifacts(args.data_dir)

    if args.max_arrival_time_min is None:
        max_arrival_time_min = float(args.Tb_min)
    else:
        max_arrival_time_min = float(args.max_arrival_time_min)

    end_time = depart_time + pd.Timedelta(minutes=max_arrival_time_min)
    end_slot_ts = floor_to_slot(end_time, args.freq_min)

    needed_slot_ts = _slot_ts_list_between(slot_df, depart_slot_ts, end_slot_ts)
    if not needed_slot_ts:
        raise ValueError("No slot timestamps found in the required range. Check freq_min/data_dir.")

    # Load tau_samples for all needed slots (slot_ts_str directories)
    tau_cache: Dict[int, np.ndarray] = {}
    omega_ref: int | None = None
    min_slot_idx_global = int(slot_df["slot_idx"].min())
    max_slot_idx_global = int(slot_df["slot_idx"].max())

    # Map slot_ts to global slot_idx
    slot_ts_to_idx = {pd.Timestamp(r["slot_ts"]): int(r["slot_idx"]) for _, r in slot_df.iterrows()}

    for slot_ts in needed_slot_ts:
        slot_idx_global = slot_ts_to_idx[pd.Timestamp(slot_ts)]
        slot_dir = os.path.join(args.samples_root, ts_to_slot_str(slot_ts, args.freq_min))
        tau_path = os.path.join(slot_dir, "tau_samples.npy")

        if not os.path.exists(tau_path):
            if args.auto_sample:
                if args.checkpoint is None:
                    raise ValueError("--auto_sample requires --checkpoint")
                # Auto sample using the same directory convention
                out_dir = slot_dir
                _ensure_dir(out_dir)
                sample_slot(
                    checkpoint=args.checkpoint,
                    data_dir=args.data_dir,
                    slot=str(slot_ts),
                    out_dir=out_dir,
                    freq_min=args.freq_min,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    device=args.device,
                )
            else:
                raise FileNotFoundError(f"Missing tau_samples for slot: {tau_path}")

        tau_s = _load_tau_samples_for_slot(slot_dir)
        if omega_ref is None:
            omega_ref = int(tau_s.shape[0])
        else:
            if tau_s.shape[0] != omega_ref:
                raise ValueError(f"Omega mismatch across slots: got {tau_s.shape[0]} vs {omega_ref}")
        tau_cache[int(slot_idx_global)] = tau_s

    if omega_ref is None:
        raise RuntimeError("No tau_samples loaded.")
    Omega = int(omega_ref)

    depart_slot_idx_global = int(slot_ts_to_idx[pd.Timestamp(depart_slot_ts)])

    best_nodes, best_prob, best_arr = beam_search_otap_time_dependent(
        G=G,
        origin=args.origin,
        destination=args.destination,
        tau_cache=tau_cache,
        cid_to_edge_idx=cid_to_edge_idx,
        depart_slot_idx_global=depart_slot_idx_global,
        Omega=Omega,
        depart_time=depart_time,
        Tb_min=float(args.Tb_min),
        freq_min=int(args.freq_min),
        beam=int(args.beam),
        max_hops=int(args.max_hops),
        min_slot_idx_global=min_slot_idx_global,
        max_slot_idx_global=max_slot_idx_global,
    )

    runtime_sec = time.time() - t0

    if not best_nodes:
        out = {
            "best_path_nodes": [],
            "best_path_edges": [],
            "on_time_prob": 0.0,
            "arrival_time_quantiles_min": {},
            "runtime_sec": float(runtime_sec),
            "status": "no_path_found",
        }
    else:
        best_edges = nodes_to_cids(G, best_nodes)
        qs = {
            "p50": float(np.quantile(best_arr, 0.50)),
            "p80": float(np.quantile(best_arr, 0.80)),
            "p95": float(np.quantile(best_arr, 0.95)),
        }
        out = {
            "best_path_nodes": [int(x) for x in best_nodes],
            "best_path_edges": best_edges,
            "on_time_prob": float(best_prob),
            "arrival_time_quantiles_min": qs,
            "runtime_sec": float(runtime_sec),
            "status": "ok",
        }

    _ensure_dir(os.path.dirname(args.out_json))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
print(f"Saved: {args.out_json}")


if False and __name__ == "__main__":
    main()

# from __future__ import annotations  # legacy duplicated code (disabled)

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from difftravel.utils_time import (
    floor_to_slot,
    parse_ts,
    slot_index_in_day,
    slots_per_day,
    add_minutes_to_slot_index,
)


@dataclass
class PlanResult:
    best_path_nodes: List[int]
    best_path_edges: List[str]
    on_time_prob: float
    arrival_time_quantiles_min: Dict[str, float]
    runtime_sec: float


def load_edge_index_from_dataset(data_dir: str) -> Dict:
    with open(os.path.join(data_dir, "edge_index.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_from_network(network_csv: str) -> nx.DiGraph:
    df = pd.read_csv(network_csv)
    df["cid"] = df["cid"].astype(str)
    df["from_node"] = df["from_node"].astype(int)
    df["to_node"] = df["to_node"].astype(int)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        u = int(r["from_node"])
        v = int(r["to_node"])
        cid = str(r["cid"])
        G.add_edge(u, v, cid=cid)
    return G


def nodes_to_cids(G: nx.DiGraph, path_nodes: List[int]) -> List[str]:
    cids = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        cids.append(G[u][v]["cid"])
    return cids


def compute_path_arrival_times_samples(
    path_cids: List[str],
    cid_to_eidx: Dict[str, int],
    tau_samples: np.ndarray,  # [Ω,M]
    depart_slot_in_day: int,
    freq_min: int,
) -> np.ndarray:
    """
    Implement the paper-style time-dependent slot update:
      k' = k + tau_ijk^ω (converted to slots)
    Here tau_samples are assumed as travel time at the departure slot, but we only have
    one vector per edge (no per-slot profile). To stay consistent and runnable:
      - we use the sampled edge time as constant for the whole traversal within the query slot.
      - slot update is approximated by advancing slot index by ceil(tt/freq).
    """
    Omega, M = tau_samples.shape
    total_min = np.zeros((Omega,), dtype=np.float32)
    slot_k = np.full((Omega,), depart_slot_in_day, dtype=np.int32)

    for cid in path_cids:
        eidx = cid_to_eidx.get(cid, None)
        if eidx is None:
            # missing edge -> make it impossible
            return np.full((Omega,), np.inf, dtype=np.float32)

        tt = tau_samples[:, eidx]  # [Ω] minutes
        total_min += tt
        # advance slot index (approx)
        for w in range(Omega):
            slot_k[w] = add_minutes_to_slot_index(int(slot_k[w]), float(tt[w]), freq_min)

    return total_min


def beam_search_otap(
    G: nx.DiGraph,
    origin: int,
    destination: int,
    tau_samples: np.ndarray,  # [Ω,M]
    cid_to_eidx: Dict[str, int],
    depart_time: pd.Timestamp,
    Tb_min: float,
    freq_min: int,
    beam: int = 200,
    max_hops: int = 40,
) -> Tuple[List[int], float, np.ndarray]:
    """
    Maximize OTAP using beam search.
    Score for partial path: upper bound of OTAP is 1; we use heuristic = 0 (no optimistic gain).
    We keep top 'beam' candidates by (estimated on-time prob, then shorter expected time).
    """
    depart_time = floor_to_slot(depart_time, freq_min)
    depart_slot = slot_index_in_day(depart_time, freq_min)

    Omega = tau_samples.shape[0]

    # state: (path_nodes, arrival_samples_min)
    init_arr = np.zeros((Omega,), dtype=np.float32)
    frontier = [([origin], init_arr)]

    best_path = None
    best_prob = -1.0
    best_arr = None

    for depth in range(max_hops):
        new_frontier = []
        for path_nodes, arr_samples in frontier:
            u = path_nodes[-1]
            if u == destination:
                prob = float(np.mean(arr_samples <= Tb_min))
                if prob > best_prob:
                    best_prob = prob
                    best_path = path_nodes
                    best_arr = arr_samples
                continue

            for v in G.successors(u):
                if v in path_nodes:
                    continue  # simple path constraint
                cid = G[u][v]["cid"]
                eidx = cid_to_eidx.get(cid, None)
                if eidx is None:
                    continue
                tt = tau_samples[:, eidx]
                arr2 = arr_samples + tt
                # quick prune: if even 10th percentile already > Tb, it's likely hopeless
                if np.quantile(arr2, 0.1) > Tb_min and depth > 2:
                    continue
                new_frontier.append((path_nodes + [v], arr2))

        if not new_frontier:
            break

        # Rank candidates
        # primary: on-time probability; secondary: expected travel time
        scored = []
        for pn, arr in new_frontier:
            prob = float(np.mean(arr <= Tb_min))
            exp_t = float(np.mean(arr))
            scored.append((prob, -exp_t, pn, arr))

        scored.sort(reverse=True)
        frontier = [(pn, arr) for _, _, pn, arr in scored[:beam]]

    if best_path is None:
        return [], 0.0, np.array([], dtype=np.float32)
    return best_path, float(best_prob), best_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network_csv", type=str, required=True)
    ap.add_argument("--samples_dir", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default=None, help="If provided, use artifacts/difftravel_run1 for edge_index.json")
    ap.add_argument("--origin", type=int, required=True)
    ap.add_argument("--destination", type=int, required=True)
    ap.add_argument("--depart_time", type=str, required=True)
    ap.add_argument("--Tb_min", type=float, required=True)
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--max_hops", type=int, default=40)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()

    G = build_graph_from_network(args.network_csv)

    tau_samples = np.load(os.path.join(args.samples_dir, "tau_samples.npy"))  # [Ω,M]
    if args.data_dir is None:
        raise ValueError("Please provide --data_dir pointing to artifacts/difftravel_run1 (for edge_index.json).")
    edge_index = load_edge_index_from_dataset(args.data_dir)
    cid_to_eidx = edge_index["cid_to_edge_idx"]

    depart_time = parse_ts(args.depart_time)

    best_nodes, best_prob, best_arr = beam_search_otap(
        G=G,
        origin=args.origin,
        destination=args.destination,
        tau_samples=tau_samples,
        cid_to_eidx=cid_to_eidx,
        depart_time=depart_time,
        Tb_min=args.Tb_min,
        freq_min=args.freq_min,
        beam=args.beam,
        max_hops=args.max_hops,
    )

    runtime = time.time() - t0

    if not best_nodes:
        out = {
            "best_path_nodes": [],
            "best_path_edges": [],
            "on_time_prob": 0.0,
            "arrival_time_quantiles_min": {},
            "runtime_sec": float(runtime),
            "status": "no_path_found",
        }
    else:
        best_edges = nodes_to_cids(G, best_nodes)
        qs = {
            "p50": float(np.quantile(best_arr, 0.50)),
            "p80": float(np.quantile(best_arr, 0.80)),
            "p95": float(np.quantile(best_arr, 0.95)),
        }
        out = {
            "best_path_nodes": [int(x) for x in best_nodes],
            "best_path_edges": best_edges,
            "on_time_prob": float(best_prob),
            "arrival_time_quantiles_min": qs,
            "runtime_sec": float(runtime),
            "status": "ok",
        }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()