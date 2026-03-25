from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sample_diffusion import sample_slot
from utils_time import floor_to_slot, parse_ts, ts_to_slot_str


def load_edge_index(data_dir: str) -> Dict:
    p = os.path.join(data_dir, "edge_index.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing edge_index.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_from_network(network_csv: str) -> nx.DiGraph:
    df = pd.read_csv(network_csv, usecols=["cid", "from_node", "to_node"])
    if not {"cid", "from_node", "to_node"}.issubset(set(df.columns)):
        raise ValueError(f"network_csv must contain cid/from_node/to_node, got {list(df.columns)}")
    df["cid"] = df["cid"].astype(str)
    df["from_node"] = df["from_node"].astype(float).astype(int)
    df["to_node"] = df["to_node"].astype(float).astype(int)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(int(r["from_node"]), int(r["to_node"]), cid=str(r["cid"]))
    return G


def _nodes_to_edges(nodes: List[int]) -> List[str]:
    if len(nodes) < 2:
        return []
    return [f"{int(nodes[i])}_{int(nodes[i + 1])}" for i in range(len(nodes) - 1)]


def _tt_from_cache_rel(
    tau_cache_rel: Dict[int, np.ndarray],
    slot_idx_vec: np.ndarray,  # [Omega] relative slot idx
    edge_idx: int,
    Omega: int,
) -> np.ndarray:
    tt = np.full((Omega,), np.inf, dtype=np.float32)
    uniq = np.unique(slot_idx_vec)
    for s in uniq:
        s_int = int(s)
        tau_s = tau_cache_rel.get(s_int, None)
        if tau_s is None:
            continue
        idx = np.where(slot_idx_vec == s)[0]
        tt[idx] = tau_s[idx, edge_idx]
    return tt


def beam_search_otap_time_dependent_relative(
    *,
    G: nx.DiGraph,
    origin: int,
    destination: int,
    tau_cache_rel: Dict[int, np.ndarray],  # rel_idx -> [Omega,M]
    cid_to_edge_idx: Dict[str, int],
    Omega: int,
    Tb_min: float,
    freq_min: int,
    beam: int,
    max_hops: int,
    max_rel_slot_idx: int,
) -> Tuple[List[int], float, np.ndarray]:
    init_arr = np.zeros((Omega,), dtype=np.float32)
    init_slot_vec = np.zeros((Omega,), dtype=np.int32)  # relative slot 0 at depart_slot
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

                tt = _tt_from_cache_rel(tau_cache_rel, slot_idx_vec, int(eidx), Omega)  # [Omega]
                arr2 = arr_min + tt

                # prune hopeless partial paths
                if np.quantile(arr2, 0.1) > Tb_min and depth > 2:
                    continue

                dt_slots = np.ceil(tt / float(freq_min)).astype(np.int32)
                slot2 = slot_idx_vec + dt_slots

                out_mask = (slot2 < 0) | (slot2 > max_rel_slot_idx)
                if np.any(out_mask):
                    slot2 = slot2.copy()
                    arr2 = arr2.copy()
                    slot2[out_mask] = max_rel_slot_idx + 1
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


def _load_or_sample_tau_for_slot_ts(
    *,
    slot_ts: pd.Timestamp,
    samples_root: str,
    data_dir: str,
    freq_min: int,
    checkpoint: str | None,
    auto_sample: bool,
    num_samples: int,
    seed: int,
    device: str,
    tau_cache_by_slot_ts: Dict[pd.Timestamp, np.ndarray] | None = None,
) -> np.ndarray:
    slot_ts = pd.Timestamp(slot_ts)
    if tau_cache_by_slot_ts is not None and slot_ts in tau_cache_by_slot_ts:
        return tau_cache_by_slot_ts[slot_ts]

    slot_dir = os.path.join(samples_root, ts_to_slot_str(slot_ts, freq_min))
    tau_path = os.path.join(slot_dir, "tau_samples.npy")
    if not os.path.exists(tau_path):
        if not auto_sample:
            raise FileNotFoundError(f"Missing tau_samples: {tau_path}")
        if checkpoint is None:
            raise ValueError("--checkpoint required when --auto_sample is enabled.")
        os.makedirs(slot_dir, exist_ok=True)
        sample_slot(
            checkpoint=checkpoint,
            data_dir=data_dir,
            slot=str(slot_ts),
            out_dir=slot_dir,
            freq_min=freq_min,
            num_samples=num_samples,
            seed=seed,
            device=device,
        )
        if not os.path.exists(tau_path):
            raise RuntimeError(f"Auto-sampling finished but tau_samples missing: {tau_path}")

    tau = np.load(tau_path)
    if tau.ndim != 2:
        raise ValueError(f"tau_samples.npy must be [Omega,M], got {tau.shape} at {tau_path}")
    tau = tau.astype(np.float32, copy=False)

    if tau_cache_by_slot_ts is not None:
        tau_cache_by_slot_ts[slot_ts] = tau
    return tau


def plan_max_otap_anytime(
    *,
    G: nx.DiGraph,
    cid_to_edge_idx: Dict[str, int],
    origin: int,
    destination: int,
    depart_time: pd.Timestamp | str,
    Tb_min: float,
    freq_min: int,
    beam: int,
    max_hops: int,
    samples_root: str,
    data_dir: str,
    checkpoint: str | None = None,
    auto_sample: bool = False,
    num_samples: int = 200,
    seed: int = 0,
    device: str = "cuda",
    tau_cache_by_slot_ts: Dict[pd.Timestamp, np.ndarray] | None = None,
    max_arrival_time_min: float | None = None,
) -> Tuple[List[int], float, np.ndarray]:
    depart_time = pd.to_datetime(parse_ts(depart_time))
    depart_slot_ts = floor_to_slot(depart_time, freq_min)
    if max_arrival_time_min is None:
        max_arrival_time_min = float(Tb_min)
    end_time = depart_time + pd.Timedelta(minutes=float(max_arrival_time_min))
    end_slot_ts = floor_to_slot(end_time, freq_min)

    if end_slot_ts < depart_slot_ts:
        return [], 0.0, np.array([], dtype=np.float32)

    step = pd.Timedelta(minutes=freq_min)
    n_steps = int((end_slot_ts - depart_slot_ts) / step) + 1

    tau_cache_rel: Dict[int, np.ndarray] = {}
    Omega: int | None = None

    for rel_idx in range(n_steps):
        slot_ts = depart_slot_ts + rel_idx * step
        tau_s = _load_or_sample_tau_for_slot_ts(
            slot_ts=slot_ts,
            samples_root=samples_root,
            data_dir=data_dir,
            freq_min=freq_min,
            checkpoint=checkpoint,
            auto_sample=auto_sample,
            num_samples=num_samples,
            seed=seed,
            device=device,
            tau_cache_by_slot_ts=tau_cache_by_slot_ts,
        )
        if Omega is None:
            Omega = int(tau_s.shape[0])
        elif int(tau_s.shape[0]) != Omega:
            raise ValueError(f"Omega mismatch across slot_ts: got {tau_s.shape[0]} vs {Omega}")
        tau_cache_rel[rel_idx] = tau_s

    if Omega is None:
        raise RuntimeError("Failed to load any tau samples.")

    max_rel_slot_idx = n_steps - 1
    best_nodes, best_prob, best_arr = beam_search_otap_time_dependent_relative(
        G=G,
        origin=int(origin),
        destination=int(destination),
        tau_cache_rel=tau_cache_rel,
        cid_to_edge_idx=cid_to_edge_idx,
        Omega=int(Omega),
        Tb_min=float(Tb_min),
        freq_min=int(freq_min),
        beam=int(beam),
        max_hops=int(max_hops),
        max_rel_slot_idx=int(max_rel_slot_idx),
    )
    return best_nodes, best_prob, best_arr


@dataclass
class _LabelRCSP:
    cost: float
    time_min: float
    zeta: int
    node: int
    depth: int
    visited: Set[int]
    parent: Optional["_LabelRCSP"]


def _extract_edges_from_label_end(end_label: _LabelRCSP) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    cur = end_label
    nodes_rev: List[int] = []
    while cur is not None:
        nodes_rev.append(cur.node)
        cur = cur.parent
    nodes = list(reversed(nodes_rev))
    for i in range(len(nodes) - 1):
        edges.append((nodes[i], nodes[i + 1]))
    return edges


def _compute_arrival_times_for_path_anyomega(
    *,
    G: nx.DiGraph,
    cid_to_edge_idx: Dict[str, int],
    depart_time: pd.Timestamp,
    path_edges: List[Tuple[int, int]],
    Tb_min: float,
    freq_min: int,
    Omega: int,
    samples_root: str,
    data_dir: str,
    checkpoint: str | None,
    auto_sample: bool,
    num_samples: int,
    seed: int,
    device: str,
    tau_cache_by_slot_ts: Dict[pd.Timestamp, np.ndarray],
) -> np.ndarray:
    # Simulate arrival time samples for each omega along a fixed edge path.
    elapsed_min = np.zeros((Omega,), dtype=np.float64)
    for (u, v) in path_edges:
        cid = str(G[u][v]["cid"])
        eidx = int(cid_to_edge_idx[cid])
        tt = np.zeros((Omega,), dtype=np.float64)
        for w in range(Omega):
            t_abs = depart_time + pd.Timedelta(minutes=float(elapsed_min[w]))
            slot_ts = floor_to_slot(t_abs, freq_min)
            if slot_ts not in tau_cache_by_slot_ts:
                tau_cache_by_slot_ts[slot_ts] = _load_or_sample_tau_for_slot_ts(
                    slot_ts=slot_ts,
                    samples_root=samples_root,
                    data_dir=data_dir,
                    freq_min=freq_min,
                    checkpoint=checkpoint,
                    auto_sample=auto_sample,
                    num_samples=num_samples,
                    seed=seed,
                    device=device,
                    tau_cache_by_slot_ts=None,
                )
            tt[w] = float(tau_cache_by_slot_ts[slot_ts][w, eidx])
        elapsed_min += tt
    return elapsed_min.astype(np.float32, copy=False)


def _sp_omega_rcsp_label(
    *,
    G: nx.DiGraph,
    cid_to_edge_idx: Dict[str, int],
    origin: int,
    destination: int,
    depart_time: pd.Timestamp,
    Tb_min: float,
    freq_min: int,
    Omega: int,
    omega_idx: int,
    mu_by_edge: Dict[Tuple[int, int], np.ndarray],
    tau_cache_by_slot_ts: Dict[pd.Timestamp, np.ndarray],
    samples_root: str,
    data_dir: str,
    checkpoint: str | None,
    auto_sample: bool,
    num_samples: int,
    seed: int,
    device: str,
    max_hops: int,
    phi: int,
    gamma: int,
    beam_time_slack_min: float,
) -> Tuple[List[Tuple[int, int]], int, float]:
    # Minimization of: p*zeta - sum_mu x along scenario path.
    # We use zeta = 1{arrival_time > Tb_min}; since edge tt are clamped positive, zeta is monotone.
    p = 1.0 / float(Omega)

    start_zeta = 0
    start = _LabelRCSP(
        cost=0.0,
        time_min=0.0,
        zeta=start_zeta,
        node=int(origin),
        depth=0,
        visited={int(origin)},
        parent=None,
    )

    heap: List[Tuple[float, int, _LabelRCSP]] = []
    counter = 0
    heapq.heappush(heap, (start.cost, counter, start))
    labels_per_node: Dict[int, List[_LabelRCSP]] = {int(origin): [start]}
    expanded = 0

    while heap and expanded < gamma:
        _, _, lab = heapq.heappop(heap)
        expanded += 1

        if lab.node == int(destination):
            edges = _extract_edges_from_label_end(lab)
            return edges, lab.zeta, float(lab.cost)

        if lab.depth >= max_hops:
            continue

        # Hard prune by time horizon for practicality.
        if lab.time_min > (Tb_min + beam_time_slack_min):
            continue

        for v in G.successors(lab.node):
            v = int(v)
            if v in lab.visited:
                continue
            cid = str(G[lab.node][v]["cid"])
            eidx = int(cid_to_edge_idx[cid])

            t_abs = depart_time + pd.Timedelta(minutes=float(lab.time_min))
            slot_ts = floor_to_slot(t_abs, freq_min)
            if slot_ts not in tau_cache_by_slot_ts:
                tau_cache_by_slot_ts[slot_ts] = _load_or_sample_tau_for_slot_ts(
                    slot_ts=slot_ts,
                    samples_root=samples_root,
                    data_dir=data_dir,
                    freq_min=freq_min,
                    checkpoint=checkpoint,
                    auto_sample=auto_sample,
                    num_samples=num_samples,
                    seed=seed,
                    device=device,
                    tau_cache_by_slot_ts=None,
                )
            tau_s = tau_cache_by_slot_ts[slot_ts]
            tt = float(tau_s[omega_idx, eidx])

            t_new = lab.time_min + tt
            z_new = 1 if t_new > Tb_min else 0

            edge_key = (lab.node, v)
            mu_arr = mu_by_edge.get(edge_key)
            mu = 0.0 if mu_arr is None else float(mu_arr[omega_idx])
            arc_cost = -mu

            # Only add p when zeta flips to 1.
            add_p = p * float(z_new - lab.zeta)
            c_new = lab.cost + arc_cost + add_p

            new_vis = set(lab.visited)
            new_vis.add(v)
            new_lab = _LabelRCSP(
                cost=c_new,
                time_min=t_new,
                zeta=z_new,
                node=v,
                depth=lab.depth + 1,
                visited=new_vis,
                parent=lab,
            )

            # Dominance: prune if an existing label at same node dominates it.
            cur_list = labels_per_node.get(v, [])
            dominated = False
            for old in cur_list:
                if old.zeta <= new_lab.zeta and old.time_min <= new_lab.time_min and old.cost <= new_lab.cost:
                    dominated = True
                    break
            if dominated:
                continue

            cur_list.append(new_lab)
            cur_list.sort(key=lambda x: x.cost)
            if len(cur_list) > phi:
                cur_list = cur_list[:phi]
            labels_per_node[v] = cur_list

            counter += 1
            heapq.heappush(heap, (new_lab.cost, counter, new_lab))

    # If no path found: return empty with zeta=1 as worst-case lateness.
    return [], 1, float("inf")


def _sp_y_bounded_min_cost(
    *,
    G: nx.DiGraph,
    origin: int,
    destination: int,
    mu_by_edge: Dict[Tuple[int, int], np.ndarray],
    Omega: int,
    max_hops: int,
    phi: int,
    gamma: int,
) -> Tuple[List[Tuple[int, int]], float]:
    # Minimize Σ_{(u,v) in y} Σ_ω μ_{uv}^ω
    # Edge weight can be negative; we bound path length and disallow revisiting nodes.

    @dataclass
    class _LabelY:
        cost: float
        node: int
        depth: int
        visited: Set[int]
        parent: Optional["_LabelY"]

    def _edges_from_label(end: _LabelY) -> List[Tuple[int, int]]:
        nodes_rev: List[int] = []
        cur = end
        while cur is not None:
            nodes_rev.append(cur.node)
            cur = cur.parent
        nodes = list(reversed(nodes_rev))
        return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    start = _LabelY(cost=0.0, node=int(origin), depth=0, visited={int(origin)}, parent=None)
    heap: List[Tuple[float, int, _LabelY]] = []
    counter = 0
    heapq.heappush(heap, (start.cost, counter, start))

    labels_per_node: Dict[int, List[_LabelY]] = {int(origin): [start]}
    expanded = 0
    best_goal: Optional[_LabelY] = None

    while heap and expanded < gamma:
        _, _, lab = heapq.heappop(heap)
        expanded += 1

        if lab.node == int(destination):
            if best_goal is None or lab.cost < best_goal.cost:
                best_goal = lab
            continue

        if lab.depth >= max_hops:
            continue

        for v in G.successors(lab.node):
            v = int(v)
            if v in lab.visited:
                continue

            edge_key = (lab.node, v)
            mu_arr = mu_by_edge.get(edge_key)
            edge_cost = 0.0 if mu_arr is None else float(np.sum(mu_arr[:Omega]))
            c_new = lab.cost + edge_cost

            new_lab = _LabelY(
                cost=c_new,
                node=v,
                depth=lab.depth + 1,
                visited=set(lab.visited) | {v},
                parent=lab,
            )

            # Simple dominance: if any old label for v has <= cost, prune.
            lst = labels_per_node.get(v, [])
            dominated = False
            for old in lst:
                if old.cost <= new_lab.cost:
                    dominated = True
                    break
            if dominated:
                continue

            lst.append(new_lab)
            lst.sort(key=lambda x: x.cost)
            if len(lst) > phi:
                lst = lst[:phi]
            labels_per_node[v] = lst

            counter += 1
            heapq.heappush(heap, (new_lab.cost, counter, new_lab))

    if best_goal is None:
        return [], float("inf")
    return _edges_from_label(best_goal), float(best_goal.cost)


def plan_max_otap_anytime_lagrangian(
    *,
    G: nx.DiGraph,
    cid_to_edge_idx: Dict[str, int],
    origin: int,
    destination: int,
    depart_time: pd.Timestamp | str,
    Tb_min: float,
    freq_min: int,
    samples_root: str,
    data_dir: str,
    checkpoint: str | None = None,
    auto_sample: bool = False,
    num_samples: int = 200,
    seed: int = 0,
    device: str = "cuda",
    omega_count: int = 200,
    lagrange_iters: int = 2,
    mu_clip: float = 10.0,
    mu_init_noise: float = 0.0,
    mu_label_phi: int = 30,
    mu_label_gamma: int = 3000,
    y_label_phi: int = 50,
    y_label_gamma: int = 5000,
    max_hops: int = 40,
    beam_time_slack_min: float = 0.0,
) -> Tuple[List[int], float, np.ndarray]:
    # Returns: best_path_nodes, best_on_time_prob_empirical_from_samples, arrival_times_min_best
    depart_time = pd.to_datetime(parse_ts(depart_time))
    origin = int(origin)
    destination = int(destination)
    Tb_min = float(Tb_min)

    Omega = int(omega_count)

    # μ storage: only for edges seen in y/x candidates.
    mu_by_edge: Dict[Tuple[int, int], np.ndarray] = {}

    # τ cache used across SP_ω and y simulation.
    tau_cache_by_slot_ts: Dict[pd.Timestamp, np.ndarray] = {}

    # helper: load tau matrix to know Omega if needed
    mu_edges_union: Set[Tuple[int, int]] = set()

    best_on_time_prob = -1.0
    best_arrival_times = np.array([], dtype=np.float32)
    best_y_edges: List[Tuple[int, int]] = []

    # Initialize μ to random small values if requested.
    if mu_init_noise > 0:
        rng = np.random.default_rng(int(seed))
        # We can only initialize for edges that appear; use empty start and grow as edges appear.
        _ = rng

    for it in range(lagrange_iters):
        # SP_ω for each omega scenario.
        x_edges_by_omega: List[List[Tuple[int, int]]] = []
        zeta_by_omega: List[int] = []

        # Ensure tau for depart slot exists (helps later).
        # Load lazily.
        for w in range(Omega):
            edges_w, z_w, _cost_w = _sp_omega_rcsp_label(
                G=G,
                cid_to_edge_idx=cid_to_edge_idx,
                origin=origin,
                destination=destination,
                depart_time=depart_time,
                Tb_min=Tb_min,
                freq_min=freq_min,
                Omega=Omega,
                omega_idx=w,
                mu_by_edge=mu_by_edge,
                tau_cache_by_slot_ts=tau_cache_by_slot_ts,
                samples_root=samples_root,
                data_dir=data_dir,
                checkpoint=checkpoint,
                auto_sample=auto_sample,
                num_samples=num_samples,
                seed=seed,
                device=device,
                max_hops=max_hops,
                phi=mu_label_phi,
                gamma=mu_label_gamma,
                beam_time_slack_min=beam_time_slack_min,
            )
            x_edges_by_omega.append(edges_w)
            zeta_by_omega.append(int(z_w))
            for e in edges_w:
                mu_edges_union.add(e)

        # SP_y: minimize Σ_{edges in y} Σ_ω μ_e^ω (bounded simple path).
        # Ensure μ vectors exist for edges seen in x^ω candidates.
        for e in mu_edges_union:
            if e not in mu_by_edge:
                mu_by_edge[e] = np.zeros((Omega,), dtype=np.float32)

        y_edges, _y_cost = _sp_y_bounded_min_cost(
            G=G,
            origin=origin,
            destination=destination,
            mu_by_edge=mu_by_edge,
            Omega=Omega,
            max_hops=max_hops,
            phi=y_label_phi,
            gamma=y_label_gamma,
        )
        if not y_edges:
            if best_y_edges:
                y_edges = best_y_edges
            else:
                return [], 0.0, np.array([], dtype=np.float32)

        # Evaluate y via samples: compute arrival time per omega along y.
        arrival_min = _compute_arrival_times_for_path_anyomega(
            G=G,
            cid_to_edge_idx=cid_to_edge_idx,
            depart_time=depart_time,
            path_edges=y_edges,
            Tb_min=Tb_min,
            freq_min=freq_min,
            Omega=Omega,
            samples_root=samples_root,
            data_dir=data_dir,
            checkpoint=checkpoint,
            auto_sample=auto_sample,
            num_samples=num_samples,
            seed=seed,
            device=device,
            tau_cache_by_slot_ts=tau_cache_by_slot_ts,
        )
        on_time_prob = float(np.mean(arrival_min <= Tb_min)) if arrival_min.size else 0.0

        if on_time_prob > best_on_time_prob:
            best_on_time_prob = on_time_prob
            best_arrival_times = arrival_min
            best_y_edges = y_edges

        # Ensure mu arrays exist for edges encountered.
        y_edge_set = set(y_edges)
        for e in mu_edges_union.union(y_edge_set):
            if e not in mu_by_edge:
                mu_by_edge[e] = np.zeros((Omega,), dtype=np.float32)

        # Compute UB and LB for step size.
        UB_late = float(np.mean(arrival_min > Tb_min)) if arrival_min.size else 1.0

        Theta = 0.0
        for e in y_edges:
            Theta += float(np.sum(mu_by_edge.get(e, np.zeros((Omega,), dtype=np.float32))[:Omega]))

        w_dual = 0.0
        p = 1.0 / float(Omega)
        for w in range(Omega):
            # theta^w = p*zeta^w - Σ μ_e^w x_e^w
            x_edges_w = x_edges_by_omega[w]
            sum_mu_x = 0.0
            for e in x_edges_w:
                if e in mu_by_edge:
                    sum_mu_x += float(mu_by_edge[e][w])
            theta = p * float(zeta_by_omega[w]) - sum_mu_x
            w_dual += theta
        LB_dual = w_dual + Theta

        # Subgradient g_{e}^w = y_e - x_e^w.
        # g^2 is 1 when mismatch, else 0.
        mismatch_count = 0
        # Track edges union to update only those.
        edges_union = list(y_edge_set.union(mu_edges_union))

        x_edge_sets_by_omega: List[Set[Tuple[int, int]]] = []
        for w in range(Omega):
            x_edge_sets_by_omega.append(set(x_edges_by_omega[w]))

        for e in edges_union:
            y_e = 1 if e in y_edge_set else 0
            for w in range(Omega):
                x_e_w = 1 if e in x_edge_sets_by_omega[w] else 0
                if y_e != x_e_w:
                    mismatch_count += 1

        if mismatch_count == 0:
            break

        g_norm_sq = float(mismatch_count)
        alpha = float(0.0)
        gap = UB_late - LB_dual
        if gap > 0:
            alpha = float((0.5 * gap) / g_norm_sq)  # beta=0.5 fixed for stability
        alpha = max(0.0, min(alpha, 1.0))

        if alpha == 0.0:
            continue

        # μ update: μ_e^w <- clip(μ_e^w + alpha*(y_e - x_e^w))
        for e in edges_union:
            if e not in mu_by_edge:
                mu_by_edge[e] = np.zeros((Omega,), dtype=np.float32)
            mu_vec = mu_by_edge[e]
            for w in range(Omega):
                y_e = 1.0 if e in y_edge_set else 0.0
                x_e_w = 1.0 if e in x_edge_sets_by_omega[w] else 0.0
                g = y_e - x_e_w
                mu_vec[w] = float(np.clip(mu_vec[w] + alpha * g, -mu_clip, mu_clip))

    # Convert best_y_edges to nodes path (best_y_edges is an edge list in order, may be empty).
    if not best_y_edges:
        return [], best_on_time_prob if best_on_time_prob >= 0 else 0.0, best_arrival_times
    # Reconstruct nodes path from edge list:
    nodes: List[int] = [int(best_y_edges[0][0])]
    for (u, v) in best_y_edges:
        nodes.append(int(v))
    return nodes, best_on_time_prob, best_arrival_times

