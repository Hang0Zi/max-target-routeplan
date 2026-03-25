from __future__ import annotations

import networkx as nx
import numpy as np


def _build_mean_min_graphs(
    G: nx.DiGraph,
    cid_to_edge_idx: dict[str, int],
    tau_samples: np.ndarray,  # [Omega, M]
) -> tuple[nx.DiGraph, nx.DiGraph]:
    mean_w = np.mean(tau_samples, axis=0)
    min_w = np.min(tau_samples, axis=0)

    G_mean = nx.DiGraph()
    G_min = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cid = str(data["cid"])
        eidx = cid_to_edge_idx.get(cid)
        if eidx is None:
            continue
        ei = int(eidx)
        G_mean.add_edge(int(u), int(v), cid=cid, weight=float(mean_w[ei]))
        G_min.add_edge(int(u), int(v), cid=cid, weight=float(min_w[ei]))
    return G_mean, G_min


def shortest_path_astar(
    G: nx.DiGraph,
    origin: int,
    destination: int,
    cid_to_edge_idx: dict[str, int],
    tau_samples: np.ndarray,
) -> list[int]:
    """
    A* baseline:
    - path cost g: mean sampled travel time
    - heuristic h: lower-bound shortest time to destination under min sampled edge time
    """
    G_mean, G_min = _build_mean_min_graphs(G, cid_to_edge_idx, tau_samples)

    rev = G_min.reverse(copy=True)
    try:
        dist_to_goal = nx.single_source_dijkstra_path_length(rev, int(destination), weight="weight")
    except Exception:
        dist_to_goal = {}

    def heuristic(u: int, v: int) -> float:
        _ = v
        return float(dist_to_goal.get(int(u), 0.0))

    return [
        int(x)
        for x in nx.astar_path(
            G_mean,
            source=int(origin),
            target=int(destination),
            heuristic=heuristic,
            weight="weight",
        )
    ]
