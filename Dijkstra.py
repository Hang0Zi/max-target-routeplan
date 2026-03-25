from __future__ import annotations

import networkx as nx
import numpy as np


def build_weighted_graph_min_sample(
    G: nx.DiGraph,
    cid_to_edge_idx: dict[str, int],
    tau_samples: np.ndarray,  # [Omega, M]
) -> nx.DiGraph:
    """
    Dijkstra baseline:
    edge weight = min over sampled travel times.
    """
    w = np.min(tau_samples, axis=0)
    out = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cid = str(data["cid"])
        eidx = cid_to_edge_idx.get(cid)
        if eidx is None:
            continue
        out.add_edge(int(u), int(v), cid=cid, weight=float(w[int(eidx)]))
    return out


def shortest_path_dijkstra(
    G: nx.DiGraph,
    origin: int,
    destination: int,
    cid_to_edge_idx: dict[str, int],
    tau_samples: np.ndarray,
) -> list[int]:
    WG = build_weighted_graph_min_sample(G, cid_to_edge_idx, tau_samples)
    return [int(x) for x in nx.shortest_path(WG, source=int(origin), target=int(destination), weight="weight", method="dijkstra")]
