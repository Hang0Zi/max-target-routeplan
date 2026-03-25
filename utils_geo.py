from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from shapely import wkt
    from shapely.geometry import LineString
except Exception:  # pragma: no cover
    wkt = None
    LineString = None


def parse_linestring_wkt(geom_wkt: str):
    """
    Parse WKT geometry into shapely LineString.
    Requires shapely.
    """
    geom_wkt = str(geom_wkt).strip()
    if wkt is not None:
        g = wkt.loads(geom_wkt)
        if g.geom_type != "LineString":
            raise ValueError(f"Expected LINESTRING WKT, got: {g.geom_type}")
        return g

    # Fallback parser (no shapely):
    # Expected format: LINESTRING (x1 y1, x2 y2, ...)
    # Return a list of (x, y) tuples.
    import re

    m = re.match(r"^\s*LINESTRING\s*\((.*)\)\s*$", geom_wkt, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unsupported geometry WKT: {geom_wkt[:80]}")
    body = m.group(1).strip()
    pts: list[tuple[float, float]] = []
    for part in body.split(","):
        toks = part.strip().split()
        if len(toks) < 2:
            continue
        x = float(toks[0])
        y = float(toks[1])
        pts.append((x, y))
    if len(pts) < 2:
        raise ValueError(f"Invalid LINESTRING WKT coords: {geom_wkt[:80]}")
    return pts


def edge_features_from_geometry(geom_wkt: str) -> np.ndarray:
    """
    Extract a small set of deterministic spatial features from a LINESTRING:
    - start_lon, start_lat
    - end_lon, end_lat
    - euclidean length in lon/lat space (not meters, but consistent)
    - num_points (scaled)
    """
    ls = parse_linestring_wkt(geom_wkt)
    if hasattr(ls, "coords"):
        coords = list(ls.coords)
    else:
        coords = list(ls)
    (x0, y0) = coords[0]
    (x1, y1) = coords[-1]
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    eucl = float((dx * dx + dy * dy) ** 0.5)
    npts = float(len(coords))
    # Feature vector
    return np.array([x0, y0, x1, y1, eucl, npts], dtype=np.float32)


def build_edge_feature_matrix(network_df) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build edge feature matrix aligned with edge index (cid order).
    Returns:
      edge_features: [M, d]
      cid_to_eidx: mapping
    """
    cids = list(network_df["cid"].astype(str).values)
    cid_to_eidx = {cid: i for i, cid in enumerate(cids)}
    feats = []
    for geom in network_df["geometry"].astype(str).values:
        feats.append(edge_features_from_geometry(geom))
    edge_features = np.stack(feats, axis=0)
    return edge_features, cid_to_eidx