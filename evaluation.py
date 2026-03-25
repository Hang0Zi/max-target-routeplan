from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _parse_list(v: object) -> List[int]:
    if isinstance(v, list):
        return [int(x) for x in v]
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [int(x) for x in obj]
    except Exception:
        return []
    return []


def node_path_to_edge_path(node_path: List[int]) -> List[str]:
    if len(node_path) < 2:
        return []
    return [f"{int(node_path[i])}_{int(node_path[i + 1])}" for i in range(len(node_path) - 1)]


@dataclass
class EmpiricalMatcher:
    by_path_slot: Dict[Tuple[str, str], np.ndarray]
    by_path: Dict[str, np.ndarray]
    by_od_slot: Dict[Tuple[str, str], np.ndarray]
    by_od: Dict[str, np.ndarray]
    min_samples: int

    @staticmethod
    def from_split_df(split_df: pd.DataFrame, min_samples: int = 8) -> "EmpiricalMatcher":
        cols = {"travel_time_min", "path_key", "od_key", "depart_slot"}
        if not cols.issubset(set(split_df.columns)):
            raise ValueError(f"split_df missing required columns {cols}")
        work = split_df.copy()
        work["travel_time_min"] = pd.to_numeric(work["travel_time_min"], errors="coerce")
        work = work.dropna(subset=["travel_time_min", "path_key", "od_key", "depart_slot"])

        def _to_map(gb) -> Dict:
            out: Dict = {}
            for key, g in gb:
                vals = g["travel_time_min"].to_numpy(dtype=np.float32, copy=True)
                out[key] = np.sort(vals)
            return out

        return EmpiricalMatcher(
            by_path_slot=_to_map(work.groupby(["path_key", "depart_slot"])),
            by_path=_to_map(work.groupby(["path_key"])),
            by_od_slot=_to_map(work.groupby(["od_key", "depart_slot"])),
            by_od=_to_map(work.groupby(["od_key"])),
            min_samples=int(min_samples),
        )

    def _prob_leq(self, arr: np.ndarray, t_target: float) -> float:
        if arr.size == 0:
            return 0.0
        n = arr.size
        k = int(np.searchsorted(arr, float(t_target), side="right"))
        return float(k / n)

    def query_on_time_prob(
        self,
        edge_path: List[str],
        od_key: str,
        depart_slot: str,
        t_target_min: float,
    ) -> Tuple[float, str, int]:
        path_key = "|".join(edge_path)
        cands: List[Tuple[str, np.ndarray]] = []
        cands.append(("path+slot", self.by_path_slot.get((path_key, depart_slot), np.array([], dtype=np.float32))))
        cands.append(("path", self.by_path.get(path_key, np.array([], dtype=np.float32))))
        cands.append(("od+slot", self.by_od_slot.get((od_key, depart_slot), np.array([], dtype=np.float32))))
        cands.append(("od", self.by_od.get(od_key, np.array([], dtype=np.float32))))
        for source, arr in cands:
            if int(arr.size) >= self.min_samples:
                return self._prob_leq(arr, t_target_min), source, int(arr.size)
        # fallback to best available even if sample count small
        source, arr = max(cands, key=lambda x: int(x[1].size))
        return self._prob_leq(arr, t_target_min), source, int(arr.size)


def evaluate_requests(
    requests_df: pd.DataFrame,
    planner_paths_df: pd.DataFrame,
    matcher: EmpiricalMatcher,
    output_col_prefix: str = "planner",
) -> pd.DataFrame:
    need_req = {"request_id", "origin", "destination", "departure_time", "T_max"}
    if not need_req.issubset(set(requests_df.columns)):
        raise ValueError(f"requests_df missing columns {need_req}")
    need_plan = {"request_id", "node_path"}
    if not need_plan.issubset(set(planner_paths_df.columns)):
        raise ValueError(f"planner_paths_df missing columns {need_plan}")

    req = requests_df.copy()
    req["request_id"] = req["request_id"].astype(int)
    req["departure_time"] = pd.to_datetime(req["departure_time"], errors="coerce")
    req["T_max"] = pd.to_numeric(req["T_max"], errors="coerce")

    plan = planner_paths_df.copy()
    plan["request_id"] = plan["request_id"].astype(int)
    merged = req.merge(plan[["request_id", "node_path"]], on="request_id", how="left")

    probs: List[float] = []
    sources: List[str] = []
    ns: List[int] = []
    edge_paths: List[str] = []
    for _, r in merged.iterrows():
        nodes = _parse_list(r["node_path"])
        edges = node_path_to_edge_path(nodes)
        edge_paths.append(json.dumps(edges, ensure_ascii=False))
        od = f"{int(r['origin'])}-{int(r['destination'])}"
        depart_slot = str(pd.Timestamp(r["departure_time"]).floor("5min"))
        p, src, n = matcher.query_on_time_prob(
            edge_path=edges,
            od_key=od,
            depart_slot=depart_slot,
            t_target_min=float(r["T_max"]),
        )
        probs.append(float(p))
        sources.append(src)
        ns.append(int(n))

    merged[f"{output_col_prefix}_edge_path"] = edge_paths
    merged[f"{output_col_prefix}_on_time_prob_empirical"] = probs
    merged[f"{output_col_prefix}_prob_source"] = sources
    merged[f"{output_col_prefix}_matched_samples"] = ns
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", type=str, required=True)
    ap.add_argument("--requests_csv", type=str, required=True)
    ap.add_argument("--planner_paths_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--min_samples", type=int, default=8)
    ap.add_argument("--prefix", type=str, default="planner")
    args = ap.parse_args()

    split_df = pd.read_csv(args.split_csv)
    req_df = pd.read_csv(args.requests_csv)
    plan_df = pd.read_csv(args.planner_paths_csv)
    matcher = EmpiricalMatcher.from_split_df(split_df, min_samples=args.min_samples)
    out = evaluate_requests(req_df, plan_df, matcher, output_col_prefix=args.prefix)
    out.to_csv(args.out_csv, index=False)
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
