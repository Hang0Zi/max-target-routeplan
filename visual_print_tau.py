from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _compute_stats(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": float(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "max": float(np.max(x)),
    }


def _fmt_stats(stats: Dict[str, float]) -> str:
    def fv(k: str) -> str:
        v = stats[k]
        if np.isnan(v):
            return "nan"
        return f"{v:.4f}"

    return (
        f"n={int(stats['n'])}, mean={fv('mean')}, std={fv('std')}, "
        f"min={fv('min')}, p05={fv('p05')}, p25={fv('p25')}, p50={fv('p50')}, "
        f"p75={fv('p75')}, p95={fv('p95')}, max={fv('max')}"
    )


@dataclass
class TauTest:
    slot_dir: str
    freq_min: int
    slot_ts: str
    Omega: int
    M: int
    tau_path: str


def _find_meta_jsons(samples_root: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(samples_root):
        if "meta.json" in files:
            out.append(os.path.join(root, "meta.json"))
    out.sort()
    return out


def _load_tau_test(meta_json_path: str) -> TauTest:
    slot_dir = os.path.dirname(meta_json_path)
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    freq_min = int(meta.get("freq_min", -1))
    slot_ts = str(meta.get("slot_ts", ""))
    omega = int(meta.get("Omega", meta.get("Omega".lower(), 0)))
    m = int(meta.get("M", 0))
    tau_path = os.path.join(slot_dir, "tau_samples.npy")
    return TauTest(
        slot_dir=slot_dir,
        freq_min=freq_min,
        slot_ts=slot_ts,
        Omega=omega,
        M=m,
        tau_path=tau_path,
    )


def _load_tau(tau_path: str) -> np.ndarray:
    tau = np.load(tau_path)
    if tau.ndim != 2:
        raise ValueError(f"tau_samples.npy must be [Omega,M], got shape {tau.shape} at {tau_path}")
    return tau


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_root", type=str, default="../outputs/samples_query_1")
    ap.add_argument(
        "--limit_tests",
        type=int,
        default=0,
        help="0 means no limit; otherwise print only first N slot tests.",
    )
    ap.add_argument("--only_freq_min", type=str, default="", help="If set, only print slots with this freq_min.")
    ap.add_argument("--out_csv", type=str, default="", help="Optional output CSV for per-slot stats.")
    args = ap.parse_args()

    samples_root = args.samples_root
    meta_paths = _find_meta_jsons(samples_root)
    if args.limit_tests and args.limit_tests > 0:
        meta_paths = meta_paths[: args.limit_tests]
    if not meta_paths:
        raise FileNotFoundError(f"No meta.json found under samples_root={samples_root}")

    only_freq: Optional[int] = None
    if str(args.only_freq_min).strip():
        only_freq = int(args.only_freq_min)

    per_slot_rows: List[Dict[str, object]] = []
    grouped: Dict[int, List[np.ndarray]] = {}

    for mp in meta_paths:
        t = _load_tau_test(mp)
        if only_freq is not None and t.freq_min != only_freq:
            continue
        if not os.path.exists(t.tau_path):
            print(f"[skip] missing tau_samples.npy: {t.tau_path}")
            continue
        tau = _load_tau(t.tau_path)  # [Omega,M]
        tau_flat = tau.reshape(-1)
        per_omega_mean = tau.mean(axis=1)

        stats_flat = _compute_stats(tau_flat)
        stats_mean = _compute_stats(per_omega_mean)

        rel = os.path.relpath(t.slot_dir, start=samples_root)
        print(f"[slot] freq_min={t.freq_min}, slot_ts={t.slot_ts}, dir={rel}")
        print(f"  tau_flat: {_fmt_stats(stats_flat)}")
        print(f"  per_omega_mean: {_fmt_stats(stats_mean)}")

        per_slot_rows.append(
            {
                "slot_dir": rel,
                "freq_min": t.freq_min,
                "slot_ts": t.slot_ts,
                "Omega": t.Omega,
                "M": t.M,
                **{f"flat_{k}": v for k, v in stats_flat.items() if k != "n"},
                **{f"flat_{k}": v for k, v in stats_flat.items() if k == "n"},
                **{f"mean_{k}": v for k, v in stats_mean.items() if k != "n"},
                **{f"mean_{k}": v for k, v in stats_mean.items() if k == "n"},
            }
        )

        grouped.setdefault(t.freq_min, []).append(tau_flat)

    if grouped:
        print("\n[summary by freq_min]")
        for freq_min in sorted(grouped.keys()):
            all_flat = np.concatenate(grouped[freq_min], axis=0)
            stats_flat = _compute_stats(all_flat)
            print(f"[freq_min] freq_min={freq_min}: tau_flat: {_fmt_stats(stats_flat)}")

    if args.out_csv:
        import pandas as pd

        _ = pd.DataFrame(per_slot_rows).to_csv(args.out_csv, index=False)
        print(f"Saved per-slot stats to: {args.out_csv}")


if __name__ == "__main__":
    main()

