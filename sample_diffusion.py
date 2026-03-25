from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from train_diffusion import ConditionalDenoiser, Diffusion
from utils_time import floor_to_slot, parse_ts, ts_to_slot_str


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_slot_idx_map(data_dir: str) -> pd.DataFrame:
    slot_csv = os.path.join(data_dir, "slot_timestamps.csv")
    if not os.path.exists(slot_csv):
        raise FileNotFoundError(f"Missing slot timestamps file: {slot_csv}")
    df = pd.read_csv(slot_csv)
    _need = {"slot_idx", "slot_ts"}
    if not _need.issubset(set(df.columns)):
        raise ValueError(f"slot_timestamps.csv must contain columns {_need}, got {list(df.columns)}")
    df["slot_ts"] = pd.to_datetime(df["slot_ts"])
    return df


def _compute_ctx_from_datetime(slot_ts2: pd.Timestamp) -> np.ndarray:
    """
    Build the same context features as in build_dataset.py:
      [weekday, hour, minute, time_of_day]
    d_ctx is expected to be 4.
    """
    # weekday: 0..6 (Mon..Sun)
    weekday = int(slot_ts2.dayofweek)
    hour = int(slot_ts2.hour)
    minute = int(slot_ts2.minute)
    time_of_day = float((hour * 60 + minute) / (24 * 60))
    return np.array([weekday, hour, minute, time_of_day], dtype=np.float32)


def _load_ctx_for_slot(data_dir: str, slot_ts: pd.Timestamp, freq_min: int) -> tuple[np.ndarray, int, pd.Timestamp]:
    slot_ts2 = floor_to_slot(slot_ts, freq_min)
    try:
        slot_df = _load_slot_idx_map(data_dir)
        row = slot_df[slot_df["slot_ts"] == slot_ts2]
        if len(row) == 0:
            # Inference-time: allow arbitrary slot_ts by computing context on the fly.
            # Note: we do not return a valid slot_idx in this mode.
            ctx_vec = _compute_ctx_from_datetime(slot_ts2)
            slot_idx = -1
            return ctx_vec, slot_idx, slot_ts2
        slot_idx = int(row.iloc[0]["slot_idx"])
        ctx = np.load(os.path.join(data_dir, "context_features.npy"))
        return ctx[slot_idx], slot_idx, slot_ts2
    except FileNotFoundError:
        # If slot_timestamps.csv is missing, we still can infer context purely from time.
        ctx_vec = _compute_ctx_from_datetime(slot_ts2)
        slot_idx = -1
        return ctx_vec, slot_idx, slot_ts2


def sample_slot(
    checkpoint: str,
    data_dir: str,
    slot: str,
    out_dir: str,
    freq_min: int,
    num_samples: int,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    ckpt = torch.load(checkpoint, map_location=device_t)
    timesteps = int(ckpt["timesteps"])
    M = int(ckpt["M"])
    d_ctx = int(ckpt["d_ctx"])
    d_edge = int(ckpt["d_edge"])

    scaler_path = os.path.join(os.path.dirname(checkpoint), "scaler.json")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler.json: {scaler_path}")
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    mean = float(scaler["mean"])
    std = float(scaler["std"])

    edge_feat = np.load(os.path.join(data_dir, "edge_features.npy")).astype(np.float32)  # [M,dE]
    edge_feat_t = torch.from_numpy(edge_feat).to(device_t)

    slot_ts = parse_ts(slot)
    ctx_vec, slot_idx, slot_ts2 = _load_ctx_for_slot(data_dir, slot_ts, freq_min)
    if ctx_vec.shape[0] != d_ctx:
        raise ValueError(
            f"context_features dim mismatch: ckpt d_ctx={d_ctx}, got {ctx_vec.shape[0]}. "
            "If you use a different d_ctx during training, update _compute_ctx_from_datetime() accordingly."
        )

    denoiser = ConditionalDenoiser(M=M, d_ctx=d_ctx, d_edge=d_edge, hidden=512, t_embed=64).to(device_t)
    diffusion = Diffusion(denoiser, timesteps=timesteps).to(device_t)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()

    _ensure_dir(out_dir)

    # Deterministic sampling:
    # - diffusion.sample() uses torch.randn to create the initial noise x; with fixed seed,
    #   the row index (omega) is aligned across different slot samples.
    torch.manual_seed(int(seed))
    if device_t.type == "cuda":  # deterministic-ish alignment for the same seed on GPU
        torch.cuda.manual_seed_all(int(seed))

    ctx_b = torch.from_numpy(np.repeat(ctx_vec[None, :], num_samples, axis=0).astype(np.float32)).to(device_t)

    with torch.no_grad():
        x0_norm = diffusion.sample(ctx_b, edge_feat_t)  # [Ω,M]
        x0 = x0_norm * (std + 1e-6) + mean
        x0 = torch.clamp(x0, min=1e-3)
        tau_samples = x0.cpu().numpy().astype(np.float32)

    np.save(os.path.join(out_dir, "tau_samples.npy"), tau_samples)
    meta = {
        "slot_ts": str(slot_ts2),
        "slot_idx": int(slot_idx),
        "Omega": int(num_samples),
        "M": int(M),
        "tt_unit": "minutes",
        "freq_min": int(freq_min),
        "seed": int(seed),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved tau_samples: {os.path.join(out_dir, 'tau_samples.npy')}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True, help="Artifacts dir from build_dataset.")
    ap.add_argument("--slot", type=str, required=True, help='e.g. "2020-09-23 01:00:00"')
    ap.add_argument("--freq_min", type=int, default=5)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    sample_slot(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        slot=args.slot,
        out_dir=args.out_dir,
        freq_min=args.freq_min,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()

