from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_artifacts(data_dir: str):
    edge_features = np.load(os.path.join(data_dir, "edge_features.npy"))  # [M,dE]
    context_features = np.load(os.path.join(data_dir, "context_features.npy"))  # [N,dC]
    tt_matrix = np.load(os.path.join(data_dir, "tt_matrix.npy"))  # [N,M]
    mask = np.load(os.path.join(data_dir, "mask.npy"))  # [N,M] bool
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return edge_features, context_features, tt_matrix, mask, meta


@dataclass
class Scaler:
    mean: float
    std: float

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-6) + self.mean


def compute_global_scaler(tt_matrix: np.ndarray, mask: np.ndarray) -> Scaler:
    vals = tt_matrix[mask]
    mean = float(np.mean(vals))
    std = float(np.std(vals) + 1e-6)
    return Scaler(mean=mean, std=std)


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class ConditionalDenoiser(nn.Module):
    """
    Denoiser epsilon_theta(x_t, t, cond) -> predicted noise
    cond = concat( context_features per-sample broadcasted + edge_features per-edge )
    We implement per-edge conditioning by embedding edges and adding to context embedding.
    """
    def __init__(self, M: int, d_ctx: int, d_edge: int, hidden: int = 512, t_embed: int = 64):
        super().__init__()
        self.M = M
        self.ctx_mlp = nn.Sequential(
            nn.Linear(d_ctx, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_edge, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.t_embed = nn.Embedding(2048, t_embed)
        self.in_proj = nn.Linear(1 + hidden + hidden + t_embed, hidden)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, edge_feat: torch.Tensor):
        """
        x_t: [B,M]
        t: [B]
        ctx: [B,d_ctx]
        edge_feat: [M,d_edge] (fixed)
        returns eps_pred: [B,M]
        """
        B, M = x_t.shape
        assert M == self.M

        ctx_h = self.ctx_mlp(ctx)  # [B,H]
        edge_h = self.edge_mlp(edge_feat)  # [M,H]
        t_h = self.t_embed(t)  # [B,tE]

        # broadcast to edges
        ctx_h2 = ctx_h[:, None, :].expand(B, M, -1)       # [B,M,H]
        edge_h2 = edge_h[None, :, :].expand(B, M, -1)     # [B,M,H]
        t_h2 = t_h[:, None, :].expand(B, M, -1)           # [B,M,tE]
        x_in = x_t[:, :, None]                            # [B,M,1]

        z = torch.cat([x_in, ctx_h2, edge_h2, t_h2], dim=-1)  # [B,M, 1+H+H+tE]
        h = self.in_proj(z)
        eps = self.net(h).squeeze(-1)  # [B,M]
        return eps


class Diffusion(nn.Module):
    def __init__(self, denoiser: nn.Module, timesteps: int):
        super().__init__()
        self.denoiser = denoiser
        self.T = timesteps
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise
        """
        # gather
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)              # [B,1]
        om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)   # [B,1]
        return a * x0 + om * noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor):
        """
        Standard eps prediction loss with missing-mask on edges.
        x0: [B,M] normalized
        mask: [B,M] bool indicating observed edges
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.denoiser(x_t, t, ctx, edge_feat)
        # MSE on observed entries only
        mse = (eps_pred - noise) ** 2
        mse = torch.where(mask, mse, torch.zeros_like(mse))
        denom = mask.float().sum().clamp(min=1.0)
        return mse.sum() / denom

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, ctx: torch.Tensor, edge_feat: torch.Tensor):
        """
        One reverse step (DDPM).
        """
        B, M = x_t.shape
        tt = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        eps = self.denoiser(x_t, tt, ctx, edge_feat)

        beta_t = self.betas[tt].unsqueeze(-1)
        alpha_t = self.alphas[tt].unsqueeze(-1)
        a_bar_t = self.alphas_cumprod[tt].unsqueeze(-1)

        # x0_pred = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
        x0_pred = (x_t - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t).clamp(min=1e-6)

        # posterior mean (simplified)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - a_bar_t).clamp(min=1e-6)) * eps)

        if t == 0:
            return mean

        noise = torch.randn_like(x_t)
        # variance beta_t (rough)
        return mean + torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, edge_feat: torch.Tensor, num_steps: int | None = None):
        """
        Generate samples x0 ~ p(x|ctx) starting from N(0,I)
        ctx: [B,d_ctx]
        returns: [B,M] normalized
        """
        if num_steps is None:
            num_steps = self.T
        B = ctx.shape[0]
        M = self.denoiser.M
        x = torch.randn((B, M), device=ctx.device)
        for t in reversed(range(num_steps)):
            x = self.p_sample(x, t, ctx, edge_feat)
        return x


class SlotDataset(torch.utils.data.Dataset):
    def __init__(self, ctx: np.ndarray, tt: np.ndarray, mask: np.ndarray):
        self.ctx = ctx.astype(np.float32)
        self.tt = tt.astype(np.float32)
        self.mask = mask.astype(np.bool_)

    def __len__(self):
        return self.ctx.shape[0]

    def __getitem__(self, idx):
        return self.ctx[idx], self.tt[idx], self.mask[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="checkpoints/difftravel_run1")
    args = ap.parse_args()

    edge_feat, ctx, tt, mask, meta = load_artifacts(args.data_dir)
    M = tt.shape[1]
    d_ctx = ctx.shape[1]
    d_edge = edge_feat.shape[1]

    scaler = compute_global_scaler(tt, mask)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ds = SlotDataset(ctx, tt, mask)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    denoiser = ConditionalDenoiser(M=M, d_ctx=d_ctx, d_edge=d_edge, hidden=512, t_embed=64).to(device)
    diffusion = Diffusion(denoiser, timesteps=args.timesteps).to(device)

    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    edge_feat_t = torch.from_numpy(edge_feat).to(device)

    _ensure_dir(args.out_dir)

    # Save scaler
    with open(os.path.join(args.out_dir, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": scaler.mean, "std": scaler.std, "tt_unit": "minutes"}, f, indent=2)

    for ep in range(args.epochs):
        diffusion.train()
        total = 0.0
        n_batches = 0
        for batch in dl:
            ctx_b, tt_b, mask_b = batch
            ctx_b = ctx_b.to(device)
            tt_b = tt_b.to(device)
            mask_b = mask_b.to(device)

            # Fill NaNs with mean for stability before normalization (masked out in loss anyway)
            tt_b2 = torch.where(mask_b, tt_b, torch.full_like(tt_b, scaler.mean))
            x0 = scaler.norm(tt_b2)

            B = x0.shape[0]
            t = torch.randint(low=0, high=args.timesteps, size=(B,), device=device, dtype=torch.long)

            loss = diffusion.p_losses(x0, t, ctx_b, edge_feat_t, mask_b)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            n_batches += 1

        avg = total / max(n_batches, 1)
        print(f"epoch {ep+1}/{args.epochs} loss={avg:.6f}")

        # checkpoint each epoch
        ckpt = {
            "model": diffusion.state_dict(),
            "timesteps": args.timesteps,
            "M": M,
            "d_ctx": d_ctx,
            "d_edge": d_edge,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "model.pt"))

    print(f"Saved checkpoint to: {os.path.join(args.out_dir, 'model.pt')}")


if __name__ == "__main__":
    main()