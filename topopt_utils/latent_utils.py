"""Latent-diffusion helpers for the object-based AE → LDM pipeline."""

from __future__ import annotations

import torch as th

from .unet import AE

# Default train-set-ish stats for the five physical condition slots.
DEFAULT_CONDITION_STATS = {
    "vf_mean": 0.4,
    "vf_std": 0.1,
    "fx_mean": 0.0,
    "fx_std": 1.0,
    "fy_mean": 0.0,
    "fy_std": 1.0,
}


def load_ae_from_checkpoint(ckpt_path, device="cpu"):
    """Rebuild a bare frozen AE from a LitAE checkpoint (no training-module import)."""
    ckpt = th.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    ae = AE(
        vector_size=hp["vector_size"],
        use_variational=hp["use_variational"],
        depths=hp["depths"],
        dims=hp["dims"],
        linear_hidden1=hp["linear_hidden1"],
        linear_hidden2=hp["linear_hidden2"],
    )
    prefix = "AE."
    state = {k[len(prefix) :]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    ae.load_state_dict(state)
    return ae.to(device).eval().requires_grad_(False)


def split_latent(z):
    """Split (B, 2, V) latent into structure (B, 1, V) and conditions (B, V)."""
    structure = z[:, :-1, :]
    conditions = z[:, -1, :]
    return structure, conditions


def join_latent(structure, conditions):
    """Join structure (B, 1, V) and conditions (B, V) or (B, 1, V) into (B, 2, V)."""
    if conditions.dim() == 2:
        conditions = conditions[:, None, :]
    return th.cat([structure, conditions], dim=1)


def normalize_conditions(conditions, nelx=64, nely=64, stats=None):
    """Normalize the five trailing physical slots of a conditions latent (B, V) or (B, 5)."""
    stats = {**DEFAULT_CONDITION_STATS, **(stats or {})}
    out = conditions.clone()
    vf = (out[..., -5] - stats["vf_mean"]) / (stats["vf_std"] + 1e-8)
    x = out[..., -4] / float(nelx)
    y = out[..., -3] / float(nely)
    fx = (out[..., -2] - stats["fx_mean"]) / (stats["fx_std"] + 1e-8)
    fy = (out[..., -1] - stats["fy_mean"]) / (stats["fy_std"] + 1e-8)
    out[..., -5] = vf
    out[..., -4] = x
    out[..., -3] = y
    out[..., -2] = fx
    out[..., -1] = fy
    return out


def denormalize_conditions(conditions, nelx=64, nely=64, stats=None):
    """Invert normalize_conditions on the five trailing physical slots."""
    stats = {**DEFAULT_CONDITION_STATS, **(stats or {})}
    out = conditions.clone()
    out[..., -5] = out[..., -5] * (stats["vf_std"] + 1e-8) + stats["vf_mean"]
    out[..., -4] = out[..., -4] * float(nelx)
    out[..., -3] = out[..., -3] * float(nely)
    out[..., -2] = out[..., -2] * (stats["fx_std"] + 1e-8) + stats["fx_mean"]
    out[..., -1] = out[..., -1] * (stats["fy_std"] + 1e-8) + stats["fy_mean"]
    return out


def compute_latent_scale(ae, loader, n_batches=10, device="cpu"):
    """Empirical std of the structure latent over up to n_batches from loader."""
    ae = ae.to(device).eval()
    vals = []
    with th.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = batch.to(device)
            z = ae.encode(batch, override_variational=False)
            structure, _ = split_latent(z)
            vals.append(structure.reshape(-1))
    if not vals:
        return 1.0
    return float(th.cat(vals).std().clamp_min(1e-4).item())
