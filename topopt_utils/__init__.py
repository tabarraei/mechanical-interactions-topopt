"""
Codebase adapted from Maze & Ahmed's TopoDiff, a diffusion model approach for topology optimization.
https://github.com/francoismaze/topodiff

TopoDiff was originally a PyTorch port of Dhariwal & Nichol's diffusion model:
https://github.com/openai/guided-diffusion
"""

from .latent_utils import (  # noqa: F401
    compute_latent_scale,
    denormalize_conditions,
    join_latent,
    load_ae_from_checkpoint,
    normalize_conditions,
    split_latent,
)
