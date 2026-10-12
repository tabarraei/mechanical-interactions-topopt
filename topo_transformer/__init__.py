"""Transformer-based INR topology predictor (physics loss and later steps)."""

from .physics_loss import PhysicsLoss, filter_density
from .problem_data import (
    ASPECT_RATIOS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    LOAD_BC_SEPARATION,
    MAX_BCS,
    MAX_LOADS,
    TopOptProblemDataset,
    collate_topopt_problems,
    generate_problem,
    get_problem_loader,
    problem_to_sample,
)

__all__ = [
    "PhysicsLoss",
    "filter_density",
    "ASPECT_RATIOS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "LOAD_BC_SEPARATION",
    "MAX_BCS",
    "MAX_LOADS",
    "TopOptProblemDataset",
    "collate_topopt_problems",
    "generate_problem",
    "get_problem_loader",
    "problem_to_sample",
]
