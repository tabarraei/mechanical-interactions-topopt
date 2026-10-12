"""INR topology rendering: field → dense grid congruent with TopOptSample."""

from .coords import (
    coords_from_grid,
    get_element_centers,
    mesh_spatial_dim,
    remesh_from_coords,
    remesh_from_grid,
)
from .encoding import FourierFeatures
from .field import (
    SIREN_HIDDEN_OMEGA_0,
    SIREN_OMEGA_0,
    SirenTopologyField,
    TopologyField,
    init_siren_packed_weights,
    topology_mlp_from_columns,
    topology_mlp_from_weights,
    topology_siren_from_weights,
)
from .render import render_field

__all__ = [
    "mesh_spatial_dim",
    "get_element_centers",
    "coords_from_grid",
    "remesh_from_coords",
    "remesh_from_grid",
    "FourierFeatures",
    "TopologyField",
    "topology_mlp_from_weights",
    "topology_mlp_from_columns",
    "SIREN_OMEGA_0",
    "SIREN_HIDDEN_OMEGA_0",
    "SirenTopologyField",
    "init_siren_packed_weights",
    "topology_siren_from_weights",
    "render_field",
]
