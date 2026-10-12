"""Unpack TopOptSample fields and compute sample-level FEA quantities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .q4 import __get_KE__, __simp_stiffness__, load_to_forces
from .solver import FeaSolver, get_default_solver


@dataclass
class UnpackedSample:
    nelx: int
    nely: int
    dx: float
    dy: float
    rho: np.ndarray
    forces: np.ndarray
    fixed_dofs: np.ndarray


def unpack_sample(sample, use_solid_material: bool = False) -> UnpackedSample:
    """Extract FeaSolver arrays/scalars from a TopOptSample. Does not solve."""
    import torch as th

    nelx = int(sample.mesh["nelx"])
    nely = int(sample.mesh["nely"])
    dx = float(sample.mesh["width"]) / nelx
    dy = float(sample.mesh["height"]) / nely
    forces = load_to_forces(sample.load, nelx, nely)
    fixed_dofs = sample.get_boundary_condition_dofs()

    if use_solid_material or not sample.has("optimal_topology"):
        rho = np.ones(nelx * nely)
    else:
        topology = sample.optimal_topology
        if isinstance(topology, th.Tensor):
            topology = topology.detach().cpu().numpy()
        rho = topology.flatten()

    return UnpackedSample(
        nelx=nelx,
        nely=nely,
        dx=dx,
        dy=dy,
        rho=rho,
        forces=forces,
        fixed_dofs=fixed_dofs,
    )


def _compliance_one(sample, penal: float, solver: FeaSolver) -> float:
    data = unpack_sample(sample)
    c = solver.compliance(
        data.rho,
        data.forces,
        data.fixed_dofs,
        data.nelx,
        data.nely,
        data.dx,
        data.dy,
        penal=penal,
    )
    sample.compliance = c
    return c


def compliance_sample(
    sample,
    penal: float = 3,
    solver: Optional[FeaSolver] = None,
) -> Union[float, List[float]]:
    """Structural compliance ``FᵀU`` for a sample, list of samples, or TopOptBatch."""
    from ..topology_data import TopOptBatch

    solver = solver or get_default_solver()

    if isinstance(sample, TopOptBatch):
        samples_list = sample.samples
        values = [_compliance_one(s, penal, solver) for s in samples_list]
        if samples_list:
            sample._field_names = samples_list[0].get_available_fields()
        return values

    if isinstance(sample, (list, tuple)):
        return [_compliance_one(s, penal, solver) for s in sample]

    return _compliance_one(sample, penal, solver)


def _energy_and_stress_one(
    sample, penal: float, use_solid_material: bool, solver: FeaSolver
) -> Tuple[np.ndarray, np.ndarray]:
    data = unpack_sample(sample, use_solid_material=use_solid_material)
    KE, D, B = __get_KE__(with_DB=True, hx=data.dx, hy=data.dy)
    U = solver.solve(
        data.rho,
        data.forces,
        data.fixed_dofs,
        data.nelx,
        data.nely,
        data.dx,
        data.dy,
        penal=penal,
        KE=KE,
    )
    element_dofs = solver.get_connectivity(data.nelx, data.nely).element_dofs
    U_elements = U[element_dofs]
    stiff = __simp_stiffness__(data.rho, penal)
    element_energy = stiff * np.einsum("ij,jk,ik->i", U_elements, KE, U_elements)
    stress = stiff[:, None] * (U_elements @ (D @ B).T)
    von_mises = np.sqrt(
        stress[:, 0] ** 2
        + stress[:, 1] ** 2
        - stress[:, 0] * stress[:, 1]
        + 3.0 * stress[:, 2] ** 2
    )
    strain_field = element_energy.reshape(data.nely, data.nelx)
    stress_field = von_mises.reshape(data.nely, data.nelx)
    sample.strain_field = strain_field
    sample.stress_field = stress_field
    return strain_field, stress_field


def energy_and_stress_sample(
    sample,
    penal: float = 3,
    use_solid_material: bool = False,
    solver: Optional[FeaSolver] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    """Element strain energy and von Mises stress for a sample, list, or TopOptBatch."""
    from ..topology_data import TopOptBatch

    solver = solver or get_default_solver()

    if isinstance(sample, TopOptBatch):
        samples_list = sample.samples
        values = [
            _energy_and_stress_one(s, penal, use_solid_material, solver)
            for s in samples_list
        ]
        if samples_list:
            sample._field_names = samples_list[0].get_available_fields()
        return values

    if isinstance(sample, (list, tuple)):
        return [
            _energy_and_stress_one(s, penal, use_solid_material, solver)
            for s in sample
        ]

    return _energy_and_stress_one(sample, penal, use_solid_material, solver)

