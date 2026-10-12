"""Q4 plane-stress element, SIMP stiffness, and load assembly."""

import numpy as np

# Natural coordinates of the element nodes in the dof order of __get_e_dofs__
# (a = top left, b = top right, d = bottom right, c = bottom left; y measured downward)
Q4_XI = np.array([-1.0, 1.0, 1.0, -1.0])
Q4_ETA = np.array([-1.0, -1.0, 1.0, 1.0])
GAUSS_2PT = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def __get_D__(E=1, nu=.3):
    """Plane-stress constitutive matrix."""
    return E / (1 - nu ** 2) * np.array([[1, nu, 0],
                                         [nu, 1, 0],
                                         [0, 0, (1 - nu) / 2]])


def __get_B__(xi, eta, hx=1, hy=1):
    """Q4 strain-displacement operator at natural coordinates (xi, eta)."""
    dNdx = Q4_XI * (1 + Q4_ETA * eta) / (2 * hx)
    dNdy = Q4_ETA * (1 + Q4_XI * xi) / (2 * hy)
    B = np.zeros((3, 8))
    B[0, 0::2] = dNdx
    B[1, 1::2] = dNdy
    B[2, 0::2] = dNdy
    B[2, 1::2] = dNdx
    return B


def __get_KE__(with_DB=False, E=1, nu=.3, hx=1, hy=1):
    """Q4 plane-stress element stiffness by 2x2 Gauss quadrature of B^T D B.

    With ``hx == hy`` this reproduces the analytic 99-line-topopt KE, which is
    scale invariant: KE depends only on the aspect ratio hx/hy.
    ``with_DB`` also returns D and the centroid B used for stress recovery.
    """
    D = __get_D__(E, nu)
    detJ = hx * hy / 4.0
    KE = np.zeros((8, 8))
    for xi in GAUSS_2PT:
        for eta in GAUSS_2PT:
            B = __get_B__(xi, eta, hx, hy)
            KE += detJ * (B.T @ D @ B)
    if with_DB:
        return KE, D, __get_B__(0.0, 0.0, hx, hy)
    return KE


# Void floor on relative stiffness (not density): E/E0 = emin + (1-emin) ρ^p
SIMP_EMIN = 0.001


def __simp_stiffness__(rho, penal, emin=SIMP_EMIN):
    """Relative Young's modulus from density (SIMP + Ersatz floor)."""
    rho = np.asarray(rho, dtype=np.float64)
    return emin + (1.0 - emin) * (rho ** penal)


def __simp_stiffness_sensitivity__(rho, penal, emin=SIMP_EMIN):
    """d(E/E0)/dρ for ``__simp_stiffness__``."""
    rho = np.asarray(rho, dtype=np.float64)
    return (1.0 - emin) * penal * (rho ** (penal - 1.0))


def __simp_stiffness_second_derivative__(rho, penal, emin=SIMP_EMIN):
    """d^2(E/E0)/dρ^2 for ``__simp_stiffness__``."""
    rho = np.asarray(rho, dtype=np.float64)
    return (1.0 - emin) * penal * (penal - 1.0) * (rho ** (penal - 2.0))


def load_to_forces(load_data, nelx, nely):
    """
    Convert TopOptSample load format to forces array.
    
    Args:
        load_data: Load data from TopOptSample (dict with 'x', 'y', 'fx', 'fy')
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        
    Returns:
        forces: Array of forces for FEA
    """
    import torch as th
    
    if load_data is None:
        return np.zeros((nelx+1)*(nely+1)*2)
    
    forces = np.zeros((nelx+1)*(nely+1)*2)
    
    arrays = []
    for key in ('x', 'y', 'fx', 'fy'):
        values = load_data.get(key, [])
        if isinstance(values, th.Tensor):
            values = values.detach().cpu().numpy().tolist()
        if isinstance(values, (int, float)):
            values = [values]
        arrays.append(values)
    x_coords, y_coords, fx_values, fy_values = arrays
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if i < len(fx_values) and i < len(fy_values):
            # Convert element coordinates to node coordinates
            elem_x = int(x)
            elem_y = int(y)
            
            # Check bounds for element coordinates (0 to nelx-1, 0 to nely-1)
            if 0 <= elem_x < nelx and 0 <= elem_y < nely:
                # Get the 4 corner nodes of the element
                # Element (elem_x, elem_y) has nodes at:
                # (elem_x, elem_y), (elem_x+1, elem_y), (elem_x, elem_y+1), (elem_x+1, elem_y+1)
                node_coords = [
                    (elem_x, elem_y),           # Bottom-left
                    (elem_x + 1, elem_y),       # Bottom-right
                    (elem_x, elem_y + 1),       # Top-left
                    (elem_x + 1, elem_y + 1)    # Top-right
                ]
                
                # Distribute load equally to all 4 nodes
                fx_per_node = fx_values[i] / 4.0
                fy_per_node = fy_values[i] / 4.0
                
                for node_x, node_y in node_coords:
                    node_id = node_y * (nelx + 1) + node_x
                    forces[node_id * 2] += fx_per_node      # x-direction
                    forces[node_id * 2 + 1] += fy_per_node  # y-direction
    
    return forces
