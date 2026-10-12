#!/usr/bin/env python
import numpy as np
import time

from math import ceil
import matplotlib.pyplot as plt
from scipy import ndimage

from .fea import get_default_solver
from .fea.q4 import load_to_forces


#region: Topology Optimization 2D
def optimize(sample_or_batch, penal=3, rmin=2.0, maxiter=100, tolerance=0.001, optimizer='OC', plotting=False, store_trajectory=False, quiet=False):
    """
    Topology optimization function that accepts a TopOptSample or TopOptBatch.
    Args:
        sample_or_batch: TopOptSample or TopOptBatch containing mesh, volume_fraction, load, and boundary_conditions
        penal: Penalization parameter for SIMP (default: 3)
        rmin: Filter radius in element-count units (default: 2.0; ~5×5 kernel)
        maxiter: Maximum number of iterations (default: 100)
        tolerance: Convergence tolerance (default: 0.001)
        optimizer: 'OC' or 'MMA' (default: 'OC')
        plotting: Whether to plot during optimization (default: False)
        store_trajectory: If True, save each iteration's filtered density on
            sample.density_trajectory with shape (n_steps, nely, nelx)
        quiet: If True, skip the per-sample convergence print
    """
    from .topology_data import TopOptBatch
    
    # Handle batch input
    if isinstance(sample_or_batch, TopOptBatch):
        for sample in sample_or_batch:
            optimize(
                sample, penal, rmin, maxiter, tolerance, optimizer, plotting, store_trajectory, quiet
            )
        return
    
    sample = sample_or_batch
    # Extract parameters from TopOptSample
    import torch as th

    nelx = int(sample.mesh['nelx'])
    nely = int(sample.mesh['nely'])
    dx = float(sample.mesh['width']) / nelx
    dy = float(sample.mesh['height']) / nely
    
    # Convert volume fraction to float if it's a tensor
    volfrac = sample.volume_fraction
    if isinstance(volfrac, th.Tensor):
        volfrac = float(volfrac.item())
    
    forces = load_to_forces(sample.load, nelx, nely)
    fixed_dofs = sample.get_boundary_condition_dofs()
    
    ## initialize 
    # design variables
    x = np.ones(nelx*nely) * volfrac
    rho = np.zeros(nelx*nely)

    # other helper variables
    num_elems = nelx * nely
    fea_time = 0

    solver = get_default_solver()

    # Density filter: rmin is element-count radius (99-line convention).
    # Kernel weights that fall outside the mesh are dropped and the rest
    # renormalized, so zero padding does not pull boundary densities toward 0.
    filter_kernel = __create_filter_kernel_nd__(rmin, rmin)
    filter_weight = ndimage.convolve(
        np.ones((nely, nelx)), filter_kernel, mode='constant', cval=0.0
    )
    trajectory = [] if store_trajectory else None

    # initialize plotting
    if plotting:
        fig, (ax1, ax2) = plt.subplots(1,2)
        plt.ion()
        plt.show()

    for iteration in range(maxiter):
        # use efficient filter to get densities from design variables
        tic = time.perf_counter()
        rho = (
            ndimage.convolve(x.reshape(nely, nelx), filter_kernel, mode='constant', cval=0.0)
            / filter_weight
        ).flatten()
        toc = time.perf_counter()
        filter_time = toc-tic
        if store_trajectory:
            trajectory.append(rho.reshape(nely, nelx).copy())

        # do FEA
        tic = time.perf_counter()
        c, dc = solver.compliance(
            rho, forces, fixed_dofs, nelx, nely, dx, dy, penal=penal, sensitivities=True
        )
        toc = time.perf_counter()
        fea_iter_time = toc-tic
        fea_time += fea_iter_time

        # plot the deformation
        if plotting:
            U = solver.solve(rho, forces, fixed_dofs, nelx, nely, dx, dy, penal=penal)
            ax2.cla()
            ax2.scatter(np.arange(len(U)/2), U[0::2], s=10)
            ax2.scatter(np.arange(len(U)/2), U[1::2], s=10)

        # chain rule of the same filter: dρ/dx = H / Hs, H symmetric
        tic = time.perf_counter()
        dc = ndimage.convolve(
            dc.reshape(nely, nelx) / filter_weight, filter_kernel, mode='constant', cval=0.0
        ).flatten()
        toc = time.perf_counter()
        unfilter_time = toc-tic

        # update design
        tic = time.perf_counter()
        if optimizer == 'OC':
            xold1 = x
            x = __oc_update_design__(x, dc, volfrac)
        elif optimizer == 'MMA':
            raise NotImplementedError(
                "MMA branch archived in topopt_things_archive/topopt_utils_partial/topopt_2d_archived.py"
            )
        else:
            raise ValueError(f"Unknown optimizer {optimizer!r}; only 'OC' is supported.")
        toc = time.perf_counter()
        optimizer_time = toc-tic

        change = max(abs(x-xold1.flatten()))
        
        # plot the design
        if plotting:
            ax1.imshow(x.reshape(nely,nelx), vmin=0, vmax=1)
            plt.show()
            plt.pause(0.001)


        # commented out 
        # print(f'Iteration {iteration:3} complete with maximum change {change:.3f} and c {c:.2f} and v {sum(x)/num_elems:.3f}')
        if change < tolerance:
            break

    if not quiet:
        print(f'converged at iter {iteration} with optimizer {optimizer} and fea time {fea_time:.1f} (avg fea time {fea_time/(iteration+1):.1f})',flush=True)

    # Store result in sample and apply thresholding
    threshold = 0.5
    sample.optimal_topology = (rho > threshold).astype(float).reshape(nely, nelx)
    if store_trajectory:
        sample.density_trajectory = np.stack(trajectory, axis=0)

    return rho
#endregion

#region: Internal Helpers
def __oc_update_design__(x, dc, volfrac):
    '''
    return an updated vector of design variables given x and dc

    This is one of those weird heuristically-derived control laws that
    does converge to a fixed point but doesn't necessarily make sense

    It seems like the gist is we're doing a bisection search to find
    the value of l that pushes us up against the volume constraint.
    l can be interpreted as a lagrange multiplier, and I think the 
    sqrt is just made up.

    All the mins and maxes just enforce a move limit and the allowable
    [1e-3,1] interval for the design variables
    '''
    l_1 = 0.0
    l_2 = 1e5
    move = 0.1

    while (l_2 - l_1 > 1e-4):
        l_mid = 0.5 * (l_1 + l_2)
        # Ensure we don't take sqrt of negative numbers
        sqrt_arg = -dc / l_mid
        sqrt_arg = np.maximum(sqrt_arg, 0.0)  # Clip negative values to 0
        x_new = np.fmax(1e-3, np.fmax( x-move, np.fmin(1, np.fmin(x+move, x*np.sqrt(sqrt_arg)))))

        if sum(x_new) > volfrac * len(x_new): 
            l_1 = l_mid
        else:
            l_2 = l_mid

    return x_new

def __create_filter_kernel_nd__(*rmins):
    """Normalized SIMP kernel: ``max(0, 1 - ||Δi / rmin||)`` on a cubic stencil."""
    rmax = max(rmins)
    kernel_width = ceil(rmax) * 2 + 1
    center = kernel_width // 2
    kernel = np.zeros((kernel_width,) * len(rmins))
    for idx in np.ndindex(kernel.shape):
        dist = np.sqrt(sum(((abs(i - center) / r) ** 2) for i, r in zip(idx, rmins)))
        kernel[idx] = max(0, 1 - dist)
    kernel /= np.sum(kernel)
    return kernel
#endregion
