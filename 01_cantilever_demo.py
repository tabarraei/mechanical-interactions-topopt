"""Cantilever demo of the SIMP / optimality-criteria optimizer."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from topopt_utils.topology_data import TopOptSample
from topopt_utils.topopt_2d import optimize


def cantilever(nelx=240, nely=40, volfrac=0.5):
    bc = np.zeros((nely + 1, nelx + 1, 2))
    bc[:, 0, :] = 1.0

    ## This is my custom data structure for storing the sample
    return TopOptSample(
        mesh={"nelx": nelx, "nely": nely, "width": float(nelx), "height": float(nely)},
        load={"x": [nelx - 1], "y": [nely // 2], "fx": [0.0], "fy": [-1.0]},
        boundary_conditions=bc,
        volume_fraction=volfrac,
        sample_id="cantilever",
    )


def main():
    sample = cantilever()

    ## All topopt happens here
    rho = optimize(sample, penal=3, rmin=2.0, maxiter=80, tolerance=0.01)


    ## Plotting
    nelx, nely = sample.mesh["nelx"], sample.mesh["nely"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].imshow(rho.reshape(nely, nelx), cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("Filtered density")
    axes[1].imshow(sample.optimal_topology, cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("Thresholded at 0.5")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("cantilever_simp.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
