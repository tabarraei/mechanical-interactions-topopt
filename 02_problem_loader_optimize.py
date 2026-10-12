# This example creates a sample and optimizes it
# This is what generate_trajectory_dataset does, but for a single sample

import io

import imageio.v3 as iio
from IPython.display import Image, display
from inr_render import remesh_from_coords
from topo_transformer import get_problem_loader
from topopt_utils import vis_toolkit as vis
from topopt_utils.topopt_2d import optimize

loader = iter(get_problem_loader(batch_size=1, seed=544536, shuffle=False, width=1, height=2))
batch = next(loader)
sample = remesh_from_coords(batch[0], nels=64)
optimize(sample, penal=3, rmin=2.0, maxiter=80, tolerance=0.01, store_trajectory=True)
vis.plot_data(sample, save_name="02_problem_loader_optimize.gif")
