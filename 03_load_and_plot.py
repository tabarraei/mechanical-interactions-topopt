import pickle
from topopt_utils import vis_toolkit as vis

# This is how you load a .pkl file
with open("dataset/simp_trajectory/topopt_samples_00000.pkl", "rb") as f:
    samples = pickle.load(f)

sample = samples[4]

# These are all the keys in the TopOptSample object
# They can be accessed by calling sample.key

'optimal_topology',
'boundary_conditions',
'load',
'sample_id',
'mesh',
'volume_fraction',
'load_type',
'density_trajectory'

# Get the trajectory

trajectory = sample.density_trajectory

# Plot the sample

vis.plot_data(sample, save_name="sample.gif")
