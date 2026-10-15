# M3.3B dataset and feature schema

The versioned contract is `m3.3b.dataset.v1`, with `m3.3b.problem.v1`, `m3.3b.trajectory.v1`, `m3.3b.features.v1` and `m3.3b.splits.v1`. A generation contains config, complete problem specifications, sample manifest, grouped splits, window index, validation results and executing source bytes. Paths in the dataset manifest are relative to the generation, so the completed dataset can be moved without changing the data. Reproduction of the mechanical fields still requires compatible recorded numerical dependencies.

Each trajectory has `problem.json`, `metadata.json`, `transitions.json` and `arrays/*.npy`. Numerical arrays remain float64 where the FEM uses float64. Iteration/count/mask fields retain integer/bool types. NumPy headers preserve shapes and dtypes; checksums cover the full files. Read-only memory maps support random state/window slices. A bounded loader cache holds at most two open trajectories by default. Arrays are never interpolated or implicitly cast to float32.

For T evaluated states, elemental x/rho/g_x/g_rho arrays have shape `(T,N)`, displacement `(T,ndof)`, compliance and diagnostics `(T,)`, and `update_change` `(T-1,)`. Element indexing is C order `(nely,nelx)`. Nodes are row-major, with alternating x/y DOFs. Iteration t means exactly t completed OC updates. The final updated x receives a FEM evaluation and has no outgoing transition. State zero uses incoming change zero plus an explicit false mask, not an undefined numeric NaN.

Transitions reference states t and t+1 and store the exact update digest, maximum change, validity, active versioned optimizer controls, selected multiplier, bracket endpoints/width, feasible-endpoint indicators and volume/search diagnostics. `Trajectory.transition(t)` returns both x arrays as views and computes the actual update. This avoids duplicate storage of x_t/x_next/delta histories while preserving and checking actual update identity.

Complete physical specifications retain mesh and physical extents, spacings, nodal coordinates, element connectivity/DOFs, actual force vectors, fixed DOFs, material/SIMP/filter definitions, exact initial x, seed, units and active optimizer option. The nested M3.1 mechanical specification retains its canonical baseline OC fields; the outer effective optimizer record is authoritative for `feasible_upper_v1`. No historical default is changed.

# Spatial feature tensors

A sample returns separate `element` and `node` tensors, respectively `(k,C_e,nely,nelx)` and `(k,C_n,nely+1,nelx+1)`. k is history length. Physical rows increase in y; plots use origin lower. Full channel order is stored in `feature_schema.json`.

| Element channel | Units and normalization | Fresh FEM required? |
|---|---|---|
| x | Dimensionless design density | No |
| rho | Dimensionless, computed by the validated local density filter | No |
| center_x, center_y | Physical domain length units | No |
| center_x_over_width, center_y_over_height | Dimensionless physical coordinates divided by domain extent | No |
| volume_fraction | Dimensionless design-volume target | No |
| iteration_over_budget | Absolute update index/configured maxiter; never final trajectory length | No |
| force_scale | Force units; L2 norm of complete actual nodal force vector | No |
| g_x_over_force_scale_squared | Raw design sensitivity divided by known force scale squared | Yes |
| g_rho_over_force_scale_squared | Raw physical-density sensitivity divided by known force scale squared | Yes |
| elastic_energy_over_force_scale_squared | Physical elastic energy per element divided by force scale squared | Yes |

Physical element energy is `0.5*s(rho)*u_e.T*KE0*u_e`. Its sum equals compliance/2 within solver precision. No reference-stiffness quadratic form is mislabeled as physical elastic energy.

| Nodal channel | Units and normalization | Fresh FEM required? |
|---|---|---|
| force_x_over_scale, force_y_over_scale | Dimensionless actual nodal components divided by force_scale | No |
| fixed_ux, fixed_uy | Dimensionless binary masks for each constrained DOF | No |
| node_x, node_y | Physical domain length units | No |
| node_x_over_width, node_y_over_height | Dimensionless physical coordinates/domain extent | No |
| u_x_over_force_scale, u_y_over_force_scale | Raw displacement divided by known force_scale | Yes |

Point loads remain on their exact nodes, including boundary/corner nodes. They are not smeared into elemental images. The sampler rejects requested points that do not coincide with mesh nodes within a 1e-10 index-coordinate roundoff tolerance. Actual resolved coordinates and requested configuration are retained. Directions are normalized before applying the explicitly requested physical magnitude. Model normalization does not rescale the force used in FEM or OC. In particular, an absolute multiplier-width tolerance can respond differently to different physical load magnitudes.

`no_fem_v1` uses the nine elemental base channels and eight nodal base channels. `fresh_gradient_v1` adds g_x, for ten elemental channels. `fresh_full_v1` uses twelve elemental and ten nodal channels. Gradient and displacement inputs require a state at exactly the supplied x and mechanical problem. Historical mechanics can be cached if still associated with those historical x values; reuse at a changed current x is not valid fresh information.

Full problem conditioning remains available through each trajectory's `spec`/`problem` and the problem manifest. The demonstration holds p/material/filter radius/OC policy constant. If a future model varies additional filter or optimizer controls, it must explicitly consume those stored controls as conditioning rather than assume that the existing spatial profile encodes every parameter.

Normalization uses only known per-problem force scale and geometry. It uses neither final compliance, realized trajectory length, held-out statistics nor any future state. `target_x`, `actual_delta`, `target_termination` and the index record are labels/annotations, not spatial input channels.

# Windows, partitions and batching

Window anchors t require `t>=k-1` and `t+h<T`. Inputs cover t-k+1 through t; targets are the actual saved x[t+h]. Demo k values are 1 and 3, with h values 1, 5 and 10. Horizons never extrapolate beyond terminal storage. Window metadata identifies its complete problem, group and partition. A problem cannot appear in multiple partitions.

Declared related families are unioned with conservative buckets over normalized geometry, support pattern and load location/direction. Magnitude, initial seed and resolution variants are grouped; horizontal-mirror clamps and force-sign reversals are grouped conservatively. Bucket boundaries do not prove absence of every possible near-duplicate: new configurations need an explicit family audit. OOD support patterns are held out as entire groups. ID is declared sampler-domain membership, not empirical coverage; the six-problem example is too small to establish coverage. Its validation pin/roller support is absent from the tiny train subset and must not be described as an observed-support holdout.

`WindowDataset` returns writable NumPy feature/target arrays from read-only mmap data. It has a bounded per-process cache and clears the cache during worker serialization. `batch_indices` groups matching mesh and history shapes; use it as a PyTorch batch sampler. Different meshes are never silently resized or padded. Example:

```python
from torch.utils.data import DataLoader
from topopt_utils.fea.trajectory_dataset import WindowDataset, batch_indices

ds = WindowDataset(generation, 'train', 'no_fem_v1', history_length=3, horizon=5)
loader = DataLoader(ds, batch_sampler=batch_indices(ds, batch_size=4, seed=10))
```

No neural model or training code is included. `evaluate_prediction(problem,x,correct=True)` performs one fresh FEM evaluation and optionally one explicit feasible-upper OC correction. Predicted box violations fail; unattainable move-limited volume correction fails with the established OC diagnostic rather than silently repairing density.

# Atomic checkpoint/restart contract

Working frames and transition diagnostics are atomically written, fsynced and hashed. A journal commit exposes a new FEM-evaluated frame and its preceding transition together. Resume validates every committed file and refuses changed source/configuration. Files written before a failed commit are uncommitted and quarantined on recovery. Completed trajectories are published by atomic rename and reused after integrity checks.

Final arrays can be rebuilt from committed frames if finalization was interrupted. Once published, checkpoint frames are redundant and removed; the journal is retained as provenance, not as the completed-sample loader's source of data. Completed arrays provide the terminal design, absolute iteration, incoming change and complete controls needed for an explicitly configured later continuation. Existing completed samples are never overwritten to extend a budget. No cross-library bitwise reproducibility claim is made, and crashes before a commit can incur extra uncommitted FEM solves.
