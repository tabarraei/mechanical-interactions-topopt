# Instructions for Codex sessions

Read `PROJECT_CONTEXT.md` before scientific or numerical changes. It describes the inspected working tree, equations, artifact provenance problems, and open research questions. Distinguish source facts, saved observations, hypotheses, and results from checks you actually run. Do not invent previous Copilot discussions.

## Navigate

- FEM/Q4/SIMP derivatives: `topopt_utils/fea/{q4,solver,sample}.py`; OC/filter/trajectory timing: `topopt_utils/topopt_2d.py`.
- Samples/storage: `topopt_utils/topology_data.py`, `generate_trajectory_dataset.py`; coordinate problem sampling/remeshing: `topo_transformer/problem_data.py`, `inr_render/coords.py`.
- M1 generation: `generate_m1_interactions.py`; spatial/scaling: `analyze_m2_spatial.py`, `generate_large_mesh_scaling.py`; state changes: `analyze_m2_topology.py`, `topopt_utils/fea/topology_interactions.py`.
- Loading/plotting: `03_load_and_plot.py`, `03_load_and_plot.ipynb`, `topopt_utils/vis_toolkit.py`. INR/loss/AE utilities are present; a full transformer/training pipeline is not.
- Use `rg --files` and `rg` for discovery. Inspect `git status --short` before work and preserve pre-existing modifications/untracked files. Generated data may be ignored by Git.

## Numerical conventions

- Arrays are `(nely,nelx)` and flatten in C order; nodes are row-major with alternating x/y DOFs. Coordinates, image origin, and element/node indexing must agree.
- Q4 is static plane stress, E0=1, nu=0.3, implicit unit thickness. Stored solver CSR is upper-triangular; reconstruct symmetry for ordinary residual calculations.
- Stiffness is `0.001 + 0.999*rho**p`; the stiffness floor differs from the OC x lower bound 0.001.
- OC uses a boundary-renormalized density filter and its transpose gradient. Its volume budget is on x, not mean filtered rho. The Torch physics-loss filter currently differs at boundaries.
- Stored trajectory frames precede each FEM/OC update; binary topology thresholds the final evaluated rho. Iteration number is optimization time, not physical time.
- P is the PSD mechanical term; H=2P−diag(D) is the physical-density compliance Hessian. Saved interactions are not filtered design-space Hessians or causal measures. R zeros outside the active mask are placeholders; compare states on the common active set and report its size.
- Existing M1 trajectories and interaction evaluations use different domains/load representations. Check full problem identity before drawing conclusions or reusing a trajectory by mesh shape.

## Test

From the repository root, the inexpensive statistics/sampling checks are:

```bash
PYTHONDONTWRITEBYTECODE=1 /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_spatial_interactions.py tests/test_topology_interactions.py tests/test_problem_data.py
```

This passed 16 tests during the 2026-10-12 inspection. `/usr/bin/python` lacked pytest; use an existing suitable environment before proposing dependency installation. Run focused checks appropriate to an authorized change. FEM tests require pypardiso/MKL; `test_m1_interactions.py` generates/optimizes two datasets and plots. Do not silently include it in an inspection-only run. Prior cache IDs and commit messages do not prove test success.

For numerical changes, verify the relevant invariants: force balance/equilibrium, energy/compliance consistency, finite-difference sensitivities/HVP, symmetry/PSD of P, H identity, normalization/masking, filter chain rule, volume and state timing. Do not validate a scientific interpretation merely by asserting the current implementation.

## Preserve research work

- During inspection/documentation, do not modify source, run simulations, submit HPC jobs, overwrite/delete datasets/results, or commit/push. Later explicit user authorization can expand scope.
- Treat `dataset/`, `results/`, `m2_spatial/`, `m2_spatial_large/`, `m2_topology/`, and existing figures/GIFs as research artifacts. For authorized experiments use new output directories and provenance; never enable `OVERWRITE` or run generators into existing roots casually.
- Analysis scripts and notebook cells also write outputs; reading them is safe, executing them can replace figures/summaries. Notebook saved outputs may be stale.
- Keep documentation updates separate from numerical fixes. Ask for scientific clarification when intent is not recoverable, while continuing independent authorized work.
- Do not change MKL's default 16-thread setting without the timing investigation requested by its source comment. Native factorization and dense interaction storage can be expensive; honor memory guards and account for analysis temporaries.
- The final M2.2 entry point is `analyze_m2_topology.py`, which delegates to `m2_topology_refinement.py`, archives existing refinement outputs, and preserves original top-level summaries. Read `m2_topology/refinement/M2_2_REFINEMENT_REPORT.md` for the new numerical evidence and limitations. The full suite passed 29 tests after refinement.
- Historical reporting issues (spatial tail labels/name collisions and legacy topology bin aliases, duplicate Model C, incomplete context loading, transposed overlays) remain in legacy code/artifacts. New refinement outputs correct the topology issues; do not call `_legacy_main` or reuse historical labels as established findings.

## M2.3 spectral work

- Spectral utilities: `topopt_utils/fea/spectral_interactions.py`; saved-array entry point: `analyze_m2_spectral.py`; analytical tests: `tests/test_spectral_interactions.py`.
- Read `m2_spectral/M2_3_SPECTRAL_REPORT.md` and `validation.json` for current spectral evidence. Outputs are Git-ignored and archived on reruns. Do not regenerate exact matrices or replace earlier M1/M2 artifacts for this analysis.
- Analyze R only on active principal submatrices. Use fixed common indices for cross-state eigenspaces; report near-degenerate eigengaps. Small stable rank and high squared-energy retention do not guarantee accurate actions on localized vectors.
- Spectral truncation is PSD; entrywise distance truncation can be indefinite. Compare actual Q/eigenvalue versus CSR storage, application FLOPs, and construction separately. A low-rank-plus-local architecture is a hypothesis, not a validated model.
- Focused check: `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m2-spectral-mpl /users/atabarra/.venv/bin/python -m pytest -q tests/test_spectral_interactions.py`. The final M2.3 run passed 20 focused tests (1.53 s) and 40 current full-suite tests (10.19 s). Full regression includes the existing small FEM and temporary generation tests.
