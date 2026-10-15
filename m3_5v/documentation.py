def readme(r):
 return f'''# Accepted topology dataset visualization

All12 accepted trajectories were visualized:3709 saved states/3697 transitions,8 change-tolerance terminations and4 maxiter caps. Neither class is a certified reference optimum. Source: the unchanged original-only approved M3.4 accepted_dataset_index.json, SHA256 {r['index_sha256']}. Unaccepted expansion cases and the historical plan10 alias are excluded.

Open `trajectory_viewer.ipynb` in VS Code with the Jupyter extension. Select the existing `/users/atabarra/.venv/bin/python` kernel, trust this locally generated notebook and run its cells. The embedded viewer offers two dropdowns, independent saved-iteration sliders, a physical-density/raw-design selector and Play/Pause. Changing a case resets its slider to0 and updates its maximum to the actual saved length. Every saved state is available. If VS Code blocks iframe/script rendering, open the adjacent self-contained `trajectory_viewer.html` in a browser; it has identical controls and requires neither a server nor ipywidgets. Python notebook cells additionally compare any two case indices and display original float64 mmap fields.

Files:

- `dataset_gallery.png`: all12 final physical-density fields with IDs, mesh, exact support/load metadata, target design volume, update count, final compliance and termination.
- `trajectory_snapshots/`:12 sheets with10 representative actual saved indices, including initial, early and terminal states.
- `convergence_plots/`:12 saved compliance/physical and design volumes/incoming change/outgoing OC max update/gradient and mechanical diagnostic panels.
- `animations/`:12 lightweight GIFs with at most64 selected saved states each. Exact indices are recorded in visualization_validation.json; no interpolation/FEM or fabricated state. The viewer exposes every state. GIF playback speed and spacing are presentation choices, not physical time.
- `comparison_00_01.png`: default coarse/fine case comparison using each native mesh and physical coordinates, initial/own midpoint/final and compliance. Notebook/HTML support other selections without resampling.

Orientation: original flattened fields reshape in C order to(nely,nelx), with Cartesian row-y increasing upward. Static plots use origin=lower and extent=(0,width,0,height), equal aspect. PNG/GIF previews reverse row-y once for the top-left raster origin. No transpose or geometry inference by mesh shape is used. Loads come from saved nodal coordinates/components; arrow length is illustrative, direction is exact. Teal vertical support ticks denote fixed x DOFs and orange horizontal ticks fixed y DOFs. Physical coordinates, not cell indices, determine overlays.

Fields and timing: rho is filtered physical density; x is raw design. Fixed visualization scale is[0,1], not a fitted normalization. Physical mean rho and design mean x are separate saved values; the volume budget constrains x. `previous_change[t]` is the incoming max absolute design update for t>=1. State0 has no incoming update; its saved zero is a sentinel. `update_change[t]` is outgoing max|x[t+1]-x[t]|, defined only for t<T-1. These are the same updates with shifted state indices, not two independent convergence measures. Final state has no outgoing action. Compliance, design/physical volume, both gradient norms, projected-gradient and box-stationarity diagnostics, residual and energy/force/moment diagnostics are stored. All32 recorded native fields are listed per case in visualization_validation.json; vector fields remain accessible by read-only Case.array(name). No mechanics were recomputed and no future-state/normalization/training assumptions were introduced.

Memory: native arrays are read-only memory maps; generation processes one case at a time. GIFs retain at most64 small presentation frames for one case; the HTML contains compressed8-bit previews and small scalar metadata, not all raw float64 histories in RAM. Browser displays decode only selected preview images. Notebook provides original float64 access. Recorded rendering process peak:{r['process_peak_RSS_MiB']:.1f}MiB; wall:{r['worker_wall_seconds']:.1f}s. These are visualization measurements, not scientific-worker resource qualifications.

Dependencies already installed: NumPy, Matplotlib and Pillow. All12 GIFs were generated with Pillow; no video dependency or package installation was needed. ipywidgets and nbformat were absent. The notebook was written as standard notebook JSON, and its interaction is self-contained HTML; optional renderer security may require opening the HTML companion. No FEM/OC/topopt_utils/Torch/PARDISO import, optimization, training, SLURM or production operation occurred.

Integrity: all{r['source_files_checked']} source dataset/index files were streamed-hashed before/after; changed0. See source_dataset_hashes_before.json and visualization_validation.json. New visualization artifact hashes are in artifact_manifest.json. Existing research/evidence files were not edited. Re-render only to a new output directory; existing roots are rejected.

From repository root, lightweight checks:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3-5v-mpl /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_5v/test_visualize.py
```
'''
