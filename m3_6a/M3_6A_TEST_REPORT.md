# M3.6A focused test report

Final command (repository root):

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/m3-6a-mpl /users/atabarra/.venv/bin/python -m pytest -q -p no:cacheprovider m3_6a/test_model.py m3_6a/test_data.py m3_6a/test_visualization.py
```

Final actual output is retained in evidence/20261015_architecture_v1/tests_final.log. Coverage: multiple mesh/native shape invariance, fixed shared parameter shapes, conservative nodal mapping, support conditioning, coordinate/aspect checks, synthetic L-mask invariance, causal temporal outputs, variable/right-padded histories, horizons1/5/10, finite backward gradients and masked loss, deterministic initialization, separate Mode B, target poisoning/future mutation isolation, actual saved delta construction, unauthorized historical ID rejection, all12 visualizations, source/termination records, corrupt/missing PNG/GIF rejection, decoded frame count/order and idempotence. Accepted-array access is read-only.

37 passed,1 skipped (CUDA unavailable). One Matplotlib pending-deprecation warning from existing plotting behavior is retained. No test failure in the recorded development runs. Initial model-only run:15 passed/1 skipped; initial combined run:37 passed/1 skipped. Logs distinguish actual runs. GPU performance or compatibility is not asserted.

A separate existing JavaScript control test passed on the new HTML: all12/3709 selectable previews, variable slider bounds, selection reset, terminal timing, rho/x labels, play/pause and independent comparison controls. This is a headless DOM/canvas-stub check, not a claim of manual VS Code rendering inspection.

The seeded real-data example used one shared10,772-parameter Mode A model, CPU16 Torch threads, histories1/3/2 and horizons1/5/10 on48×24,48×32,96×48. All predictions were finite. Timed forward section0.161663s; process peak RSS617.738MiB (includes imports/data verification). These are one diagnostic's measurements, not training resource qualification or an isolated visualization benchmark. No resource safety claim for future training/batches.

Every full-state GIF was actually decoded and matched all3709 quantized RGB frame hashes; first/last native-field rerenders verify actual orientation.108 PNGs/12 GIFs cover all12. Synthetic corrupt fixtures are temporary copies, never authoritative artifacts. Broad integrity has one exception: the old M3.5V notebook changed after baseline and contains executed outputs; attribution is not established, and the notebook was preserved.17053 other protected files match. See integrity_final.json and integrity_exceptions.json for exact paths/hashes; inspected source identities and source/archive acceptance files are separately inventoried. No expensive scientific regression suite was launched.

FEM0, OC0, optimizer/training steps0, numerical replay0, new trajectories0, SLURM0. Backward computes synthetic derivatives through the network without parameter updates. No prediction is independently mechanically validated.
