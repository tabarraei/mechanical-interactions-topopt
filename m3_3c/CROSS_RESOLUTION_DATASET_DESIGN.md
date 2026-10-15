# Proposed M4/M5 cross-resolution contract

This is an experiment design, not an implemented learner or released training population. The M3.3C pilot validates native M3.3B trajectory storage at 48×24 and 96×48. Larger configurations have not undergone FEM or feature construction.

Retain exact native element and nodal tensors and raw arrays. Element tensors use `(history,channels,nely,nelx)` and nodes `(history,channels,nely+1,nelx+1)`; no hidden resize, padding, load redistribution or mechanics change. Partition and batch by complete physical family and mesh. Every history/horizon window, resolution, load-magnitude and start-seed variant of a family belongs to one split. Audit mirrored and nearby mechanics for near-duplicates. This single-family scaling pilot provides no train/test generalization evidence and should not be assigned artificial independent splits.

A proposed versioned conditioning vector must declare physical width/height, dx/dy, physical filter semi-axes, index radii, p, Emin/E0/nu/thickness, volume target, point-load versus traction identity, optimizer version, update cap and stopping policy. These are known controls, not future labels. Physical and normalized coordinates remain separate. Absolute update index and configured progress are retained; normalization cannot use terminal compliance or realized final length. Match optional mechanics features to current x and full problem identity. Report fresh FEM work needed for gradient/full profiles at inference.

Scalar index-radius filters on rectangular cells are physically elliptical. An anisotropic circular-radius adapter would be a separately validated numerical variant, with explicit axis ordering `(R/dy,R/dx)` for image row/column offsets. Do not conflate it with the current scalar DensityMap.

For a future two-grid architecture, retain a nodal branch for exact forces, support masks and optional u, and an element branch for x/rho/gradients. An incident-node/shape-function aggregation can supply additional element embeddings with stated weights, but cannot replace the authoritative nodal load/support arrays. Connectivity may convey information between branches. That mapping changes only neural features; FEM always uses the original nodal point force.

| Experiment | Training | Evaluation | Required separation |
|---|---|---|---|
| Same resolution | Independent families at48 | Held-out loads/supports at48 | Whole-family split |
| Mixed resolution | Independent families at48/96 | Held-out families at48/96 | Every resolution of a family grouped |
| Extrapolation |48/96 only | Frozen model on independent192 families | No test-driven retraining or position-embedding adjustment |
| Domain growth | Separately labeled fixed-spacing data | Held-out larger domains | Physical dimensions/load location conditioning |

Compare multiscale U-Net, temporal U-Net, bottleneck attention and full Transformer only where feasible. Match training examples, mechanics-feature availability, stopping/quality targets and approximately parameters. Report training/inference time, memory and fresh FEM calls rather than parameter count alone. Test physical receptive fields and continuous physical positions under changed spacing. A fixed pixel receptive field has a different physical extent after refinement; learned flattened absolute embeddings require an explicit unseen-grid policy. Attention memory includes heads, batch, history and training intermediates, not just one score matrix.

Predeclare success relative to explicit feasible-upper OC/FEM: comparable or improved compliance at strict volume/box feasibility and independently screened stationarity, stable multi-step rollouts, fewer actual FEM solves and lower end-to-end time at controlled resource limits. Include separate x/rho errors, mechanical checks, failures and censored capped cases, unseen loads/supports/resolutions, and prediction/validation/repair costs. The current20-step pilot provides dynamics data only; selecting useful training/reference labels requires a stopping/quality policy beyond design change.

Fixed-domain comparisons use aligned cell-average restriction and physical integrals. Domain-growth data require separate interpretation and cannot use that restriction across unequal domains. Keep transfer loss, discretization differences and finite-update optimization differences separately labeled; a20-step comparison cannot demonstrate mesh convergence or isolate all three errors.
