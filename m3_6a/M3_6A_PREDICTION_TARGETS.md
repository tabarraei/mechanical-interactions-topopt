# Prediction targets

Primary target is Δx[t,h]=x[t+h]−x[t], with h∈{1,5,10}. Candidate future design is x[t]+predicted_delta. Samples require actual saved endpoints; no interpolation, free-running optimizer or crossing termination. State t precedes its outgoing update. The final saved state has no outgoing target.

Targets are design variables, not physical rho; predicting rho would be a different task. `masked_loss` is MSE over explicit valid design cells. No loss fitting or optimizer steps occur here. Finite synthetic backpropagation only verifies differentiability; real forward outputs are untrained. The decoder is unconstrained and may violate bounds/volume. There is no clipping, volume projection, filter application, FEM or independent physical certification of predictions. Any later postprocessing must preserve raw prediction identity and cannot be called an OC update.

All12 source trajectories are training-only diagnostics. Eight stopped at change tolerance, four at maxiter; neither condition creates a reference optimum. No held-out error, speedup, compliance/feasibility result or learned rollout is reported.
