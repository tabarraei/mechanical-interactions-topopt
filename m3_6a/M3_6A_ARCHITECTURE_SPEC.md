# M3.6A architecture specification

Implemented: a continuous coordinate-conditioned, two-scale quadrature operator, a causal temporal Transformer on spatial anchor tokens, and a native-grid coordinate-kernel update decoder. This is a neural-operator prototype, not the M3.5A CNN under a new name. There are no convolutions. The default Mode A model has 10,772 parameters (width24, four attention heads, one temporal layer, 8 coarse plus24 fine anchors). Its parameters do not depend on element count.

Pipeline: explicit nodal/element/global inputs → pointwise lifting → global and compact-support kernel integrals → bidirectional cross-scale exchange → independent causal temporal attention at each anchor → coarse/local coordinate-kernel decoding, spatial skip and horizon embedding → unconstrained design delta. See MULTISCALE_OPERATOR_DESIGN and TEMPORAL_TRANSFORMER_DESIGN for equations and limits.

Implemented and tested: Mode A; separately configured Mode B; horizons1/5/10; variable native grids, histories and batch padding; explicit synthetic nonrectangular masks; CPU forward/backward; verified original-only data and complete visualization gates. One shared Mode A instance produced finite native outputs on48×24,48×32,96×48. These facts establish executable resolution-aware discretization, not discretization convergence or learned mesh generalization.

Designed but not implemented: accepted nonrectangular-family ingestion, nonzero prescribed-displacement values, passive-cell semantics, protected future-release integration, training/evaluation orchestration and independent validation of predicted candidates. Future hypotheses: learned multistep updates might reduce FEM calls; multiscale tokens might transfer across families. Neither hypothesis was tested. No accuracy, feasibility, optimality or speedup is claimed.

All work is under m3_6a. Original numerical code, protected policies, authoritative acceptance and old visualization outputs remain unchanged. No FEM/OC imports or calls, optimizer steps, numerical replay, SLURM, commit or push.
