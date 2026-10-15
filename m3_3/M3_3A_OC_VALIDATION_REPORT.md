# M3.3A completed generation

Full measured report: [M3_3A_OC_VALIDATION_REPORT.md](generations/20261013T183614565029Z/M3_3A_OC_VALIDATION_REPORT.md).

20 specified problems, 60 common trajectories; all 20 feasible-upper runs satisfy the updated volume budget. Tightening stops earlier on 10 cases; 8 remain capped. Endpoint PG improves on 13 and worsens on 7. The matched feasible-upper call costs1.87 times a single historical call, so precision is not a free speed improvement.

Recommendation: explicitly versioned feasible-upper for the next controlled pilot, with .001 stopping,500 cap and stationarity/feasibility flags. Resolve capped behavior and production timing before100–200 trajectories.20 focused and103 regression tests passed; 835 protected files unchanged.
