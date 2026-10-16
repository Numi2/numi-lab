# Bulk FEM transformation and assembly replay

`cardboard_solver_assembly_audit.py` consumes a complete rich capture and the
hash-bound output of `cardboard_solver_pair_analyzer.py`. It executes no native
steps and changes no solver state. The local paper reference retains every
captured high/low deformation and candidate word. The finite elastic starch
reference uses the captured per-environment Lamé parameters.

```sh
python3 matter/tools/cardboard_solver_assembly_audit.py \
  /path/to/capture.jsonl /path/to/pair-analysis /path/to/new-assembly-analysis
```

The audit checks stress and tangent frame rotations, their four-node force
transformations, and the captured production nodal gather. It separately
reports `M delta_v` and `dt² K delta_v`, preserving
`A delta_v = M delta_v - dt² K delta_v` and the negated residual difference.
It compares the independent tangent at the captured material candidate with
the independently reprojected tangent, rather than silently mixing states.
For both perturbation sizes it checks the realized velocity/displacement
relation and residual assembly. A frozen external impulse is inferred from
the center residual; this is explicitly not independent source-force evidence.

This is the captured bulk-plus-mass operator. It does not include an
independent replay of every nonlinear contact or cohesive term. Complete
production contact correctness requires the separate contact diagnostics and
final accepted certificates. The selected Newton state can precede a later
accepted step; it is not labeled a failure merely because an earlier run used
a shorter Newton budget.

Two branch crossings occur in the larger perturbation of capture061. Its
centered action error of 3.83% is therefore not a smooth tangent discrepancy.
The smaller perturbation has no detected trial-yield crossings and gives a
descriptive 0.397% centered-action difference. At that selected bent state,
frame/element errors are below 5.91e-7, captured-force gather error is 2.18e-7,
and the FP64 candidate-state assembled action difference is 3.22e-6. The
reference environment has zero action and remains unresolved. These are
captured-state comparisons, not mesh/time convergence or calibration results.
No empirical noise floor has been admitted; every strict gate stays
inconclusive and the successful diagnostic command returns exit code 3.
