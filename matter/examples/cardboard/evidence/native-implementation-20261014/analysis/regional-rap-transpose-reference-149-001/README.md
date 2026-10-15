# Regional RAP right-map and transpose reference (source149)

This CPU-only package audits the source149 regional outer-FGMRES RAP preconditioner and gives an independent mathematical reference for applying it as a right map `S` and applying the exact algebraic transpose `Sᵀ`. It does not change production files or run Metal/GPU work.

Source149's selector 3/4 action is

```text
c = R⁻¹ Pᵀ r,      R = Pᵀ A P,      Q = A P
S r = P c + B (r - Q c)
```

where `B` is either the marked per-node 3×3 inverse block (row0.w > 0.5), with scalar inverse-diagonal fallback for unmarked nodes, or the scalar diagonal map for the non-block selector. Reversing that composition gives

```text
Sᵀ y = Bᵀ y + P R⁻ᵀ (Pᵀ y - Qᵀ Bᵀ y).
```

The coarse RAP factor is generally nonsymmetric. Its stored final row permutation is consumed as `rhs[pivots[row]]`; the transpose solve therefore solves `Uᵀ`, then `Lᵀ`, then scatters through the inverse permutation. The CPU checks include nontrivial pivots and nonsymmetric `A`, `R`, and nodal blocks, comparing against an independent dense/Gauss-Jordan oracle and checking the adjoint dot identity.

The reference handles source149's base mode count with padded augmented stride, current-restart augmented modes, invalid-factor fine-only fallback, invalid augmentation with zero extra `P/AP`, selected/healthy state, fixed masks, scalar-diagonal fallback, and strict source/world/program/candidate plus microtick/microgeneration/dt/Newton/restart-generation binding. It never borrows a terminal RAP factor for another generation.

This is a mathematical/CPU reference, not native kernel parity. For a native comparison, use the *same captured generation* and exact `effective_P`, `AP`, LU, pivot permutation, validity, block inverse rows, scalar fallback diagonal, row mask and health/selection state. Compare production forward and manufactured-vector outputs to `apply_regional`; compare the transpose reference by independent adjoint tests unless a production transpose kernel exists.

Scope distinction: source149's CGLS recurrence currently uses a separate node-local block3 scale `S` (initial `Sᵀ Aᵀb`, forward `A(S p)`, gradient `SᵀAᵀr`). It does not apply regional RAP inside CGLS. Source149 has no regional RAP transpose kernel.

Selector 5 is deliberately unsupported in this package. The exact source path is audited, and its real-arithmetic map can be written as `H = G D E`, `S = H + (P - H AP) R⁻¹ Pᵀ`. `E` extracts each patch's local residual; `D` is the graph patch solver including its one residual-refinement pass; `G` averages overlapping patch-node outputs by the captured per-node coverage. Its transpose must reverse this composition and use the transpose of each local refined solve. That derivation alone is not a same-generation replay. The owner confirmed that the source151 sidecar format can carry the required graph state, but no selector-5 same-generation capture exists yet. The exact required same-generation arrays, identity fields, and additional native-output limitation are recorded in `source-contract.json` and `report.json`.

Run the CPU tests with:

```sh
cd /Users/home/cardboard-implementation-20261014/analysis/regional-rap-transpose-reference-149-001
/Users/home/.pyenv/versions/3.10.12/bin/python3 audit_source_contract.py
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v test_regional_rap_reference.py
```

Results and exact source bindings are recorded in `report.json` and `manifest.json`. The test suite covers only selectors 3/4; it does not provide selector-5 evidence.
