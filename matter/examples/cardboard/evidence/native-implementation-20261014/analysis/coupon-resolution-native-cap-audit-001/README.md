# Source-bound coupon resolution mesh/cap audit 001

This is a CPU-only preflight of the mesh ladder already declared by the source-118 resolution-study planner, bound to native run 392. It checks exact cross-section mesh counts and the runtime's budgeted selector-5 graph-Schwarz layout for all 27 combinations of `nx_per_pitch={8,16,32}`, `ny={2,4,8}`, and `thickness_slices={1,2,4}`. No shared source, native solver, Metal library, or GPU was changed or run.

## Result

The actual runtime entry is `runtime.mm:1331`, which calls `buildBudgetedRegionalGraphSchwarzLayout`. The primary `budgeted-results.csv` uses this function and the runtime's 2-environment layout. The complete independent-axis sequences each pass graph-layout admission on CPU:

- `nx`: `(8,2,1)`, `(16,2,1)`, `(32,2,1)`; the finest node/tet counts are 2,610/5,568 and the graph arena estimate is 99,840,612 bytes, below the 96 MiB limit (100,663,296 bytes).
- `ny`: `(8,2,1)`, `(8,4,1)`, `(8,8,1)`; the finest counts are 2,646/7,680 and graph arena estimate is 50,717,580 bytes.
- thickness: `(8,2,1)`, `(8,2,2)`, `(8,2,4)`; the finest counts are 2,430/7,680 and graph arena estimate is 95,828,780 bytes.

Mixed mesh combinations can fail before the end of the full 3×3×3 grid. The 128-patch limit rejects `(8,4,4)`, `(16,2,4)`, `(32,2,2)` and other combinations shown in the CSV. `(16,2,2)` is admitted by the budgeted layout. The 96 MiB arena rejects `(16,4,2)` at 284,391,332 bytes. The graph compiler/runtime node limit rejects `N>4096`; this includes `(16,8,1)` and the combined refinement `(32,8,4)`. The combined case has 20,682 nodes and 89,088 tetrahedra, so selector 5 stops at its explicit node-count guard before graph layout construction. All candidate rows also include the source fixed-mask node count; the baseline helper matches 392 exactly at 36 fixed nodes. The planned time-refinement triplet uses `(32,8,4)` for all three time levels, so each time case is blocked by the same count guard. Its authored schedule is `dt={100,50,25} µs` and phase step counts scale by `{1,2,4}`; loading/hold/unloading/relaxation/release durations remain `{1.2,0.3,0.6,0.3,0.3} ms`. The study planner also scales the relative residual tolerance by `{1,1/2,1/4}` to keep tolerance/dt fixed as declared in the plan; this audit does not assess whether that choice is sufficient numerical error control. No graph arena estimate is claimed for rejected-before-layout meshes.

Native run 392 was itself not a completed baseline: it used `(8,2,1)` but rejected its first bent step with status 5 at tetrahedron 858 (`J=0.9977478981`) and `accepted_steps=0`. The held-reference environment advanced, but the process specimen did not. Thus the layout pass only establishes preflight resource admission, not a usable numerical/convergence baseline or a physical result.

The independent axis tests exercise the current bounded graph layout builder, not the complete native mechanics. `deviceBytes` is the builder's checked GPU-arena estimate, including per-environment dense patch factors and a retained source-matrix workspace, pivots, corrections, validity, and metadata; it is not measured allocation or peak memory. Runtime initialization also checks each buffer against `MTLDevice.maxBufferLength` and 32-bit environment-major indices. The 96 MiB layout limit, 4,096 node limit, 32,768 tet limit, 128 patch limit, and 256 node-per-patch maximum are retained; this audit does not recommend raising them.

`contact_capacity_if_contact_enabled=8*T` and its unrelated full-contact-cap comparison column are reported only as separate capacity indicators. They do not gate these selector-5 runs: the planned study is explicitly contact-free, and graph Schwarz preflight rejects contact/cohesive/rigid/multiphysics scopes. Do not apply the full-contact `N/C` admission limits to this selector-5 path or treat them as bypassed for contact runs.

## Superseded first pass

`nonbudgeted-superseded.csv` is preserved from the first pass, which called `buildRegionalGraphSchwarzLayout` directly and omitted the runtime's bounded fallback (`buildBudgetedRegionalGraphSchwarzLayout`, which retries smaller patches and groups uncovered nodes). That first pass incorrectly reported several individual-axis cases as rejected. It is retained for audit history only. `intermediate-budgeted-results-before-final-columns.csv` preserves the next intermediate budgeted result before the final CSV header/fixed-mask columns were added. The corrected primary table uses the exact budgeted runtime builder and call arguments.

## Reproduction

Run from this directory:

```sh
/usr/bin/clang++ -std=c++20 -O2 -Wall -Wextra -I. resolution_cap_audit.cpp -o resolution_cap_audit
./resolution_cap_audit > budgeted-results.csv
```

This executable rebuilds only the deterministic mesh-count and graph-layout helper. The copied input headers and source/build/run hashes are listed in `bindings.json`. `source118-cardboard_resolution_study.py` is included as the exact planner source for the declared mesh/time ladder.

## Next execution requirement

First obtain a complete, accepted `(8,2,1)` physical/protocol baseline under strict unchanged solver criteria. Then keep selector-5 eligibility and current hard bounds, but redesign its bounded coarse/local layout so `(32,8,4)` can be admitted without an unbounded dense patch matrix or raised count/arena caps. The present all-node dense patch representation has a sum-of-squares cost in patch DOFs; a valid extension needs a bounded sparse/streamed or multilevel representation with fresh native operator/preconditioner parity and resource measurements. The current report does not choose or validate such an implementation. Even if spatial/time numerical gates become accessible, the end-gripped coupon omits scored folds, contact and damage activation; a separate full-stage comparison remains necessary.
