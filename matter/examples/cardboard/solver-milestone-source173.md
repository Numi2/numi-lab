# Source173: retain the linear candidate and repair checkpoint arena sizes

Source173 integrates opt-in warm-start CGLS refinement and exact physical-arena
checkpoint validation. All 804 selected repository source files match its frozen
snapshot. The recipe, physical operator, restart 16 ceiling, local tolerances and
final nonlinear certificate are unchanged. Complete coupon histories, resolution
studies, calibration, assembly and crushing remain open.

## Candidate conditioning

At native468/469 control 20/Newton 63, the last FGMRES candidate has fresh true
residual2.03208955e-7. The subsequent zero-start 64-iteration CGLS candidate has
residual5.41764666e-6. Discarding the former is a candidate-selection problem,
separate from correctness of the constitutive derivative. Both complete attempts
still reject control 20.

`NM_FEM_BASE_CGLS_WARM_START_REFINE=1` retains x0, freshly computes b-Ax0, and
solves for a scaled correction. The installer freshly applies A to x0+S*y and
installs it only when its true residual is strictly lower. A declined refinement
preserves x0 and its certificate. An installed correction clears the old
certificate and still needs nonlinear residual descent and final production
acceptance. The option is off by default and requires explicit primary,
all-Newton and block3 controls.

A finite optional-CGLS bad-denominator exit can decline refinement only after a
parallel scan proves all participating vectors, scalars, seed and active scales
finite. Mixed error bits, non-finite state, malformed scales and operator errors
remain fail-closed. No physical law or acceptance tolerance changes.

The source173 native Metal helper passes 264 checks on the original M4 and a
separately built M4 Pro. It uses the production seed, add, finite-breakdown and
install kernels with a synthetic six-DOF operator. It does not execute the full
production CGLS recurrence or qualify a nonlinear trajectory. The earlier 179-check
source169 helper pass and initial failed checker attempt remain retained.

Native472 runs the entire default-off attempt. All 42 observation rows,
40 accepted OBJ files and common non-timing production traces equal native468.
Both reject control 20 with residual5.424712981e-6 against5e-7. This establishes
feature-off output compatibility for a complete failed attempt. Native468's
incomplete initial environment seal remains a provenance limitation.

## Accepted checkpoint serialization

Native470 accepts controls 0..19, then fails checkpoint creation with
`cohesive interface history`. Its accepted observations and geometry equal the
earlier prefix. No usable checkpoint was written; the failure is preserved.

The runtime stores full arenas with physical size
`max(logical_count*sizeof(T), sizeof(T), 16)`. The old reader expected logical
counts. Source173 derives checked record counts from exact physical bytes for
arena fields, retaining overflow/alignment checks. Logical read-count fields
remain strict. Zero-count padding is not physical cohesive history. Actual
native write/restore and continuation parity remain separate gates.

Source173 passes 29 focused CPU cases on the original host. Local available disk
later falls below the unchanged 2 GiB launch floor, so a byte-verified transfer and
fresh source173 build are prepared on macmini, an M4 Pro with 24 GiB RAM. The build
binds 804 source and 1,034 configuration-support files. Two checkpoint CPU tests,
11 environment tests and 264 native helper checks pass there. The first validation
using Python3.9 fails the test API `Path.hardlink_to`; the failure remains beside
the successful rerun with an existing Python3.13.

Prepared original-host commands473/474 remain unexecuted. Fresh host cases476/477
use the same 60-control protocol and material bytes: checkpoint after accepted
control 0, then restore and attempt controls 1..59. Their results retain the new
host identity. A resumed segment alone cannot establish uninterrupted-trajectory
parity or mesh/time convergence.

## Corrected independent material reference

MP004 incorrectly applies medium parameters to a liner and overwrites successful
FP64 rows. Its apparent 7% stress discrepancy and derived large assembled
discrepancy are invalid reference-tool results, not native defects. Both invalid
analyses and their adjudication remain retained.

Corrected MP005 binds each tetrahedron's material index, name, parameter offset
and accepted history. It preserves 1,724 successful paper-center tangents bitwise
and uses independent 70/90/110-digit roots for 17 failed captured-state rows. Four
center roots still fail the strict FP64 projection floor. High-precision fallback
does not turn those failures into FP64 passes. Tiny yield-predicate differences,
endpoint branches and FP32 finite-difference noise remain unresolved.

With the corrected mixed reference at the captured center, all 17,280 native cached
tangent columns have maximum relative error5.722942e-4, element action error
5.363624e-5, and assembled free nodal action error8.507027e-6. Assembly uses the
exact 882-node/1,920-tet geometry, masses, constraints and frames, with A=M+dt²K and
the captured displacement/velocity relationship. These descriptive center checks
meet the relative material/element/assembled gates. They do not establish the
all-FP64 smooth derivative gate, a resolved FP32 noise floor or full nonlinear
convergence. The original031 control 19/Newton 3 failure remains distinct from
this later control 20/Newton 63 capture.

The separate source174 package prepares actual Metal replay of 9,600 captured-state
rows and 17,280 center basis rows, plus bounded epsilon/trace/reference comparisons.
Its CPU admission and compilation are not native material accuracy evidence.

## Shipping continuation

The source170 parallel sparse-contact scan and source175 integration are separate
experimental overlays. CPU contracts and O0/O3 Metal compile/link checks pass.
One shipping environment uses 36,189,188 bytes of sparse pair scratch, below its
unchanged 36 MiB ceiling. Direct native differential and integrated runtime checks
remain separate. The shipping mesh still exceeds 8,192-node/32,768-tet contact/FEM
admission limits; this milestone lifts no cap and proves no 16 GiB process peak.

| Qualification axis | Verdict |
| --- | --- |
| Solver correctness | Selected component evidence; full nonlinear/contact qualification open |
| Numerical convergence | Open; no completed coupon or mesh/time series |
| Component calibration | Open; analogue fits do not identify exact kraft/NSSC damage or local starch fracture laws |
| Analogue validation | Open; conditioning, rate and transfer limits remain |
| Physical assembly in simulation | Open; no scored, folded, bonded, fixture-free shipping checkpoint |
| Crush/impact prediction | Open; no qualified capacity, gravity crush or drop result |

No exact-recipe or impact-validation gap is declared closed. Separate source,
build and host identities, failed histories and prepared commands are retained.
