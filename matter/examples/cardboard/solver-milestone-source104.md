# Source104: diagnostic parity and the actual failed linear solve

The production changes in this milestone are diagnostic only. Private Armijo
metrics now retain the pre-trial production status and distinguish healthy
comparisons from values computed after a latched failure or in an unused trial
tail. Generic vertex/triangle CCD failures have disjoint diagnostic markers.
The physical operator, contact recurrence, thresholds, iteration limits,
acceptance certificates and accepted-history updates are unchanged.

All 68 build targets and 73 selected CPU tests pass. Native362 repeats the
source103 first-fatal fixture: all 145 center arrays (11,670,948 bytes), all
32 production Armijo records and six physical state payloads match, allowing
only the separately verified serialized device-program identity. No vertex/
triangle marker changes occur in this fixture, so exercising that marker is
still a separate check. The physical fixture continues to reject.

## What the failed coupon now establishes

Native360 captures the actual source101 control2/Newton23 failure. Its
38,105,896-byte capture retains the candidate, accepted history, correction,
mass, constraints, frames, deformation words, material projections, element
actions and residual terms. All 37 ordinary outputs match capture-off352.
The perturbations fail their constitutive checks; their finite differences
remain inconclusive and are not smooth-tangent evidence.

At cycle6, the Arnoldi estimate is 9.7662e-6 Ns against a 1.1993e-5 Ns forcing
target, but the separately recomputed fresh residual is 2.1578e-4 Ns. The
estimate prematurely latches linear completion. Arnoldi uses the cached
algorithmic element action; that observer uses the fresh material path.
Those paths must be compared directly before treating either as an exact
production residual. A prospective repair must test the actual accumulated
correction with the same configured operator and preserve the existing gates.

Native361 captures the regional operator at that exact Newton candidate.
Candidate, history and cached tangents match360 byte-for-byte; its37 ordinary
outputs and64 same-root Armijo rows also match. Independent FP64 assembly gives
relative errors of1.03e-7 for AP,1.38e-7 for coarse-factor reconstruction and
4.26e-7 for the worst local patch factor. These are internal operator/factor
checks, not independent constitutive accuracy or full-solve convergence.
The retained006 audit's older free-only layout constructor does not reproduce
the current two-partition graph; layout generation is explicitly unqualified
by that audit.

A separate FP64 direct solve of the captured cached matrix resolves its linear
residual to1.53e-13 Ns. Its estimated1-norm condition number exceeds2.22e10.
Casting that one solution to FP32 gives1.30e-5 Ns; this sensitivity observation
does not prove a universal FP32 noise floor or that no better representable
solution exists. Global tolerance relaxation remains excluded.

## Retention and qualification limits

Commands, receipts, complete failed captures, analysis failures and lossless
gzip restoration hashes are indexed in
[the evidence archive](evidence/native-implementation-20261014/summary.json).
The source104 public restore package verifies all1,848 files against its fixed
Git base and hashed blobs. Publication verification restored98,112,063 bytes
from257 archived files and matched all778 selected source files to the build.

An evidence-handling error is retained explicitly: the health-overlay author
overwrote its README and five validation logs after sealing. The original
manifest and old hashes survive, but the original log bytes were not recovered.
The overlay source did not change. Independent source104 build, CPU and native
parity receipts remain the supporting evidence; the rewritten logs do not
stand in for the lost originals.

| Verdict | Status |
|---|---|
| Solver correctness | Diagnostic parity and captured algebraic checks pass; actual nonlinear failure remains open. |
| Numerical convergence | Two of27 coupon controls complete; no complete mesh/time/slower-loading series passes. |
| Component calibration | Published kraft/NSSC transfer is versioned; matched progressive damage and starch/hot-melt fracture parameters remain unidentified. |
| Analogue validation | Held-out tests and Robertsson absolute-energy interpretation remain open. |
| Physical assembly in simulation | No completed scoring sequence, frozen score allowance or fixture-free shipping checkpoint. |
| Crush/impact prediction | No qualified Fpeak, frozen heavy masses, gravity crush or drop results. |

The recipe and explicit-flute300×200×200 mm internal FEFCO0201 target remain
unchanged. Shipping arena admission and measured peak memory remain separate
requirements; no unqualified shell or fitted crease replaces the board.
