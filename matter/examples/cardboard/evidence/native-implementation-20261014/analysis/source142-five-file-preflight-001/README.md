# Source142 five-file CGLS preflight

## Finding

The source142 mask-predicate repair is supported by a stage-level reproduction: native446 using the diagnostic checker/helper from source141 reaches `measure_initial`, sets selected status 5 and error bit 16 (`NonzeroConstrainedRow`), then cannot advance. Source142 changes that predicate to reject nonzero residual entries only on constrained rows (`mask.xyz == 0`). Native447 with the source142 helper passes its direct helper checks. This localizes and repairs the CGLS admission bug.

## Source/build binding

Source142 freezes 787 files. Compared with source139, the exact five changed/added paths are in `report.json`; compared with source141, only `matter/src/metal/base_cgls.metalinc` changes. Its one-line semantic change is the mask comparison described above. The source142 build record matches every frozen source hash, exits 0, and records the probe target. The root reports the full 69-target build and 74 CPU tests passing.

## Native evidence

- **443:** original direct checker exits 1. It uses the source141 helper and source material documented in the 443 `source-overlay.json`.
- **446 attempt 002:** diagnostic stage capture exits 1 without timeout; all input hashes verify against the 443 overlay. Initial state, transpose action, and vector setup succeed; `measure_initial` fails on constrained-row validation (status 5, error bit 16).
- **447:** repaired helper exits 0, 678 checks, 19 CGLS iterations, residual 4.63681 to 1.3328e-7, maximum direction error 5.96046e-8.

These are helper/recurrence and synthetic CSR checks. They do not qualify the production FEM tangent action or complete a physical nonlinear step. The selected source142 native450 full-probe run was still active during this audit and is not adjudicated here.
