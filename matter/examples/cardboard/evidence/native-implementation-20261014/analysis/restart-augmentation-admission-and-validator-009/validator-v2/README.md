# Restart augmentation validator v2

This is an offline diagnostic validator for same-Newton regional restart augmentation. It checks source-bound rich operator captures, recomputes fresh base `A*P` actions, validates per-environment generation/attempt semantics, and for an actual attempted environment checks fresh `A*q`, scale/orthogonality, and pivoted nonsymmetric RAP/LU reconstruction. It does not alter solver operation or certify convergence, native benefit, or physical behavior.

Run the CPU tests with:

```sh
python3 -m unittest -v /Users/home/cardboard-implementation-20261014/analysis/restart-augmentation-admission-and-validator-009/validator-v2/test_restart_augmentation_validation_v2.py
```

`capture323-base-ap-v2.json` verifies the feature-off 323 capture's base operator at its exact captured center. Fresh `A*P` max relative errors are `5.034e-7` and `5.240e-7` for environments 0 and 1. This native run rejected before accepting any physical step.

`capture182-base-ap-v2.json` deliberately preserves the prior mismatch: conditioning P and effective P match the rich sidecar, but AP does not. Fresh recomputation finds an environment-0 AP max relative error of about `0.99999945`; environment 1 is about `5.18e-7`. The validator exits 2 and retains the mismatch instead of replacing it with the recomputed values.

`capture337-restart-diagnostic-v2.json` audits the augmented-on N=16 source-100 run. Its base A*P is consistent (max relative errors `5.189e-7` and `5.240e-7`), but it records **no augmentation attempt**: both environments have zero attempt count, `UINT32_MAX` generation, zero mode-valid, zero q/Aq and attempt scalars. Environment 0 has `factor_valid=1`, while environment 1 has 0; both augmented RAP/pivot payloads are zero. The validator reports that factor bit as fallback/scratch metadata only, never as evidence that q existed or its augmented factor was valid. The rich run shows at most 15 Arnoldi columns with restart 16, so these captures do not exercise a second restart cycle. The root-generated on/off parity report confirms identical bounded outputs for 337/338, but this is not an augmentation-benefit result.

The captured runs are diagnostic only. The native runs failed with zero accepted steps and `physical_validation=false`; no solver performance, physical assembly, or material qualification follows. A later run must produce a non-sentinel per-environment generation and positive attempt count before q/Aq/RAP gates can apply. No CPU-generated result is promoted into native evidence.

## Validator semantics

The generation tuple is the attempt latch. `UINT32_MAX` in every generation word plus zero attempt count and zero mode-valid means no attempted q. In that case q/Aq and attempt-owned scale inputs must be exactly zero. The factor-valid word and augmented RAP/pivots are reported as unattempted fallback/scratch fields and are not used to claim augmentation validity. For a non-sentinel generation, the validator requires matching captured control/Newton identity, positive attempt count, mode/factor validity, and checks q/Aq and augmented RAP against the rich capture's same-center operator.

Thresholds are hard-capped at the v2 defaults; CLI settings cannot weaken them. The retained synthetic test exercises the full positive branch and the quiet-environment/fallback path, but is not native evidence.
