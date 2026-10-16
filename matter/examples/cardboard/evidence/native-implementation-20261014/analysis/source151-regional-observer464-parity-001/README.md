# Source 151 regional observer parity: native 464 vs 462

This package compares the retained native attempt in `solver/cgls-regional-state-capture-464` against `solver/cgls-block-right-disabled-parity-462`. It is a CPU-only audit; it does not launch Matter or make a convergence/physical claim.

`audit.py` checks native command tails, the exact allowed environment delta, verified frozen source/build identities, the complete 35-file output tree, shared binary sidecars, and all stderr event records. It excludes only the six timing records from each run, the mutually exclusive legacy coupled-observer enabled/skipped markers present in 462, and the regional-capture completion marker present in 464. It normalizes run-local paths only. The common numeric arrays remain exact: 4,096 true-residual-cycle records, 112 Armijo records, and 64 CGLS direction records.

The 35 output paths are identical. Thirty-four files are byte-identical; `output/manifest.json` differs only at `materials.glue_path`, `materials.liner_path`, and `materials.medium_path`, which name equivalent files in the two immutable source snapshots. `stdout.log`, the 409,368-byte base-CGLS payload, the 4,327,464-byte 24-array state payload, and the 13,568-byte first-status5 payload are byte-identical. State JSON adds `micro_generation: 2`; first-status JSON has only the run-local payload path difference. 464 additionally contains the 77,619,992-byte regional payload and its manifest.

The two runs end identically: exit 2, failed result, one accepted step, no timeout or termination error, and unchanged source inputs. The parity verdict is `PASS_FULL_ATTEMPT_PARITY_WITH_OBSERVER_METADATA_DEFECT`; the regional sidecar selection field remains unqualified. This is not a successful solve.

The regional reader validates payload hashes, state join, graph topology/mask/layout checksum, valid base factors, pivots, and finite operator arrays. It does not admit augmented RAP for replay because the consuming restart cycle is absent, and projected material-history currentness remains unavailable. The `primary_policy_selected_environment` raw words are `[0, 0]`; source inspection shows the observer copies that buffer before the later CGLS layout/selection dispatch. The field is stale pre-layout scratch, not authoritative selected-environment metadata. It is retained as an observer metadata defect and is not used to qualify the parity result.

`failed-attempt001.json` and `failed-attempt002.json` preserve two comparator-preflight mistakes and their corrections; no native evidence was changed by either attempt.
