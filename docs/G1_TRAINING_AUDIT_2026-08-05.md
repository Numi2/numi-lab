# G1 Adult Curriculum Audit — 2026-08-05

Scope: read-only review of the authored adult task, native stress path, held-out selector, curriculum supervisor, and status/selector overlays. The audit itself made no code or configuration changes. A follow-up reliability patch was applied afterward and is recorded below.

## Current state

- Mac mini host: `ns-Mac-mini.local`.
- Active run at the last bounded snapshot (16:01:28 CEST): `g1-adult-band2-after-band1-1e-5-seed2650445045-retry1`, 4096 environments, 250 updates, chunk 16.
- Supervisor and native trainer were alive; selection was still pending with zero held-out evidence files.
- No thermal/performance warning was recorded; swap was about 738 MiB used.
- The latest visible training line was update 32/revision 33. This is not adult proof: no candidate has passed the held-out promotion gate, and no physical-robot evidence exists.

## Findings

### High — selector fallback can terminate the curriculum supervisor

`tools/adult_curriculum_supervisor.sh:126-136` calls `numi_curriculum_launch` under `set -e`. The deployed `tools/numi_train_selector_guard.sh:12-29` intentionally returns status 1 after writing an incumbent-retaining `selection.json` when the selector fails. Because the train command is the final command in `numi_curriculum_launch`, that nonzero status can exit the supervisor before it reaches the retry logic at lines 139-157. An inline shell semantics probe reproduced status 1 without reaching the continuation marker.

Impact: a selector failure may leave the run decision on disk but kill the parent curriculum process instead of retrying. This is an operational reliability risk, not a reason to weaken the promotion gate.

### High — stale existing runs can be waited on forever

`tools/adult_curriculum_supervisor.sh:127-131` loops with `sleep 30` until `selection.json` appears, without a timeout, trainer liveness check, or stale-run escape path. A crashed selector or abandoned run can therefore suspend the entire curriculum indefinitely.

### Medium — adult survival gates use indirect reward contributions

The authored adult task declares `standing_completion` and `restoration` as `TaskOutcomeSource::rewardContribution` (`src/core/LocomotionWorld.cpp:1777-1784`). The selector compares their means (`python/metalrobo/policy_selection.py:336-384`) and uses them as monotonic guards. These are useful guardrails, but they are not direct counts of stable standing, recovery completion, duration, or push survival. Promotion can therefore be supported by shaped reward evidence without a fully explicit physical success denominator.

### Medium — clock-stress implementation is not locally verifiable

The repository header declares `MR_TASK_PROGRAM_CLOCK_STRESS` (`include/metalrobo/task_program_types.h:36-40`), but this audit could not verify a matching consumer in the readable local `TaskProgram.cpp`/`LocomotionTask.metal` sources. Those files are marked `compressed,dataless` on this checkout, so this is a verification gap rather than a confirmed runtime defect. The compiled runtime fingerprint and hydrated source must agree before treating biological-clock stress as proven.

### Medium — active runtime/source provenance is split

The active trainer command uses the v5 native library/metallib but the v4 `--python-root`; the v4 Mac mini source tree does not expose the adult-task or clock-stress symbols searched during this audit. This may be an intentional artifact split, but the run currently lacks a directly recorded source revision proving that the v5 binaries contain the authored adult task. Record the runtime build manifest and source fingerprint together before accepting adult evidence.

### Resolved — evidence cache integrity is bound to exact trace content

The selector now publishes a v2 cache record only after validating the complete
canonical trace: exact SHA-256 and byte count, one correctly sized finite row
per requested step, and the expected terminal step. Reuse also verifies the
exact evidence JSON hash and trace manifest; stale v1, truncated, altered, or
partial artifacts are cache misses and are regenerated. Evidence and metadata
are atomically replaced after validation.

## Strengths verified

- Adult curriculum has 11 authored difficulty bands and keeps the previous band in the held-out comparison.
- Adult selection vetoes failed environment steps and previous-band regressions.
- Incumbent deployment is copied before evaluation; rejected candidates are not made parents by the supervisor’s intended state machine.
- The 4096-environment/chunk-16 training configuration is consistent with the measured Mac mini capacity envelope.
- `bash -n`/`sh -n` passed for the three curriculum scripts; Python AST parsing passed for `policy_selection.py`. `shellcheck` is unavailable locally.

## Recommended order, not applied

1. Fix supervisor error handling and add a bounded stale-run/liveness timeout.
2. Hydrate or fingerprint the clock-stress consumer and record it in run provenance.
3. Add direct held-out physical outcomes: stable-standing rate/duration, push survival, recovery completion, and termination causes.
4. Only after the current band completes, qualify selector chunk 1 versus chunk 16 with exact state-trace parity.

## Follow-up reliability patch applied after the audit

- `tools/adult_curriculum_supervisor.sh` now consumes a durable incumbent fallback inside an explicit `if` boundary, preventing `set -e` from bypassing retry logic.
- The same supervisor now bounds pre-existing-run selection waits to 7200 seconds by default (`NUMI_CURRICULUM_SELECTION_TIMEOUT_SECONDS` can override it), instead of sleeping forever.
- `tools/numi_train_selector_guard.sh` returns success only when its incumbent fallback was written successfully, allowing the supervisor to consume the recorded retry decision.
- Both files were backed up on the Mac mini before installation, deployed without stopping the run, and passed remote shell syntax checks.
- Verified SHA-256: supervisor `8905807577a9f27c02074b513fbcd7eda0788dffaacdf9350f7e03f8029122f0`; train overlay `ecb27035876d0658c74493222584a3f1d2cd3b631d63fd684f245ae7193b3105`.
- The known supervisor and native trainer PIDs remained alive after deployment; promotion criteria and held-out gates were unchanged.
- The current run records v5 native hashes plus the v4 bundled train command and selector hashes in `runtime.sha256`, but not the user overlay hash. The deployed overlay now appends its own SHA-256 to `runtime.sha256` for future runs; the current artifact is intentionally unchanged.
