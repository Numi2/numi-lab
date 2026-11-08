# American crow: coupled object-manipulation life continuation

Source: `8bef3fc5c2b8c9db167261d1dfc3ce1593c5c9ef` on
`agent/american-crow-rl-20261107`, macmini, 8 November 2026.

The ordinary v14 Swift rollout and recurrent MLX trainer now preserve native
physical/contact/object/RNG/GRU state, delayed camera state, next policy, model
and Adam state across process interruption. The existing `NUMILF01` format keeps
three components. V13 requires its finite-food component; resource-free v14
requires that component to be empty. This does not add nutrition, grasping or
social state to the object fixture.

A new publication-only native lifetime-clock API checks accepted submissions
against completed transitions. V13 additionally checks its resource device
clock. V14 uses the accepted native submission stream, rather than inventing a
device food clock. Swift verifies checkpoint manifest ticks immediately after
native restore. Device stepping and inference remain native; no per-step host
readback or alternate simulator is introduced. Existing policy/runtime guards,
32-substep hybrid CCD, fixed learning rate and optimizer diagnostics remain.

## Verification

- Full Release builds pass. Four focused regressions pass: native life
  checkpoint, recurrent rollout, retinal API and temporal history.
- The two-lane object life probe compares three uninterrupted47-tick updates
  with one plus two fresh-process updates at chunks1/8. Five shared optimizer
  boundaries match policies, candidates, rollouts, all41 model/Adam arrays and
  metadata, and all causal native world/GRU/retinal components.
- A scripted recurrent160-tick fixed-policy control matches two fresh113-tick
  continuations from tick47 at chunks1/8. Every future actor/critic observation,
  latent, old log probability/value, transition, GRU input/reset, outcome and
  bootstrap value is byte exact. Final native components and clocks160 match.
  The initial47-tick save retains pending camera state. The physical object
  moves0.273775573m; this is a scripted engineering input, not learned behavior.
- Six invalid restores reject: missing publication marker, resealed wrong
  lifetime ticks, unexpected resource component, corrupted retina, truncated
  world, changed learner discount. No new life checkpoint publishes.
- Current-runtime retinal API admission and the ordinary16-lane v13 coupled
  training regression pass. Four uninterrupted37-tick updates match two plus
  two at chunks1/8 through ticks148, including all checkpoint payloads/model/Adam
  and native components. All nine original rejection controls pass. One lane's
  consumed20000ug persists across restore; all food conservation checks pass.

The original object probe's native runs succeeded, but its checker mistakenly
treated flattened multi-feature arrays as sample rows. Its failed summary,
source snapshot and logs remain archived. The corrected checker reshapes by
sample before selecting the future suffix. Reanalysis validates original
commands and stdout/stderr hashes, reconstructs the exact scripted policy, and
reuses those successful native runs; it does not rerun or select a new outcome.
The six originally planned negative controls then execute once. The corrected
result is `object-life-01/reanalysis-summary.json`; the original failure remains
in `object-life-01/summary.json`.

## Bounded performance and storage

One128-lane run, two160-tick updates, accepts40960 samples at714.781 transitions/s
in57.304s, with a complete coupled checkpoint after each update. Publications
take0.1362115s total. MLX peak is577401924B; native reported retained buffers are
104109234B and transient private buffers76476188B. MLX allocations do not measure
whole-system peak. Swap is zero before/after and physics rejects zero steps.
No controlled comparison or sustained production throughput is claimed.

Each checkpoint directory is131421528B, mostly its116573162B generating rollout.
The380873227B raw performance evidence exceeds the prospective200MiB estimate;
`storage-budget-review.json` preserves the overrun and disposition. All outputs
remain in the archive, including repeats/copies, with content deduplication.
The complete capture covers781 files /1068473472 raw bytes, adding178437581
compressed bytes to Git and reusing prior archive blobs.

`manifest.json` binds every captured source path, hash, length, mode and mtime.
`runtime-recovery.json` identifies exact native executables/library/three shaders,
committed Python sources and canonical visual assets, saved prefix life/policy,
coupled checkpoint and expected futures. `restore_runtime.py` restores these
from actual Git objects. `verify_recovery.py` runs fresh fixed-policy and coupled
learner continuations, then verifies future bytes, native components and all
model/Adam arrays/metadata. Process-dependent time and allocator measurements are
recorded separately. Its final result is `verification.json`.

Raw cleanup is permitted only after committed-object content verification and
actual restored-runtime continuation. The cleanup tool requires remote HEAD
agreement, unchanged hashes/modes/mtimes and closed handles. Its receipt records
the exact deletion set. Archive checkout duplicates may then be omitted while
remaining recoverable from Git. No shared Git provider, unrelated run or prior
unpublished evidence is removed.

## Research limits and next work

This closes engineering persistence for the existing fixed object-push fixture.
Two lanes share one apparatus; they are not independent ecological challenges.
The32/64-substep contact sensitivity from the earlier object-push archive remains
unresolved; this extension does not alter or requalify that physics. Learned
manipulation, broader held-out layouts/materials, solid-food pickup/carrying,
persistent needs and social learning remain open.

The user's optimizer questions are separately addressed in
`artifacts/crow-rl-optimizer-audit-20261108`: final per-factor KL, actual actor
movement after shared clipping, and retained-rollout head advantage/return
comparisons. The audited recipe uses gamma0.999/lambda1. No adaptive learning-rate
trial is selected here. Behavioral selection remains held-out cue-conditioned
choice and intake, rather than KL target attainment.
