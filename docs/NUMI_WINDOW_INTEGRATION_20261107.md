# Numi Window integration research checkpoint — 7 November 2026

Status: **work in progress, not a unified release**. This branch selectively
integrates the native window and training UI from `origin/numisolver`
`698dd34fb77ba93d4a82a826e548fbd28b735371` onto main
`304fd4be5013effeb997813139c24fce811cd988`. Their merge base is
`f52e88cb0f0ff6acdb4d81c904929dd4f30576ef`. It does not merge branch build
artifacts. The independent offline refinement check was restored first.

The first native compilation failed because the main C interface did not
include measured-dove, Numifly, interaction-push, and presentation fields
required by the newer window. The first catalog compilation also lacked
visual-link transforms and terrain-height access. The exact failure logs are
retained under `docs/evidence/window-integration-20261107/`. Those interfaces
have since been selectively integrated and the native rollout, training,
window-check, catalog, Booster compiler, and measured-surface probe targets
have linked. This is build evidence, not end-to-end qualification.

## Formats and transaction boundaries

Both branches assigned actuator wire tag 7 but to different physics: main
uses Millard excitation and numisolver uses measured-surface mechanics. This
checkpoint keeps Millard 7, assigns measured surface 8, writes actuator pack
version 3, and sets task-program GPU ABI 44. A version-2 pack with tag 7 is
rejected with an explicit ambiguity error unless the caller supplies verified
branch provenance. `metalrobo_actuator_pack_migrate main|numisolver INPUT
OUTPUT` writes a new version-3 pack; it does not infer the source. Nonambiguous
version-2 tags remain readable. The migration test checks both mappings,
round trips, atomic ambiguity rejection, and an unambiguous tag. It also
reads and round-trips minimal artifacts produced by the original branch
writers. The numisolver writer itself rejected tag 7, so the numisolver
tag-7 mapping remains a synthetic compatibility case.

The inspected source branches serialize task-pack versions 17 and 19 using
the same field sequence. This checkpoint writes version 20 and reads those
two versions explicitly. The CPU test round-trips minimal artifacts made by
both original branch writers; it does not establish compatibility with every
historical user pack. The window's metadata reader accepts policy formats 4
and 5, including exact version-2 multi-task bindings. CPU tests exercise a
native writer-generated version-5 policy and malformed synthetic cases; the
native runtime remains responsible for full content and semantic validation.

Measured-surface mechanics owns private state outside MetalWorld. A resident
continuation checks the same owner and mechanics fingerprint; mutable owners
are reserved to one executor pool and one live resident. An unsubmitted
initialization is invalidated on abort; a failed submitted command poisons
the owner. Simultaneous device mechanics and device physics (including Matter)
are explicitly rejected pending a verified joint transaction contract.
MetalWorld's resident-state fingerprint returns an explicit error for external
mechanics or device physics because it cannot digest their external state.
Observation and action callbacks have separate unresolved external ownership,
as described below. There is no external surface save/restore implementation
in this checkpoint.

## Retained checks and limits

- The CPU catalog metadata check passed version-4, native-written version-5,
  multi-task matching, malformed inputs, and scene-catalog cases.
- The CPU migration test passed synthetic and historical producer fixtures
  for both lineages, including format round trips.
- On Apple M6, `metalrobo_measured_surface_robot_probe --ownership
  assets/numifly/maeda-wing-pack-v1/manifest.json` passed with one environment
  and two steps. Two fresh executions produced bit-identical generalized
  state and accepted surface state/evidence; the probe also exercised owner,
  fingerprint, concurrent use, mixed-owner, and unsupported-digest gates.
  The output and SHA-256 hashes of the exact binary, library, metallib, selected
  source files and model inputs are retained in the evidence directory.
- The short repeat is independent re-execution, not restoration from a saved
  external mechanics state. It does not qualify physical fidelity, force
  accuracy, long-duration stability, training, rendered UI, or throughput.
- A second bounded M6 ownership probe passed the in-flight inspection gate:
  inspection was rejected while the submission was pending and succeeded
  after completion. Its separate output and exact hashes are retained as
  ownership-fenced evidence. Aborted first-command initialization and
  submitted GPU failure poisoning have not yet been fault-injected.
- The merged native window and guided training have not yet been exercised
  end to end. Real simulation pause, scene and policy switching, artifact
  restoration, exact experiment replay, and native/CLI/agent consistency
  remain release gates. Wider historical pack coverage is needed and
  ambiguous provenance must never be guessed.
- The benchmark-accounting branch is separate. Its local or published evidence
  is bound to its own revisions; it has not been integrated here.

Both directional CCD and the experimental augmented-Lagrangian contact law
remain disabled by default. This checkpoint does not change their promotion
status.

## Synthetic custom-scene and bootstrap check

The clean checkout does not contain the catalog compiler's G1 URDF, Franka
meshes, or PX4 presentation mesh tree. The built-in catalog therefore reports
each missing path before trying to build; an explicitly authored scene bypasses
those optional inputs. The CPU bootstrap check uses stubbed build and rollout
commands and only tests this shell contract. The PX4 mechanics source has an
upstream revision and required-file hashes, but its presentation meshes are
not covered by those hashes. No local matching G1 or Franka source was found.

A bounded native M6 window check used a separately cooked, synthetic
presentation-only tetrahedron attached to the G1 pelvis, sourced from the
tracked probe mesh. It is not Unitree G1 presentation geometry. The native
window initialized one G1 ground/velocity environment, exercised rejection of
a nonexistent scene and restoration of the previous configuration, then
closed successfully. Output, generator, fixture, and hashes are retained in
the evidence directory. This is not saved simulation-state restoration,
actual G1 visual qualification, training, or physical validation.

An earlier attempt timed out after 45 seconds during the CPU-only Release
rebuild, before native simulation. That full build log is retained separately.

## Training and policy wire compatibility

The first bounded trainer attempt failed: its native initial PolicyPack was
version 5, while the Python MLX reader accepted only through version 4.
The previous Python ctypes policy structure also omitted the native C ABI's
task-binding fields. The exact failure is retained. The matching numisolver
v5 reader, worker and C ABI bridge have been integrated; CPU tests exercise
actual native version-4 and version-5 producer files, Python-to-C round trips
for single and multiple tasks, and malformed/truncated bindings. The test is
available as metalrobo_policy_wire_compatibility when
METALROBO_MLX_TEST_EXECUTABLE selects a Python with MLX and NumPy.

A second M6 attempt completed eight samples from one G1 ground/velocity
environment and one PPO update without failed environment steps. It produced
a rollout, candidate, and deployment-format candidate; no deployment
selection was performed. The exact trainer binary hash was not captured before
a later rebuild, so this result does not establish an exact-binary identity.
The captured initial and candidate hashes and full output remain evidence.

The native window discovered and loaded that unselected candidate from the
policy catalog on the synthetic scene, rejected an invalid scene and restored
the previous configuration. The output and scene/policy hashes are retained.
This is not a successful guided UI training session, evidence of learned
control, saved-state replay, or a performance comparison.

Guided training now forwards the window build override, locates a workspace
dispatcher where available, and separates the window command's logs from the
trainer's own stdout/stderr files. The CPU environment checks pass; actual
guided training remains an end-to-end release gate.

## Snapshot/restore boundary

MetalWorld exposes resident initialization, continuation and inspection-only
digest, but no general resident-state export/import transaction. Measured
surface owns additional accepted state and evidence outside that resident.
Restoring only either half would not reproduce the coupled continuation.

A narrow implementation should reserve the idle resident and external owner
together; validate pool, world/task and extension fingerprints, environment
count, arena generation, reset semantics and all persistent logical buffer
lengths; capture the world buffers and extension accepted state in the same
reserved transaction; and publish the snapshot only after all readbacks
succeed. Restore should validate the entire immutable identity and content
before staging all replacement buffers, then atomically swap/publish both
owners after successful GPU completion. Failure must leave the prior accepted
continuation intact. The existing resident digest must retain its present
identity semantics; a portable archive, if later required, needs a separate
explicit versioned fingerprint rather than weakening that digest.

No such cross-owner snapshot/restore exists in this checkpoint. Combined Matter
and measured surface remains explicitly unsupported; no exact replay claim is
made. Fault-injected partial failures, identity mismatches, and post-restore
continuation comparisons are required before exposing this as supported.

## Guided command cancellation

The training command now forwards interruption to its active trainer or
selector process tree and removes its run lock. CPU-only stub-process checks
pass through the actual dispatcher and cover both phases, descendant
termination and lock cleanup. The actual native
guided UI cancellation path and a complete guided session have not been
qualified.

## Pinned command training and selection

At clean commit 3f9188d4474663c28e805949a8db45f72cdf5f73, the actual
numi train dispatcher completed a one-environment, eight-step, one-update G1
ground/velocity workflow, then evaluated incumbent and candidate with one
environment and the configured 1,000-step horizon. The selector retained the
incumbent and did not advance the candidate. Both evaluations reported zero
clean-horizon rate and 12 physical terminations; this is an operational
pipeline check, not successful locomotion or a qualified controller.

The complete run directory, selection traces, policy and learner artifacts,
logs, source/runtime hashes and pre/post identity check are preserved in
command-train. The binary and source hashes matched before and after the run.
This command-line exercise does not establish that the window's guided
training panel, live cancellation, or saved-state restore works end to end.

The opt-in native guided-cancellation probe injects a training request through
the same scheduler mailbox as the window, waits for the real trainer and MLX
worker, and closes the AppKit window only after both are observed. It checks
the child exit, lock removal, observed descendant cleanup and source/binary
identity. The request is test-injected rather than a click through the setup
panel. The first attempt did not reach a live trainer, as described below.

The first guided-cancellation probe did not reach a live trainer: its harness
resolved the virtualenv symlink to a system Python without MLX. The window
reported the missing interpreter and removed its lock; this does not test
cancellation. That failure is retained under guided-cancel-001. The harness
now preserves the virtualenv path and suppresses generic success after a
training-start failure.

The revised M6 probe passed at commit 45b6af1fe49946c4f63edfa8ddfca6ab4854097c.
The test-injected request reached the guided training scheduler, and its
monitor observed the native trainer and MLX worker before requesting an
AppKit window close. The command exited 143, its lock was removed, no
observed descendants survived, and source/binary hashes matched before and
after. The trainer and worker were observed during startup; no completed
update is claimed. Both complete attempts and the canceled run's artifacts
are retained. This is not a click through the setup panel.

At inspection the host's Aqua console session was locked. An accessibility
query did not complete before it was stopped, so the actual panel interaction
and live cancellation by a human operator remain unqualified. Full
MetalWorld/measured-surface snapshot restoration is still unsupported.

### Manual setup-panel qualification procedure

When the console is unlocked and the operator can interact with the desktop,
run the following from the clean integration worktree. The catalog and its
synthetic visual pack are retained under the repository evidence directory.
The environment must contain the MLX-enabled interpreter and built binaries.
This mode does not inject a training request or a close event.

```sh
python3 tests/numi_guided_cancel_native_probe.py --manual-panel \
  --workspace "$PWD" \
  --build /Users/numi/numi-lab-release-integration-build-20261107 \
  --scene-catalog "$PWD/docs/evidence/window-integration-20261107/training-catalog" \
  --python /Users/numi/numi-window-integration-venv-20261107/bin/python \
  --output /Users/numi/numi-guided-panel-manual-evidence-20261107
```

Press T in the window to open Configure Training. Enable Customize, set
parallel environments to 1, steps per update to 8, learning updates to 1000,
GPU submission chunk to 1, and checkpoint interval to 1000. Start training.
After the monitor reports both the native trainer and MLX worker, close the
window. The monitor requires the canceled command status, removed lock, no
surviving observed descendants and unchanged hashes. It stops the process
group after 180 seconds. Review its result and retained logs; this procedure
has not yet been run on the locked host.

The ownership probe now also checks that unsupported digest, changed owner,
changed fingerprint and mixed-owner requests leave accepted surface bytes
unchanged, then compares the challenged resident's next continuation with an
independent unchallenged resident. It checks the pending-inspection gate too.
On Apple M6, the expanded probe passed at revision `0d1064c0` with one
environment and two steps; timestamped output and binary, source and input
hashes are retained in `physics-digest-fixed`. This is a fail-closed
continuation test; it does not create or restore a snapshot.

An additional pre-fix diagnostic uses a synthetic device-physics owner with
external host state. At revision `840a0898`, changing only that state from 11
to 29 returned success and the same resident digest both times, demonstrating
an incomplete digest. The retained baseline includes binary, source and model
hashes and both wrapper errors: the first shell failed to record its exit code
because `status` is read-only in zsh; a second attempt used a wrong binary path
and exited 127 without starting the probe. This is a synthetic ownership test,
not a Matter simulation or physical test.

Every valid device-physics program has a nonzero fingerprint but exposes no
callback to capture its private state. The resident digest now rejects every
such program, as it already rejected external device-mechanics owners; it does
not infer completeness from an owner fingerprint. Device observation and action
callbacks also have external contexts and no state-digest callback; their
continuation ownership is not qualified by this change. Atomic cross-owner
snapshot and restoration remain unsupported. The post-fix synthetic-owner
probe passed at revision `0d1064c0`: both requests returned `invalid_model` and
zero. The first post-fix attempt passed a directory instead of its manifest and
failed before simulation; its outputs remain in `physics-digest-wrong-path`.
Neither test establishes Matter physics or complete external-owner replay.

The callback boundary audit covers device physics, device mechanics, device
observation, device action and inspection. Physics and mechanics are rejected
by the digest. Observation and action have opaque contexts and can affect the
next policy or physical action; the resident currently records neither owner,
so their complete continuation state cannot be claimed by a successful digest.
Inspection is declared presentation-only. Multicopter and Millard programs are
source-authored programs, not opaque owner callbacks; their broader replay has
not been qualified by these checks. A complete-state contract must also resolve
the observation/action ownership gap before integration.
