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
mechanics because it cannot digest those external buffers. There is no
external surface save/restore implementation in this checkpoint.

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
