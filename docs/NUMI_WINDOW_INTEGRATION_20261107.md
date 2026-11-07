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
round trips, atomic ambiguity rejection, and an unambiguous tag. It uses
synthetic fixtures; historical producer-built artifacts still need testing.

The inspected source branches serialize task-pack versions 17 and 19 using
the same field sequence. This checkpoint writes version 20 and reads those
two versions explicitly. The CPU test exercises that decoder using synthetic
current-format payloads with historical version fields; it does not establish
compatibility with actual retained historical packs. Policy format 4 decoding
is retained alongside version 5.

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

- The CPU catalog metadata check passed for two policies and two scenes.
- The CPU migration test passed synthetic version and provenance cases.
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
- Aborted first-command initialization and submitted GPU failure poisoning
  are implementation safeguards that have not yet been fault-injected.
- The merged native window and guided training have not yet been exercised
  end to end. Real simulation pause, scene and policy switching, artifact
  restoration, exact experiment replay, and native/CLI/agent consistency
  remain release gates. Historical task/actuator packs need branch-produced
  fixtures and ambiguous provenance must never be guessed.
- The benchmark-accounting branch is separate. Its local or published evidence
  is bound to its own revisions; it has not been integrated here.

Both directional CCD and the experimental augmented-Lagrangian contact law
remain disabled by default. This checkpoint does not change their promotion
status.
