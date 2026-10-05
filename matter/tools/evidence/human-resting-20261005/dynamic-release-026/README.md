# Explicit resting release initialization

The supine source pose initially has only one active bed witness; 31 other
witnesses are separated and receive zero compiled support force. Reusing the
standing recruitment cap produced a numerically stable 320-second run but
crossed the legs (1,102 exact inter-leg skin triangle pairs at the terminal
state; see `../costal-stability-018/terminal-posture`). That result remains a
failed anatomical acceptance case.

The optional `--resting-release-initialization` permits a non-equilibrated
initial pose to settle through the existing Metal contact and MyoSim/tendon
owners. It relaxes only the startup acceleration-residual admission threshold;
solver success, non-worsening recruitment residual, valid source pose,
geometric support admission, and every physical-step guard remain required.
It requires the resting owner, rejects standing recruitment caches, enters the
source fingerprint, and explicitly reports `balanced=false` when applicable.
It supplies no root force or pose-holding force. Existing default behavior is
unchanged.

The existing recruitment cap can now be selected through the Human launcher.
The trial value 0.01 is a numerical initialization choice, not a measured adult
muscle excitation. Short sensitivity runs also used 0.005 and 0.02. All builds,
tests, simulation, and exact geometry computation ran on the SSH Mac mini
(Apple M4 Pro); the full source, executable, asset, configuration, and command
identities are retained in each invocation.

## Observed results

| Run | Scope | Result |
| --- | --- | --- |
| 020 | Zero cap, existing startup | Rejected before physical steps: equilibrium configuration requires a positive cap. |
| 021 | Cap 0.01, existing startup | Rejected before physical steps by the standing acceleration-residual bound. |
| 022 | Explicit release, cap 0.01, 6 seconds | Completed; no assistance or numerical guard failure. |
| 023 | 60-second capture request | Rejected before physical steps because requested frames missed the accepted presentation cadence. |
| 024 | Explicit release, cap 0.01, 60 seconds | Completed in 590.326 native wall seconds, RTF 0.101639. Exact selected inter-leg skin checks at steps 3199, 4991, 14975, 29999 found zero pairs. |
| 025 | Release plus existing upper passive joint rows, 6 seconds | Completed; an optional reference-model experiment, not resolved hand physiology. |
| 026 | New binary with default startup, 6 seconds | Coupled CSV, surface CSV, initial pack, accepted body/respiration hashes and rendered vertex buffers match source013 exactly. |
| 027 / 028 | Release, cap 0.005 / 0.02, 6 seconds each | Completed; selected inter-leg skin checks at 639, 2783, 2999 found zero pairs in both runs. |

The default-regression accepted packs differ only in their metadata section;
the new default-false option changes the source/root identity included in pack
provenance. All other sections match. See `default-comparison.json` for the
whole-file differences and the separate section/accepted-state comparisons.

Every successful run retained matched body/physiology clocks and passed the
native rejection/retry probe: rejected work preserved body, MyoSim, circulation,
respiration, and Brain history, and accepted retry matched uninterrupted replay.
The 60-second run's root assistance columns remain zero. All its surface status
guards are zero; maximum functional volume relative error is recorded in
`summary.json`. Full CSVs are losslessly compressed here, not decimated.

## Acceptance boundary

These are startup and bounded posture checks, not a five-minute posture proof.
The 60-second COM still creeps laterally, and the exact leg check excludes the
adjacent groin and does not establish whole-skin clearance. Read-only native
terminal-pose images expose distorted hand poses in both 024 and 025. The
existing wrist/finger model omits thumb passive mechanics and its measured
aggregate stiffness may overlap with MyoSim passive forces; it remains optional.
The source provenance corrections and this limitation are documented in
`docs/NUMI_HUMAN_UPPER_PASSIVE_JOINT_TISSUE.md`.

Known rib/lung, cardiac-wall, and visceral-interface defects remain. No clinical,
whole-body anatomical, steady resting-posture, or real-time claim follows from
these runs. The subsequent 320-second experiment is separate evidence.

The full continuous native movies, MRVPACK captures, and receipts remain at
the exact Mini paths and SHA-256 identities in `summary.json`. The captured
terminal-pose PNGs use the native Metal forward-kinematic inspection path with
zero physical steps and are not a physiological replay.
