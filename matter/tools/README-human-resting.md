# Coupled resting physiology instrument

`numi-human-resting` runs the existing Matter CVSim21 circulation, existing
MyoSim compliant muscle/tendon laws, two respiratory mechanical coordinates,
conservative O2/CO2 reservoirs, tissue metabolism, and NumiBrain's respiratory
chemoreflex on one Metal command-buffer timeline. Its CSV contains calculated
accepted state. This instrument does not yet qualify a supported anatomical
whole human or provide the final native scene.

Build explicitly with a NumiBrain checkout containing the respiratory ABI:

```sh
cmake -S matter -B build/resting -DCMAKE_BUILD_TYPE=Release \
  -DNUMI_BRAIN_ROOT=/absolute/path/to/numi-brain
cmake --build build/resting --target numi-human-resting -j 4
build/resting/numi-human-resting \
  matter/tools/fixtures/cvsim21.native.v3.json \
  matter/examples/resting-reference-respiration.json /tmp/resting.csv \
  --steps 6000 --dt 0.001
```

`--verify-transaction` intentionally rejects a physical step and verifies that
circulation, its clock, respiration, and controller history remain bitwise
unchanged. It restores the preceding accepted snapshot and verifies bitwise
replay. This checks the live transaction; it does not claim portable archive
identity. `--drive-intervention START_SECONDS END_SECONDS SCALE` scales only
the delivered respiratory excitation during the half-open accepted-time window.
`--instrument-excitation DIAPHRAGM INTERCOSTAL` disables regulation for actuator
calibration and is explicitly reported in the output.

The reference JSON declares units, parameter sources, and inferred reductions.
Respiratory compliance acts on the sum of diaphragm and rib swept volumes;
pressure differences across airway resistance produce flow. Expiration follows
muscle deactivation and recoil. Gas transport uses actual vascular flow and
lung volume, and all transfers share a mass ledger. CO2 content uses a local
fixed-pH approximation; nitrogen, acid-base effects, Haldane effects, diffusion
heterogeneity, and anatomical deformation are outside this instrument's scope.
The two muscle aggregates still require registered anatomical binding in the
whole-body scene. The circulation remains bound to its source CVSim identities.

The first retained six-second smoke run and rejection/replay checks are under
`evidence/human-resting-20261005`. These establish numerical integration only,
not five-minute stability, physiological validation, or anatomical pumping.

### Retained-pose skin replay

The viewer captures its uploaded skin map, complete positive influence interval,
anatomy parameters and pinned native kernel owner once when geometry capture is
requested. These are existing ABI records; no simulation state is added.

Build the `numi_human_resting_skin_replay` target, then run:

```sh
numi_human_resting_skin_replay SOURCE_RECEIPT SOURCE_SHA256 POSE_RECEIPT POSE_SHA256 NEW_OUTPUT_DIRECTORY
```

The tool dispatches the existing Metal skin kernel at the retained accepted
body poses and respiratory swept volumes. Its output is counterfactual geometry
evidence, not a new accepted physical state. Establish bitwise position parity
against the source scene's native capture before using other replayed poses for
anatomy checks. Old receipts without captured influences and kernel identity
are rejected; do not fabricate these fields.

The focused GPU regression uses an explicitly selected binary and existing
HumanRespiration library:

```sh
NUMI_RESTING_SKIN_REPLAY_BIN=/absolute/path/to/numi_human_resting_skin_replay \
NUMI_RESTING_SKIN_REPLAY_METALLIB=/absolute/path/to/HumanRespiration.metallib \
python3 matter/tools/test_resting_skin_replay.py -v
```
