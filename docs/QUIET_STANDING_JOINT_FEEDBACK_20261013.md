# Native quiet standing with prepared joint feedback

The standing launcher previously prepared a tonic-only v2 Brain source unless an
external program requested joint calibration. Tonic excitation completed 10,000
accepted physical roots but lost all support after 3.081 s and reached 224.33 m/s.
An explicitly prepared v3 source also could not bootstrap its existing frozen v4
controller without an external program containing that source's hash.

Brain main now selects its existing v4 program for an explicitly calibrated v3
source and exports the actual admitted immutable program for replay. The launcher
pins Brain `0f7f1e4dcb29c885eddb8ca2a673f26dba537004`, selects joint calibration by
default when no external program supplies its own mode, and preserves the native
prepared source and program in its content-addressed artifact store. Explicit
`NUMI_HUMAN_BRAIN_JOINT_PATH_CALIBRATION=0` retains the tonic ablation; existing
external v1-v3 programs retain their inferred source mode.

The frozen joint law is unchanged: length gain 10, velocity gain 1 s, maximum
correction 0.2. The Human source remains
`f9d50d0ecaa3394f19f056aa0f28eb52a4e2770e`; native physics uses 1 ms roots, 64 ordered
contact sweeps, all 416 muscles and zero root assistance. No pose lock or acceptance
tolerance was relaxed.

## Observed engineering gate

The M4 Pro native episode completed 10,000 physical roots with 10,000 valid sensory
and motor records. Minimum support contacts were six, peak penetration 0.694 um,
peak root speed 0.002600 m/s, maximum COM drift 0.206 mm, final COM drift 0.121 mm,
and maximum root rotation 0.000202 radians. The verifier requires complete ordered
roots and motor audits, consistent native transaction fingerprints, all ten valid
touch receptors, the native sensory clocks, 64 sweeps and zero assistance. It checks
contact count, penetration, speed, COM excursion/height and rotation over the whole
episode rather than only the final state.

The first ten-second run used Brain `3bb4190ac5671e05201d9ccb7cdbf38367913412`.
A subsequent native default smoke run with the export-only Brain change produced a
byte-identical prepared source and the same admitted program fingerprint
`1942320659943724188`. Every motor receipt in the original episode has that
fingerprint. A separate recovered receipt adds this actual native exported program
without altering the original run or manifest; its reconstruction record names both
revisions and exact source/program/manifest hashes. This is receipt recovery, not a
second ten-second replay or a performance measurement.

The retained tonic episode completes its numerical transaction but fails all six
physical quiet-standing bounds. Its native v1 program was reconstructed using the
same exact-source/fingerprint procedure and remains explicitly failed evidence.
Four-step runs are rejected as evidence for the ten-second gate.

```sh
python3 tools/verify_human_brain_quiet_standing.py --run NEW_NATIVE_RUN
```

This is a simulated engineering stance on the prepared MyoSim source. It does not
qualify physiology, a learned balance skill, perturbation recovery, compliant whole
body anatomy, or five-second wall-clock execution. The separate 1.5 s spindle
exploration is retained as early control evidence and is not substituted for this
gate. Performance variants retain their original source-bound acceptance criteria.

## Published-default native repetition

The published default launcher (`187f5b60`, Brain `0f7f1e4`, Human source
`f9d50d0e`) also completed all 10,000 roots on an Apple M6 using the retained
M4-built native binary/resource packet. This is native M6 Metal execution, not
a claim that the M6 Command Line Tools compiled the packet. The actual admitted
program and prepared source are exported and hashed by this run; reconstruction
is unnecessary. The complete run passes the same unassisted engineering gate.

The ordered printed physical progress rows match the retained M4 run. The full
physical and joint state fingerprints diverge from root 5,889, and motor audit
activations differ by up to `1.17357976e-38`. Sensor values otherwise match at
printed precision. Complete cross-device exact replay is therefore **not
qualified**. No values were rounded or normalized to suppress this difference.

The explicit `cpu-one-handoff` backend also completes the same ten-second gate:
minimum six contacts, maximum COM excursion `0.00019748282323674482 m`,
maximum root speed `0.00246495974435 m/s`, maximum penetration
`5.89759338254e-7 m`, and maximum root rotation `0.00020798174131248496 rad`.
Its native physical/sensory/motor trace differs from all-GPU starting at root one.
This independent engineering pass does not close exact GPU/CPU replay or the
original five-second end-to-end performance goal. The default remains all-GPU.

The corresponding machine-readable native gate receipts and ordered trace
comparison are retained beside this document. The complete logs, manifests,
program/source exports and native artifact packet are preserved in the
reconciliation recovery archive; the receipts bind their SHA-256 identities.
