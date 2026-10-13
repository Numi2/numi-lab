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
