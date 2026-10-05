# First integrated complete breath, 2026-10-05

The native whole-body scene accepted 6,000 steps of 1 ms on the SSH Mac mini
(Apple M4 Pro). It completed one breath and seven cardiac filling/ejection
cycles, with zero root assistance. This is a six-second implementation check,
including initialization, **not** the required five-minute qualification.

The existing articulated dynamics, full-weight skin contact support, MyoSim
respiratory mechanics, Matter circulation and NumiBrain regulation ran in the
same accepted-state transaction. The retained physiological CSV comes from that
state. The companion surface CSV checks each presented frame: no nonfinite skin
vertices or vertices more than 1 mm below the bed, and maximum functional
surface-volume relative error 1.118e-6. These checks do not establish anatomical
embeddedness or resolve intended tissue interfaces.

Measured body-horizon wall time was 208.054721 s for 6.000000285 simulated s:
0.0288386 times real time. The completed tidal volume was 552.823 mL and the last
LV stroke volume 70.530 mL. The maximum blood-volume accounting error was
0.002794 mL out of 5,150 mL. See `summary.json` for ranges and trace hashes.

`native-viewer.mov` is a continuous recording of the native renderer's Metal
framebuffer. The Mini desktop was locked, so the MTK window had no drawable;
this is not evidence of an unlocked interactive desktop. Its 189 image frames
span 208.8167 s of wall time, preserving the slow execution. The inspection log
distinguishes five zero-sample AV timing/edit markers from image frames, checks
all actual presentation timestamps, and extracts the exact final frame. The
layer-tour source edit was not in this binary; the movie displays the skin layer.

The exact launch, device and asset/binary/library identities are in
`invocation.json` and `run-metadata.json`; all checked files stayed unchanged
during the run. Full native log and intermediate source snapshots remain at
`/Users/n/numi-human-resting-evidence-20261005/native-breath-001` and the sibling
`native-breath-001-source` on the Mini. This pre-commit diagnostic is bound to
those binary hashes, not retrospectively claimed to match a later source commit.

Known failures under repair: source lung-lobe defects, source/cycle cardiac
intersection defects, and whole-skin self-intersection qualification. The
upstream CVSim21 LV volumes also give a low ejection fraction for the intended
reference adult. This run cannot support a claim of an anatomically coherent,
physiologically qualified resting human. Earlier failed runs remain retained.
