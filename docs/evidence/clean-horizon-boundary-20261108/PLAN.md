# Native horizon-boundary diagnosis

Use the retained synthetic Franka WorldPack and pinned incumbent policy from
the merged training smoke. Run the same native rollout, seed and geometry at
255, 256 and 257 control steps, chunk 1. The authored episode horizon is 256.
Predict no termination at 255, a timeout at 256 and a timeout followed by a
new episode at 257. A clean-horizon measure intended to mean no termination
before the requested boundary should be 1, 1 and 0 respectively. The current
implementation counts every timeout, so it is expected to give 1, 0 and 0.
Do not infer trained skill, contact accuracy or quality from this fixture.
Retain exact inputs, outputs, runtime identities and all failures. The
contact-reference CPU run overlaps; no performance claims are made.
