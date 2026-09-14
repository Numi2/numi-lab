# Numi Human static reaction handoff

The native large-state stand owner now accepts an explicit environment-major `generalizedForcePreload` stream. The source static solve supplies support, scalar equality, active position-limit, and registered passive-coordinate reactions through that stream. The native owner adds the stream to the MyoSim generalized force before its mass solve; gravity, activation dynamics, contact projection, and joint equality projection remain owned by the native step. Root assistance is not used.

The persistent probe also carries only the six witnesses with positive static normal reaction into the dynamic contact set. This keeps the unilateral problem identical at release instead of admitting all ten geometrically near-plane witnesses.

On the M4 Pro Mac mini with the current one-adult payload, the 12.5 microsecond one-step release changed from approximately 8.67e3 m/s2 pre-projection acceleration and 1.56e-2 m/s velocity increment to 3.01 m/s2 and 7.86e-5 m/s. The 100, 50, 25, and 12.5 microsecond results converge to the same approximately 3 m/s2 release acceleration. A 64-step, 0.8 ms horizon stays finite with 5.76 m/s2 maximum pre-projection acceleration and 5.91e-3 m/s maximum final velocity delta.

The source static certificate still reports `balanced=false` with normalized residual RMS 0.8608. The 64-step horizon is therefore a bounded release and handoff receipt, not a 10–60 second assistance-free standing qualification. Perturbation recovery, walking, subject calibration, deformable materials, and blood-to-tissue closure remain open.

Evidence: `media/numi-human-static-preload-v1/qualification.json`.
