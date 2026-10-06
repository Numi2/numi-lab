# Native ventricular material binding and source reconciliation

Five scoped runtime/compiler files extend the existing anatomical heart owner.
The loader binds the repaired ID23 material boundary to its exact source,
RV/LV lumen ownership and bounded inferred displacement map. GPU code derives
the wall deformation and material-volume closure from the accepted chamber
state, rebuilds normals and checks every presented functional surface for
degenerate triangles and volume disagreement. It adds no pump state, force
owner, controller history or per-step CPU mechanics.

The SSH Mac mini built published Lab revision 6c54145 plus the exact five-file
patch retained here. Brain revision a1cf721 was clean. Native run123 completed
3,000 accepted 2 ms steps using the unchanged run114 input scene. Both full
CSV traces and all four accepted geometry packs (steps 0, 639, 2783, 2999) are
byte-identical to frozen build026/run114. Terminal q and v also match exactly.
The rejection/replay probe passes for body, MyoSim, circulation, clocks,
respiration and Brain history, including rejection after an accepted prefix
within the same command buffer. This is recorded in `native.log`.

Native wall time was 88.7468 seconds for six simulated seconds (0.067608 times
real time); the wrapper took 101.1737 seconds including setup and dynamic-loader
diagnostics. The loaded `libmetalrobo.dylib` path matches the invocation's
hashed runtime dependency. All runtime libraries and shaders are hashed before
and after execution. Full movies/packs remain at manifest-bound Mini paths.

This verifies compilation, accepted-state transaction behavior and equivalence
to the existing integrated trajectory. It does not qualify all cardiac tissue
interfaces: atrial/ventricular and wall/cavity neighbor checks remain unresolved,
as do respiratory and other passive-anatomy defects. The scalar volume check
does not prove tissue validity, mechanical stress accuracy or clinical realism.
The later geometry corrections must be checked in the same native scene before
the final five-minute paired demonstration.
