# Empty plastic bag: two Franka handoff

One eight-second scene, 1,921 native Metal states at 240 Hz, zero coins. The first Franka picks up the dropped pouch by an opening corner. The second approaches from above with its fingertips downward and closes on the opposite opening corner, then the first opens and retreats. The second holds the pouch while the first retreats. The final normal-speed video shows both the complete scene and a closer contact view. Blender renders saved states and supplies no dynamics.

## Physical construction

The approved empty pouch remains 200 × 230 mm, 75 µm LDPE, density 923 kg/m³, Young's modulus 100 MPa, Poisson ratio 0.45, mass approximately 6.52 g. Folded bottom, welded classical side seams, open mouth. There are 1,081 dynamic nodes, 2,112 triangles and zero fixed nodes. Film/table/self contact and membrane/hinge constitutive laws and acceptance tolerances are preserved.

The native scene adds a second independent finite compliant fingertip contact state. All film nodes are eligible for contact; there are no selected-node attachments or pins. Pad spring stiffness is 200 kN/m per node, friction coefficient 0.8, normal force cap 20 N per node. The donor retains its approved grip. The receiver closes to a 71 µm total gap, approximately 0.4 N per jaw. This gentle pinch provides approximately 0.64 N friction capacity for a 0.064 N bag weight. Stiffness and force cap are unchanged.

The donor has the approved authored 17 × 17 × 3 mm rubber pads and central rigid backing. The receiver uses a visible 12 mm forward pad extension: pad centre local z = 56.5 mm rather than 44.5 mm. Its central carrier is 4 × 8 × 4 mm, recessed 10 mm at the leading tip from the original 14 mm carrier. The pad and support are authored fixtures, not stock Panda fingertips. The original solid white finger and arm housings remain in all saved-state mesh checks.

`bind_handoff_fixture.py` recomputes native Franka FK for the extended fixture while retaining every world pad input byte. The maximum native FK contact-centre discrepancy is below 5 nm. Articulated geometry metadata is rebound after physical integration; the source scene consumes the separate world pad trajectory for film forces and work. Original raw poses and failed smaller-fixture evidence remain preserved. This is an equivalent prescribed-kinematics scene, not a rerun of articulated motor dynamics.

## Solver change and lineage

Only the gripper material-point velocity/work calculation changes: it includes the rigid pad's rotational displacement as well as translation. The corresponding terms are used consistently in relative friction velocity and actuator work. Film/table/self and coin kernels are unchanged. Both arms contribute spring energy and signed/positive actuator work to the unchanged energy acceptance rule; rollback retains both arms' contact state.

- States 0–1,200: exact accepted approved empty-bag drop/pickup history. This prefix used the older translation-only gripper work model.
- States 1,201–1,560: native corrected rotating-pad continuation, receiving fingertip interception from above and confirmed two-hand contact. The corner lies at local x=z=7.5 mm in the finite receiver pad plane.
- States 1,561–1,680: native direct handoff and first-hand release.
- States 1,681–1,920: exact continuation of the 7.00 s checkpoint, donor retreat and receiver hold.

Rigid FK metadata is rebound to the authored recessed receiver carrier. No world pad path byte, film state, force or work value changes in this fixture variant. Original raw geometry metadata and rejected long-carrier audits remain preserved.

Checkpoint joins share their physical state exactly. No film interpolation or Blender physics is used. Rejected opening-spread runs and earlier geometric approaches are retained as failures in the release archive. Their housing crossings are not successful handoffs.

## Replay

The archive contains source, Metal library, native executable, exact input paths, native FK source/headers, asset licenses, checkpoint segments and per-file hashes. The executable embeds the Metal library path `/Users/n/numi-plastic-bag-franka-handoff-20261024/build/PlasticBag.metallib`. Restore that path for the original executable, or adjust `build_only.sh` and rebuild in the new directory.

Place the exact solver input files in the working directory, restore `evidence/finish_prefix/checkpoint_1680.bin` under `finish_prefix/`, and run:

```sh
./build/numi_classic_bag_handoff --empty-bag --second-gripper --output finish --resume finish_prefix/checkpoint_1680.bin --steps 1920 --checkpoint-every 120
```

This resumes the exact accepted 7.00 s state, retaining all dynamic film and both contact histories. The original prefix and capture are included. To regenerate the capture segment, restore `evidence/checkpoint_1200_parent.bin` as `checkpoint_1200_parent.bin` and run the same executable with `--output capture --resume checkpoint_1200_parent.bin --steps 1560 --checkpoint-every 120`. The approved initial five seconds are inherited exactly; they are not a new from-start measurement. Joining their native saved states at the shared checkpoint reconstructs the complete delivered eight seconds. Full from-start reintegration is a separate reproducibility experiment.

## Measurements and limits

See `evidence/assessment.json` for actual transfer timing, measured support, energy balance, floor gap and source identities. `grip_alignment.json` checks forces against the visible native FK pad faces and finite footprints. Film/housing and inter-arm mesh audits inspect every saved state, including the rigid pad carriers; they exclude the compliant contact patches.

Robot joints are prescribed native kinematics, not coupled motor dynamics. Housing clearance is a saved-state geometric audit, not continuous whole-robot collision detection. Finite pad patch entry does not include a smooth tangential elastic edge potential. A rejected high-preload hold exposed this discontinuity; reduced pinch preload is a scene choice, not a solver repair. The fingertip law models finite inward normal/friction patches, not volumetric rubber or complete back/side surface contact. Material constants are uncalibrated and air drag remains absent. 240 Hz is simulation sampling, not real-time throughput. Legacy native `release_event` and `motion_phase` strings predate this scene and are not measurements; `handling_plan.json` and measured contact records define this handoff.
