# Native contact-VF run 029

Execution exited 1 after 467.835851167 s. The invocation requested 32 physical steps of 125 microseconds (4.0 ms), with FGMRES budget 64. It accepted 22 coupled rows through 22 steps (2.750 ms), then failed after two microsteps in the deformable FEM quality gate.

The failure is matter code 5, object 15, global tetrahedron 18652, local tetrahedron 14905, with rejected Jacobian J=0.313928186893 below the configured floor 0.35. The accepted rolled-back-state witness for that tetrahedron has J=1.001905063546; this is not the rejected candidate.

The root-5 VF contact rejection and the root-20 vascular rejection seen in earlier runs did not recur: this run advanced to root 22, and the failure owner is the whole-body skin-fat bonded shell. The failure report identifies passive_tissue_runtime as the owner. The prior failures are not upgraded to proof of broad contact qualification.

Across the 22 accepted CSV rows, maximum recorded normal impulse is 0 Ns, maximum root assistance is 0 N and 0 Nm, minimum reported contact gap is 1.52555904e-05 m, and maximum peak penetration is 0 m. No nonzero bed-contact impulse or root assistance is recorded in these rows; this run does not demonstrate sustained bed support. The requested 32-step/4-ms horizon did not complete.

The run admitted 28 registered ligament force rows and 26 fascia regions with zero legacy reduced-fascia ligament rows. The exact source revision, executable, Metal library, 13 argv asset paths and their retained SHA-256/size manifest are recorded beside the full CSV and logs. All 13 argv asset files were rehashed after the run and match that manifest. Large render packs and geometry captures are excluded. This is mechanics-only evidence, not a solved static equilibrium, physiological/clinical result, sustained-support result, or real-time performance claim.

Mass calibration: resting_mass_calibration source_composite_kg=97.1319506911 declared_reference_kg=97.1319506911 executed_kg=97.1319506911 uniform_mass_inertia_scale=1 measured_person=false additional_organ_or_blood_mass_kg=0
