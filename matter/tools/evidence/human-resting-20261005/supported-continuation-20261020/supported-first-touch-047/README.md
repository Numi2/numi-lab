# Supported first touch and CC integration047

This increment builds the existing integrated human on SSH Numi (Apple M6) with 30 massless registered-ligament rows, including two inferred effective coracoclavicular resultants. The separate source-geometry/CPU evidence is in ../coracoclavicular-046. The accepted back/skin coordinates, tissue mass, force owners, physical gates and scalar FEM solver are unchanged. No visual verification was performed.

The control and candidate differ only by a fresh-scene Float32 root Z translation of -15.2587890625 micrometers, derived from the previously measured minimum support gap. All body, skin/fat, vessel, organ and fascia references are rebuilt from that common pose. The current-q GPU callback supplies the actual skin support witnesses; there is no in-place pose correction after tissue construction.

| Measurement | Control | Candidate |
|---|---:|---:|
| Initial minimum bed gap | +15.2555905 micrometers | -0.00319871 micrometers |
| Closed support rows, 1 micrometer tolerance | 0 | 1 |
| Recruitment support force | 0 N | 404.754 N |
| Accepted root steps | 2 | 2 |
| Accepted simulated duration | 0.25 ms | 0.25 ms |
| Maximum penetration in accepted trace | 0 m | 2.6730721e-8 m |
| Root assistance | 0 | 0 |
| Body mass accounting residual | 0 kg | 0 kg |

The candidate has nonzero contact impulses (0.0108196763 and 0.0108643388 N s); every separated initial support row retains zero support force. The retained tissue audits pass, including all26 fascia regions and30 registered rows. Minimum/maximum FEM J are0.717178/1.105527, with maximum displacement0.579331 mm. The candidate body-contact diagnostics count303 closed internal samples; these are not bed support rows and do not establish that all anatomical interfaces are covered.

The late-reject/retry run passed the existing body/tissue/vascular/Brain/respiration rollback test. Its accepted coupled trace equals the ordinary candidate trace exactly. Each trial completed two125 microsecond roots. Runtime physical-segment real-time factors remain about1.3e-5 for ordinary trials; this is not a real-time or long-run qualification.

The consolidated preparation verification passes the skin/fat and mixed-donor transfer fixtures, closed boundary and exact Float32 geometry audits, catalog fixture, body/FEM interface, support recruitment/callback capture, contact geometry/sort, FEM backtracking/reference/release/snapshot and vascular principal-block/predictor controls. See verification/ and verification.log. No extra physical readback or per-step CPU implementation was added.

One closed row has zero world-XY span and cannot establish a support polygon. Recruitment explicitly reports unbalanced initialization (maximum floating-root acceleration residual162.505). This is first contact, not settled equilibrium, whole-body contact completion, all-ligament coverage, sustained stability, or physiological/clinical validation. Remaining ligament representations include spine, wrist/hand and several elbow restraints; the effective shoulder anchors are declared anatomical inferences.

Reproduce each native trial using its invocation.json and provenance.json. frozen-source.json and source.patch bind the dirty source atop the recorded base revision to the exact binary. Child packages retain compact logs/traces, execution timing and source/asset identities; the original binary/payload/pack remain in the build directories. The launch-path typo is retained separately in launch-path-error.log and never executed the simulator.
