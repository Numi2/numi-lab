# Native quality run 019

Run outcome: failed after 18 accepted coupled rows of 32 requested at 125 us. This is a failure witness, not a pass. Invocation method scope is mechanics-only, with a 4 ms requested horizon.

The accepted CSV spans step 1 at 0.000125000005937 s through step 18 at 0.00225000010687 s (0.00225000010687 s accepted simulated time). Process wall time was 321.227655875 s. This is not a real-time throughput claim.

Contact: sum of recorded normal impulses across accepted rows = 0 N s; maximum per-row normal impulse = 0 N s; peak penetration maximum = 0 m; minimum recorded contact gap = 1.52555903696e-05 m. Therefore sustained_contact=false (zero recorded normal impulse).

Mass at final accepted row: reference positive-body mass = 82.7159423828 kg; candidate = 82.7159423828 kg; external mass = 0 kg; body-mass residual = 0 kg; maximum absolute residual = 0 kg. Transfer momentum and angular residuals are 0 kg m/s and 0 kg m2/s. Root assistance at the final row is 0 N and 0 Nm; maxima over accepted rows are 0 N and 0 Nm.

Ligament/fascia ownership at initialization: 28 passive ligament force rows, 28 registered ligament rows, 0 legacy fascia-reduced ligament rows, 2 ligament FEM objects; fascia admission reports 26 active source regions, 12 material objects, 1202 nodes, 2127 tetrahedra, source volume 0.000129712913 m3 and mass 0.129712913 kg. The source records include the bilaterally inferred ACL FEM and the registered regional rows. These are runtime admission diagnostics, not a material calibration claim.

FEM failure: after 18 accepted steps, Matter status 5 (invalid deformation) occurred on passive-tissue object 15, global tet 21728. The stderr witness identifies the BodyParts3D whole-body skin-fat bonded shell, local tet 17981, rejected-trial J=0.33994558453559875, accepted effective J=0.36213095505356147, and state rolled_back_accepted_not_rejected_candidate. Preserve that distinction: the reported accepted coordinates are the effective rolled-back state; the rejected trial is not the accepted geometry.

Whole-mesh final minimum/maximum J are unavailable: NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE=0 in the invocation, so no full endpoint state was captured. The failure receipt supplies the failing tet's rejected J only; stderr supplies that tet's accepted effective J. No global extrema are inferred from those two values.

The skin/fat preparation line reports the global inner-boundary quality correction, with original exterior/topology/ancestry and donor mass debits preserved. Native evolution remains unqualified. execution.json records exit code 1 and wall time above.

## Run binding

- Invocation SHA-256: 382cdd35657d7f3e1091232e2a0f621bef74e5e722eecac53d88030f3b5992d1
- Provenance SHA-256: 6ddcbf39e1d220013e7a9033560f64ae28228a99a48d800fe4b63d09a9b9c1d4
- Coupled CSV SHA-256: d04e4060d51da2237ce3926b5fe0df5e7cecff02306716b6a86c53efd433be16
- Registered ligament header SHA-256: a1d5adb7a6fc601b0e97dcebe1bbb2d36010bdcefe935bf424943d9b56528d25
- Input files and SHA-256 values are listed in input-assets-sha256.txt; input assets were not copied.
