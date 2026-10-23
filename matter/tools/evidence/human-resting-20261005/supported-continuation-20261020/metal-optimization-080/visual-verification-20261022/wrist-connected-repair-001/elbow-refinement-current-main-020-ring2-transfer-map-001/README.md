# D20 ring-2 transfer-map diagnostic candidate

This isolated candidate combines the unchanged D19 refined skin/fat geometry and explicit child-tet labels with the tested two-ring, stable-source-face transfer search. It also changes hand/wrist replay telemetry to report `not_requested` when deterministic replay was disabled. The ring-2 transfer map is source registration only; it does not alter FEM geometry, force laws, materials, or admission gates.

Candidate ID: `elbow-refinement-current-main-020-ring2-transfer-map-001`  
Candidate identity SHA-256: `a56b4c8d6eeec7b918ecb446c5ba8800928d4a2977c294ce4a8b5f23af58fe4f`  
Parent D19 manifest SHA-256: `eb46feff07a3acc226341cf8755b17a2e374649643a7e03bd724221fdab55cf5`  
Ring transfer owner SHA-256: `b7f23c1d9f7279e05f995fcc1cb1d417675f3ffac845220e42ffc8204cf2a72d`

The 4,777 face rows carry stable source-face IDs that can repeat. D20 passes those IDs aligned with the current topology and uses two edge-connected face rings, with deterministic stable-ID tie-breaking. The transfer is a bounded source-registration map, not a global surface or anatomical-coverage guarantee. Physical gates and D19's unadmitted status are retained. No native run is included in this preparation.

## Verification

The D20 app compiled and linked with `-Wall -Wextra -Wpedantic -Werror`. The ring-2 selector test passed against the D20 transfer header, including legacy default parity, exact source stability, connected-component and donor-mismatch handling, deterministic stable-face tie-breaking, projection monotonicity, partition of unity, and invalid-bound rejection. The independent explicit child/parent-label test also passed (27,714 parents, 28,632 children, 4,777 faces). The source-only launch snapshot is frozen and has not yet been executed.
