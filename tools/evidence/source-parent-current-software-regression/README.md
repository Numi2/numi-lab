# Source-parent presentation regression

Literal self references and individually byte-identical source vertices reuse
admitted child influences after all descriptor, coefficient, index and load-budget
checks. Mixed surfaces keep nontrivial interpolation for changed vertices only.
This avoids rounding unchanged positions and normals through a parent round trip.

Dense parent positions use compensated accumulation. An optional source build
pins a preprocessed shader by SHA-256 and compiles only the existing resting skin
pipeline at load time with safe Metal4 arithmetic and precise functions. Other
physics and physiology pipelines keep their existing library.

Generate the source with tools/prepare_human_resting_skin_source.py, supplying
--worktree, --brain-source, --output and --receipt. Configure CMake with
-DNUMI_HUMAN_RESTING_SKIN_SOURCE=<generated source>, then build numi-human-native
and numi-human-parent-precision-gpu-check. The GPU checker accepts the generated
source, optionally a published baseline source and a term count from 1 through 512.
It requires an actual Metal device. A changed configured source is rejected.

qualification.json records native identity and mixed midpoint regression,
all 11 negative-admission cases and arithmetic coverage over every admitted
influence count. These are software qualification results. The separately
preserved dense-parent anatomy candidate still has three exact self intersections
at step 7423, including with its conditioned index counterfactual. This publication
does not admit that anatomy, extend physiology, or claim a performance gain.
