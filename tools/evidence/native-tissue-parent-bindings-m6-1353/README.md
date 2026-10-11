# Passive source-parent interpolation on the existing native GPU path

Refining a muscle surface by averaging its position and skinning weights can bend it away from its source parent triangle. The optional `passive_tissue_parent_bindings` field in the existing anatomy receipt now binds every refined vertex to positive raw NHTISS source-vertex coefficients. Scene loading expands those coefficients into the existing GPU influence buffer. The accepted body poses still own motion; no additional physics, GPU kernel, per-step CPU deformation, force, mass, or controller state is introduced.

Source and child payload hashes, registration, indices, duplicate terms, positive coefficients, and partition of unity are checked. A 2 micrometre registered-rest reproduction gate rejects displaced maps; this is **not** an anatomical intersection tolerance. Static CPU compilation buffers are released after upload. Existing receipts retain their existing path.

## Native evidence

Apple M6, SSH numi. The final executable SHA-256 is `147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174`. Final build004 is byte-identical to the native-tested build003; the only subsequent source edits guard the optional member in non-resting builds and indent the unchanged branch. Both resting build/link and non-resting syntax compilation pass.

* Identity references on the existing right temporalis: 16.00000076 simulated seconds / 105.3833 wall seconds. All referenced triangle positions of 867 surfaces and all body poses at eight accepted captures are bit-identical to the existing integrated scene. All 1,000 physiology rows match.
* Nontrivial midpoint regression: 16.00000076 simulated seconds / 106.1170 wall seconds. One FDS132 test vertex follows half of source vertex611 plus half of926 through the GPU. Maximum error versus the average of native source positions is 37.44 nm; the old averaged-position/averaged-weight helper differs by 198.36 micrometres at the witness pose. All other referenced vertex positions, body poses, and all 1,000 physiology rows are unchanged.
* Ten early native invalid-input cases reject the intended condition. A separate corrected test rejects a map that changes the registered rest geometry. The first rest-gate test stopped earlier on an invalid absolute cardiac-sidecar path; its failed harness result is retained.

The midpoint fixture moves one source vertex 18.24 mm in its source frame specifically to expose noncommutation. It is a mechanism regression in the integrated scene, **not an anatomically admitted repair**, and its recording must not be presented as the final human deliverable. No five-minute completion or performance improvement is claimed. Achieved factors are about 0.152 and 0.151 times real time.

The compact evidence preserves declarations, exact source snapshots, native traces, recording, checks, failed test-harness evidence, and generation scripts. Large source assets remain at the hash-pinned retained paths on numi. Historical build002/003 source paths have since received the guarded final edit; the exact executed bytes are archived under each build's executed-source directory. No historical execution receipt was rewritten.
