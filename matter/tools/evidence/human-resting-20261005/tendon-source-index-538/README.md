# Preserve tissue source vertex allocation in the native viewer

The right and left Achilles source meshes each allocate 2,353 vertices but their first referenced vertex is 43. The viewer previously subtracted the first referenced native vertex to look up source deformation bindings. That selected the wrong source vertex and produced 23 exact zero-area triangles per Achilles mesh.

The viewer now derives the full source allocation base from native/source index correspondence, checks every triangle index and the complete allocation bounds, and uses that base when reading existing deformation bindings. The source geometry, MyoSim/compliant-tendon force owner, and all physical update kernels remain unchanged.

A 32-step native Metal run on the Mac mini captures actual accepted GPU vertex buffers at presentation steps 0 and 31. Both tendons have zero collapsed or nonfinite triangles. All 859 other surfaces at initialization have bit-identical position/normal records. The coupled trace, COM trace, and support impulse prefix are exact matches to the unchanged baseline. Transaction rejection/retry probes pass. Independent integration of the seven cardiac surfaces passes both captured endpoints, with maximum relative volume error 6.835e-6.

The exact coordinate-quotient self-intersection audit still reports right/left Achilles pair counts 115/64 at step 0 and 71/19 at step 31. These retained findings are not waived by the indexing repair; complete anatomical acceptance remains open. The source tendons also retain declared anatomical insertion boundaries, so this regression does not claim closed tendon surfaces or clinical qualification.

The JSON reports bind the input tissue source, exact native captures, runtime metadata, and tested source. Large source assets and continuous framebuffer recording remain at the retained Mac mini paths. check-source-index-regression.py is the exact host-bound audit driver, not a general artifact format. The recorded false passed_surface_index_fix in its report combines zero-area and self-intersection checks; verification-summary.json separates the successful indexing/physical noninterference regression from the remaining geometric defects.

Source geometry uses BodyParts3D/FMA identities. Preserve the source receipt's CC BY 4.0 notice and the embedded OBJ's CC BY-SA 2.1 Japan notice. This mixed-source reference adult includes inferred registration and is not measured anatomy of one person.
