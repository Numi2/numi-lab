# Existing coupled cavity controls

Both default0 and enabled1 policies passed the existing production cavity
fixture on Apple M6: pressure/volume and wall-force JVPs,32 accepted moving-wall
steps in two environments, blood/tracer conservation, fixed-wall control,
bitwise snapshot replay, isolated rejected-environment rollback and reset.
Numerical tolerances remain those of the existing fixture.

The test source is unchanged at the recorded base revision and its hash was
checked against git show. Runtime and shader working-source hashes, executable,
compiler/runtime libraries and metallib identities are in the provenance file.
These are synthetic cavity-interface controls, with no biological calibration
or whole-body qualification. No wall-time/performance claim is made.
