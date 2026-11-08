# Current main compatibility

The integrated Window branch and main have different renderer implementations. Before publishing the regression to main, compile the exact retained world-depth fixture against main source headers and a freshly checked main Metal library and shader. Run two captures and one capture-free control with the same physical inputs and acceptance checks. The expected contract is identical, but do not require main and the integration branch to produce identical geometry or pixels. Bind main source revision, probe, runtime, shader and verifier by hashes. Retain a main failure without weakening the test or merging the integration branch.

If the direct main check passes, add only the self-contained fixture and registered test to a separate main-based checkout, build that version, and rerun the new CTest and relevant visual tests before publishing the scoped result. This does not establish a unified release, sensor time correctness, or policy continuation.
