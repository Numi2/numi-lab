# Directional CCD instrument

Build only the production collision shader and native instrument:

~~~sh
cmake -S tests/ccd -B /path/to/build-ccd -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/build-ccd -j2
ctest --test-dir /path/to/build-ccd --output-on-failure
~~~

The same targets/tests are included by the root CMake project. The probe takes
a metallib path, geometric scale, environment count and timing-repeat count:

~~~sh
/path/to/build-ccd/bin/metalrobo_ccd_advancement_probe \
  /path/to/build-ccd/shaders/NumiCCDProbe.metallib 1 1024 9 > observation.json
python3 tests/ccd/analyze.py observation.json > analysis.json
~~~

Each case calls the production mr_world_resolve_ccd kernel with the existing
bound and with MR_METAL_WORLD_CONTACT_DIRECTIONAL_CCD. The integrated runtime
exposes the latter as MetalWorldStepConfig::directionalCCDAdvancement, default
false. No shader variant replaces the production distance query.

The instrument uses 128 advancement iterations to expose the distribution;
the engine default remains 16. Inputs reproduce the projected speculative
margin at speculative scale 1 of the producer's envelope.
Ten box/plane and box/box paths cover thin impacts, tangential misses, coherent
translation, relative catch motion, rotating edge hits/misses, and antipodal
quaternion representations. FP64 plane support / 15-axis OBB SAT supplies hit
brackets; misses also require an analytic continuous separation certificate.
The 65,536-sample hit bracket followed by bisection is a bounded reference,
not a general exact continuous collision algorithm.

Records and status buffers must replay byte for byte. GPU timing samples
exclude two warmups and are all retained in JSON. Environments repeat the same
sweep to supply GPU work; neither those replicas nor timing repetitions are
independent geometric units. Arm execution order alternates between cases.

Zero failures means no directional silent miss, false hit or late reported
bracket was observed **or the case was explicitly unresolved/rejected**. An
unresolved candidate can use the existing speculative fallback; it does not
establish that the physical interaction was resolved. The probe records these
states separately. This is not integration, friction, torque, rollback, force,
controller, broadphase or accepted-simulation-throughput qualification.

See [the measured decision](../../docs/CCD_DIRECTIONAL_EXPERIMENT_20261107.md).
