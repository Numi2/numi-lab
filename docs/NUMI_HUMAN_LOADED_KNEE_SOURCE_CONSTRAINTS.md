# Loaded-knee source-constraint execution boundary

The loaded-left-knee candidate uses the immutable NHEQ2 and NHLIM1 source-law
programs. Human authors those bytes. Matter owns their runtime forces and
accepted constraint state. Admitting the programs does not bind a prepared
state and is not evidence of runtime execution, standing, production physical
ownership, or clinical validity.

The authenticated Human source-compliance companion currently binds:

- NHEQ2: 5,792 bytes, 51 rows, SHA-256
  `12db05fddb492e77e7fd461fad566d3e1e75390f2cb6f77f26568254a6cb4477`
- NHLIM1: 9,840 bytes, 122 rows, SHA-256
  `c583611fcedc326a32c6f69504a65e675c8e0adc987c6db622ca0d95a02438d3`
- source archive SHA-256
  `280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975`
- source dimensions: `nq=129`, `nv=128`, policy 1, REFSAFE enabled

## Exact admission/configuration check

Build the checker and run the complete six-file admission chain. The manifest,
binding, ownership manifest, and source-compliance companion are required; the
binary programs cannot authenticate themselves:

```sh
cmake --build BUILD_DIRECTORY \
  --target metalrobo_numi_human_loaded_knee_source_constraint_check -j 8

BUILD_DIRECTORY/bin/metalrobo_numi_human_loaded_knee_source_constraint_check \
  /absolute/path/HumanPack.loaded-anatomy-knee.v1.json \
  /absolute/path/HumanPack.loaded-anatomy-knee.binding.v1.json \
  /absolute/path/HumanPack.ownership.v1.json \
  /absolute/path/HumanPack.loaded-anatomy-knee.source-compliance.v1.json \
  /absolute/path/myosim-fullbody-joint-equalities-source-compliance.nheq \
  /absolute/path/myosim-fullbody-joint-limits.nhlim
```

Success prints `candidate_only=true`, `runtime_initialized=false`, and
`runtime_executed=false`. The checker opens regular files without following
symlinks, verifies exact SHA-256 and byte counts, validates both complete
headers and every scalar row, and constructs the Matter configuration fields.
It does not initialize or step Matter.

## Runtime call site

After the loaded-knee base manifest and additive source-compliance companion
have both been authenticated, the owning execution path performs this binding
before its one `Runtime::initialize` call:

```cpp
metalrobo::NumiHumanSourceConstraintProgramV1 constraints;
std::string error;
metalrobo::NumiHumanLoadedKneeBindingAdmissionV1 base;
require(metalrobo::loadNumiHumanLoadedKneeBindingV1(
            manifestPath, bindingPath, ownershipPath, base, error),
        error);
metalrobo::NumiHumanLoadedKneeSourceComplianceAdmissionV1 source;
require(metalrobo::loadNumiHumanLoadedKneeSourceComplianceV1(
            sourceCompliancePath, equalityPath, limitPath, base, source,
            error),
        error);
require(metalrobo::loadNumiHumanLoadedKneeSourceConstraintProgramV1(
            equalityPath, limitPath, base, source, constraints, error),
        error);

numi::matter::RuntimeConfiguration matterConfiguration{
    .metallib = matterMetallib,
    .environmentCount = 1u,
    .captureEvents = true,
    .captureDiagnostics = true,
    .automaticIdentification = false,
    .adaptiveTransfer = false,
};
require(metalrobo::configureNumiHumanSourceConstraintsV1(
            constraints, matterConfiguration, error),
        error);
const auto initialized = matterRuntime.initialize(
    compiledLoadedKneeWorld, matterConfiguration);
```

`constraints` must remain alive until `initialize` returns because the
configuration borrows its row spans. The initialized runtime is then supplied
to the existing `MetalNumanXHumanMatterContext`; that owner already supplies
the accepted source velocity and effective-tangent factor to Matter and owns
prepare/ACK/apply/publication/rollback. The deferred tendon/FEM program may
lend its external-force field to that same request, but it must not create a
second Matter transaction.

The standalone visual probe still enters the legacy NHEQ1 equilibrium compiler
and does not construct `MetalNumanXHumanMatterContext`. NHEQ2/NHLIM1 must not be
routed through that path. Its NHEQ1 behavior remains unchanged until the loaded
knee is called from the NumanX lifecycle above.
