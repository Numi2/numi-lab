# Run owner artifact load diagnostics

A previously retained native Window launch failed with the generic message
`one or more WorldPack run owner artifacts failed to load`. That result was
preserved in `../inspector-failure-reporting-20261108/`. Rerunning the same
incompatible inputs after this repair returns exit 1 and identifies the actual
failure: the TaskPack wire version is unsupported, with its file path. The
repair applies the same owner, path, status and loader message to TaskPack,
RobotActuatorPack, SensorProgramPack and RealityProgramPack for both WorldPack
and imported URDF runs. Failure ordering is deterministic; this is a diagnostic
change and does not convert an incompatible input to a successful run.

A native C API regression first failed against the old generic diagnostic. It
then passed all four missing-artifact cases for both source modes and a corrupt
TaskPack case. The independent Window bootstrap CTest also passed. A matching
synthetic Franka fixture completed an eight-step native inspector rollout,
while the deliberately wrong camera was still rejected; plan, runtime and
input hashes and output are retained. These checks do not establish UI
interaction, training, physical accuracy or general package compatibility.
