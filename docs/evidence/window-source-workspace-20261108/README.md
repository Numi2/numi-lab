# Source runtime with an independent Window workspace

The source dispatcher did not discover its checkout's `.numi/commands/window`
when launched in a different project. Even when invoked directly with
`NUMI_LAB_ROOT` set, the Window command configured the project as the CMake
source instead of the runtime checkout.

The source dispatcher now searches the source checkout's project commands
after project, user and regular runtime commands. Window uses `NUMI_LAB_ROOT`
as the source directory and retains the project workspace for its scene and
build paths. The fallback for a directly invoked command without a runtime
root remains the workspace.

Before the fix, the separate-workspace discovery regression failed with
`unknown capability 'window'`. An independently instrumented bootstrap check
on the integrated Window implementation failed with `rollout configured from
the wrong source`. After the fix, the main-branch CTest
`metalrobo_numi_window_source_workspace` passed on the M6. Its stubbed CMake
and rollout exercise a workspace containing spaces, source capability
discovery, scene forwarding, correct source selection, and project override
precedence. It is a shell contract check, not a native rollout, visual, or
performance qualification. Both research contact options remain disabled.
