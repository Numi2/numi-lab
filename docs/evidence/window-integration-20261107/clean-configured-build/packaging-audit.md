# Packaging and checkout audit

The retained fresh configure and bounded target build are not an installation.
A first `cmake --install` attempt to a new, isolated prefix failed after copying
two Matter shader libraries: the bounded build had not built the install-listed
`libnumi_matter_runtime.a`. The partial prefix is retained on the host and the
exact installer output is in `package-stage-failed.log`. This establishes that
these selected build targets alone do not form an installable package; it does
not establish that an appropriately complete build cannot install.

The install declaration includes the dispatcher, trainer and rollout, but does
not install the workspace-only `.numi/commands/window` command or the Window
catalog compiler. The dispatcher searches the workspace command directory and
then the installed `libexec/numi`; outside a workspace, an installed package
therefore has no installed Window command. Simply copying the existing wrapper
would not complete the integration: it assumes a source workspace, invokes CMake
and uses workspace-local builds and assets. A supported installed Window path
needs a defined runtime and catalog location, package entries and a native
out-of-workspace test. The MetalWorld source already searches for the installed
library-relative `lib/metalrobo/MetalRobo.metallib`; relocation is not claimed
broken or qualified without a staged runtime test.

The built-in catalog requires a G1 URDF, Franka collision meshes and PX4 mesh
tree at the three paths listed in `.numi/commands/window`. None is tracked at
those paths in revision `e4d80357`. Their presence on a developer machine is
not a reproducible package input. The existing explicitly supplied synthetic
scene remains a bounded bootstrap fixture, not a replacement for those assets.

The controlled worktree's sparse patterns deliberately omit the tracked
packaging inputs under `.agents`, plugins, docs, licenses, schemas, Matter
materials and the dispatcher script until needed. A complete packaging test
must materialize those tracked sources, build every required install artifact,
stage into an isolated prefix, and exercise dispatch and native execution from
outside the source checkout. The fresh CMake build selected an interpreter
without NumPy/SciPy, so the associated optional Python tests did not run; MLX
and NumPy are additional runtime requirements for guided training. No guided
window panel or cross-owner snapshot restore was qualified by these checks.
