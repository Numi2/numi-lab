# Installed Window package candidate

Source commit `9e6e506b` builds and installs the Window command, catalog and
Booster compilers, and contract-check binary along with the existing rollout,
training runtime and library. An installed wrapper resolves the runtime and
metallib relative to the package prefix and does not invoke a workspace build.
The guided-training environment adds the installed dispatcher directory to PATH.
The four CPU checks include installed-mode shell tests, but the shell and
training-cancellation cases use stubs.

A full selected install target build succeeded; the previous partial-install
failure remains in the separately retained clean-build evidence. The resulting
prefix was copied to a different location. From a workspace outside the source
checkout, installed `numi window --help` worked, and default launch reported the
three missing G1, Franka and PX4 presentation-asset paths. Installed
`numi window --check` passed with a separately supplied synthetic tetrahedron
visual and a 16-step, one-environment bound. The run exercised failure and
restoration of a missing scene selection. Output, command, timestamps, fixture
and installed binary/library/shader hashes are retained here. The synthetic
visual is not Unitree G1 geometry. This is not evidence of real guided-window
training, policy quality, physical accuracy, saved-state restoration or broad
release qualification.
