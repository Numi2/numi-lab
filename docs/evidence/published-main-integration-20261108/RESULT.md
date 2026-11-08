# Published main integrated into research candidate

Merge f62970afd2667ab7953b26d3cffc827be6109615 has published
main 910aa100b10de9e95115bfffd0b45456a432e5ad as an ancestor. The two
conflicts were in policy_selection.py and MetalWorld.mm. The integration
parent already contained the same imported-policy failure veto and the
initial-private-upload abort guard. Its versions were retained because they
also preserve measured-dove failure handling and the integration-only device
mechanics guard. The first-parent merge difference is the retained published
main evidence directory; source and tests matched the integration parent.

The existing incremental build recompiled MetalWorld and linked the native
library, trainer, rollout and Window checker. Four CTests passed: policy
status, Window metadata, Window bootstrap and training cancellation. Bootstrap
and cancellation use stubs; they are not a live UI/trainer check. Both existing
previously compiled private-upload owner probes were rerun against the freshly
rebuilt library and matched control versus rejected-then-retry bytes (2016 each).
Their binary, source, library, shader and verifier hashes are retained.

All 45 selector tests other than the known guided-G1 reset-only assertion
passed. The unfiltered 46-test run still has that one failure. A fresh
one-environment, eight-step, one-update synthetic Franka training and matched
selection smoke at this merge passed and retained the incumbent; its own plan,
runtime identities and output are under ../franka-train-merged-20261108.
It does not demonstrate a learned skill or physical qualification.

The contact reference was running concurrently, so this is not a performance
measurement. The integration checkout remains sparse and the build incremental.
This is not a clean machine, an installed release, or an end-to-end visible
Window test. The integration branch has not been published.
