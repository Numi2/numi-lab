# Integrated catalog-status build

A fresh isolated worktree at revision `54def09736bd8b156077bf04381e3327f79000db` configured with Ninja and BUILD_TESTING=ON. The rollout, trainer, Window metadata checker and policy-status checker and their required native libraries and shaders were freshly linked. Two CPU CTests and 28 Python selector tests passed. Exact build logs, file identities, source revision and status, and test output are retained. No Metal workload was executed. This does not qualify the visible UI, a physical control policy, or a complete install.

File creation and modification timestamps bound the CPU configure/build activity; compare them to concurrent studies before interpreting any study timings. The prior staged Window package was not modified.
