# Owner diagnostics on the integrated Window branch

The merged revision in `tested-revision.txt` combines the main-branch owner
artifact diagnostics with the newer Window. Its compiled native library passed
the owner-load regression and both Window bootstrap/source-workspace CTests.
A separate bounded native Window probe rejected an invalid camera with exit 1
and completed a synthetic Franka runtime check with exit 0, including recovery
after a deliberately unavailable selection. Plans and results include source,
fixture and runtime hashes. The probe is a launch/recovery qualification, not
a manual UI, training, physical accuracy or performance result. The unchanged
main-branch diagnosis and its original incompatible wire-version case remain in
`../run-owner-diagnostics-20261108/`.
