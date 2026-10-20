# FEM backtracking negative control 030

This retained negative control is bound to source recovery revision 567b527fe7d21cc7234d42d04aa7e75398ce162b. It uses the old shader/metallib (hashes and paths are in source-binary-reference.txt; neither binary is copied into this evidence package).

The old-metallib test exited 1. Its exact stderr says: fem reference check failed: nonleader invalid tet was omitted from FEM backtracking reduction alpha=1 minJ=1. This is the expected negative-control behavior: the old reduction omitted a nonleader invalid tetrahedron, leaving alpha=1 and minJ=1 despite the bad thread-57 witness. The compile/link exit-code files and command lines are preserved verbatim; their recorded codes are 2 and their stdout/stderr logs are empty.

This package retains only commands, exit codes, logs, and hash references. It excludes the old test executable and metallib; the originals remain in the build directory. The patched reduction is independently covered by verification-032's focused fixture, which reports lane 57 rejected J=0.3139282, accepted alpha=0.5, accepted minJ=0.656964, and passing fixed-node and all-valid controls.
