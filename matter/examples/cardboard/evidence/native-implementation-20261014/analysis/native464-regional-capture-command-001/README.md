# Source151 regional CGLS capture command

This package prepares the source151 diagnostic run at `solver/cgls-regional-state-capture-464`. It reuses the exact physical, material, Newton, FGMRES, phase, and acceptance arguments from source149 case 462. It removes only the legacy coupled-capture selector/path, which cannot admit the two-environment case, and adds `NM_FEM_BASE_CGLS_STATE_CAPTURE_REGIONAL=1` alongside the existing state-capture selector. The CGLS selected root/Newton, iteration cap, and primary fallback settings remain those in the command JSON.

`native-464-command.json` is the exact argv for the source-bound runner. `run_native464.py` performs read-only source/build/input and disk preflight, then can launch the command; it has not been run. The planned case path must remain absent before launch. The root owns GPU dispatch.

The exact compiled package is parsed to reconstruct the graph layout: 2 environments, 882 nodes (36 fixed, 846 free), 1,920 tetrahedra, 27 graph patches, 2,089 patch-node entries, 6,267 DOFs, and FNV-1a layout checksum 1,292,406,985. The conservative readback upper bound is 82,441,544 bytes against the unchanged 96 MiB cap. This is a capacity preflight, not proof of runtime capture completion or solver/physical qualification.

The v2 reader validates the joined state, payload hashes and extents, graph checksum, and pivot permutations. Augmented RAP remains inadmissible for same-Newton replay: the captured generation tag describes producer restart cycle `c`, while consumption is at `c+1`; this capture does not independently record the consuming cycle. A matching tag prefix, validity flag, or factor alone cannot establish currentness.

No source files or frozen snapshots are modified by this package, and no native 464 execution has occurred.
