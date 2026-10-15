# Prepared native461 command and inputs

This is a source-bound preparation only; no native command was launched. It derives from retained native459, changing the source/build to snapshot149/build149, relocating the fresh output/capture paths, adding `NM_FEM_BASE_CGLS_RIGHT_SCALE=block3`, and binding the sealed149 integration, overlay, checker, and build evidence. Physical protocol flags remain the same: CGLS64, capture Newton48, primary-after48, 12/3/6/3/3 phases, 8x2x1 mesh, dt 0.0001, local material budget32, FGMRES restart16/1024, tolerance1e-6. The post-release relaxation option remains omitted and therefore defaults to zero in source149.

The root should run only through its serialized GPU schedule. The package does not assert native CGLS parity, convergence, or performance. Native457 results were mixed; no benefit is inferred.

Canonical command: `native-461-command.json`. Source149 input binding list: `149-input-bindings.json`; copied unique-basename manifest files are under `input149/`.
