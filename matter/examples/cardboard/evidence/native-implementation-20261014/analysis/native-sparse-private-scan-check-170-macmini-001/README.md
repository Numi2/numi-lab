# Source 170 sparse private-scan checker — macmini build

This host-bound package contains a fresh arm64 checker binary compiled on macmini from the exact sealed source-170 checker and linked to the exact source-170 O3 metallib. It compiles against source-173 `shared.h`; the header SHA-256 matches the source-169 base header byte-for-byte (`cd173c…`). Source 173 and its build tree were read-only inputs.

The package carries the 49-case inventory (6,366,845 environment-pair slots; largest case has 6,335,568 pairs and a 65,536 candidate cap), the source-170 overlay manifest/patch lineage, source/build bindings, and the exact compile command. The sparse checker itself still contains 49 GPU differential fixtures.

Run `/opt/homebrew/Cellar/python@3.13/3.13.14_1/bin/python3.13 runner.py` for preflight only. It never dispatches Metal by default. `runner.py --verify-only` checks package/source/build/hash bindings without dispatch. Only `runner.py --execute` requests the GPU checker. Execution is guarded by a single-worker check, a 2 GiB minimum free-space check, a 128 MiB actually-written reserve, a 512 MiB abort floor, and a 900 second timeout. The parent task owns GPU queue authorization; this package was not executed with `--execute`.

Validation performed on macmini: fresh Objective-C++ C++17 compile with warnings-as-errors succeeded; `sparse-scan-checker --cpu-only` passed three CPU contracts and reported `metal_kernels_executed=0`. No native checker dispatch, performance claim, or solver/physical qualification is made.
