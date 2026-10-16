# Source215 pure FEM Armijo/CGLS arena

This isolated candidate is based directly on source-true-residual-8192-213-001 (frozen source SHA-256 70976d03d6b096d6d40cf44b853d7f2118b44f74aa4a23c72a0c6b5a0b5225fa). It is not based on source210.

It adds checked admission/planning for N8192 residual-Armijo scratch and optional Cauchy/base-CGLS/right-scale scratch, limited to one static explicit FEM object with deformable self-contact disabled. Contact, cohesive, rigid, field-boundary, puncture, MPM, and unsupported object/matter modes fail closed. Existing non-pure-FEM and contact caller limits remain unchanged. No new environment key is introduced.

For E=2, N=8192, T=32768 with CGLS and right scaling, the planner accounts 25 private buffers totaling 12,714,240 bytes. It checks each buffer against MTLDevice.maxBufferLength, checked 64-bit arithmetic, uint32 dispatch/count limits, the 16 MiB scratch ceiling, and allocator-reported resident bytes plus planned scratch against 16 GiB. This is not a whole-process peak measurement.

Validation against the corrected active frozen-source.json: the runtime library, CPU planner checker, and native N8192 Metal helper all compiled in buildfix004. The focused CPU planner CTest and direct planner executable passed. The native helper was not run; no GPU dispatch occurred. The first frozen map with an incorrect parent SHA is retained as frozen-source-attempt001-stale-parent-hash.json; the active map binds source213's actual frozen hash. Earlier Release/assert, undefined shader-directory, and missing-framework link failures are preserved.

The source delta is in overlay.patch. Build commands and artifact hashes are in source-overlay.json and build-commands-final-004.json. Build logs are preserved.
