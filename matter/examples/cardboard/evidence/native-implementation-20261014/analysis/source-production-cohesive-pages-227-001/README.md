# Source227: production cohesive paging slice

This candidate is based on source225-002. It wires bounded face pages into the production cohesive force/residual path and four cohesive operator consumers while keeping face identity and accumulation order explicit. It preserves the full global FEM state, cohesive histories and face state.

The source snapshot is bound by frozen-source.json (823 source files). source-overlay.json binds the seven-file delta and patch; source-build.json binds the compile artifacts and CPU planner result. Configure/build and all four requested targets succeeded on macmini. The CPU planning check passed 30 checks.

The new production-kernel helper compiled but was not executed. It checks the accumulator clear/page gather/apply and operator-accumulator kernels on a synthetic two-environment case; it does not exercise actual face constitutive kernels or the complete runtime path. Root can run the helper with the command in source-overlay.json.

For the authored shipping count the plan describes three pages of 87,381 faces (last page 86,073) and 25,232,256 bytes for the two face-output pages plus node accumulator in the reported E=1 case. The planner still reports whole_process_admission=false; dense contact-pair sample storage/consumer remains unpaged. This candidate does not qualify a full shipping world or a whole-process memory peak.
