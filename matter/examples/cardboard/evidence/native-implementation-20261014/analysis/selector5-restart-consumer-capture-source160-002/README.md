# Selector-5 restart-consumer capture overlay (source160-002)

This external, default-off diagnostic overlay is based on immutable source157 and preserves source160-001 as a separate historical artifact. The capture point remains immediately after selector-5 graph apply and its buffer barrier, before the FEM operator consumes `z`. It changes no solver equation, accepted state, or certificate.

The selector remains `control:physics:newton:restart:column:environment`, but configuration now fails closed unless `maximumRateExponent == 0`, `microtick == 0`, and `microtickCount == 1`. The capture-site gate repeats that scope check, and an explicit one-shot claim suppresses any duplicate selector match. Schema v2 records the selected microtick, microtick count, micro-generation, maximum rate exponent, physics-substep count, and one-shot attempt fields. The validator rejects v1, nonzero/multiple microticks, inconsistent micro-generation, or non-unique attempt metadata. These restrictions prevent same-selector readback duplication under a multi-microtick execution.

The payload cap is 64 MiB for the one selected-environment payload. The size preflight uses source464's exact 882-node, 1,920-tet, 27-mode, 4,721,337-graph-float layout and the source160 append inventory (including the optional algorithmic tangent cache): 36 arrays totaling 39,825,484 bytes. The two graph matrix arrays alone account for 37,770,696 bytes. The previous 32 MiB cap rejects this layout by 6,271,052 bytes; the 64 MiB bound leaves 27,283,380 bytes of payload headroom. This is a payload/snapshot bound, not a measured whole-process or Metal-driver peak. Existing source buffers and driver copy overhead are not included. Other FD/RAP/CGLS and Newton-trace readbacks are rejected while this capture is enabled; in particular `NM_FEM_NEWTON_TRACE_ROOT` is now included in the exclusion list.

## Validation

- C++ selector, one-shot, microtick-scope, and source464 exact-payload-size contract test: pass.
- Python capture checksum/schema/RAP reconstruction and one-shot/microtick failure controls: 6/6 pass.
- Objective-C++ source syntax (`-Werror`): pass.
- Source464 size-only preflight: pass; 39,825,484 bytes for the exact reviewed append layout.
- No project build, native capture, or GPU run was performed.

`source-overlay.json` binds the runtime/header source changes and hashes. `source-overlay.patch` is generated against source157. `preflight/source464-size-preflight.json` records every array's size and the source464 manifest hash. This package is not a solver/convergence qualification.
