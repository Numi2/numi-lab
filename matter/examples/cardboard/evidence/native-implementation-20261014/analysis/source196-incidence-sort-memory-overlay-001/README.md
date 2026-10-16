# Source196 incidence-sort memory accounting overlay

This isolated overlay is based on frozen source195:
 /Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-contact-damped-cgls-memory-195-001

It changes only the shipping memory estimator/probe and its CPU check. Source195 already has the runtime selector, exact-value admission, padded-capacity checks, and the two per-environment buffers. The patch leaves that runtime implementation untouched and accounts for it in prebuild and compiled memory reports.

For contact capacity C and environment count E, runtime computes logical=6*C, padded=bit_ceil(logical), then allocates E*padded uint64 keys and E*padded uint32 payloads. The checked byte count is therefore 12*E*padded. The estimator requires the same nonzero counts, uint32 payload identity bound, logical-domain <= 2^31, padded dispatch bound, node-count dispatch bound, and host-size arithmetic. It adds this amount to persistent contact memory only when NM_CONTACT_NODE_INCIDENCE_SORT is exactly '1'; absent means disabled and other strings fail closed.

At C=131072 the padded count is 1,048,576, giving 12,582,912 bytes for E=1 and 25,165,824 bytes for E=2. The prebuild default of eight candidates per tet is also checked (32 tets -> C=256 -> 24,576 bytes). The overlay does not raise any topology, allocator, or contact-capacity limit.

The shipping probe now captures and strictly parses the option in both compiled-memory input and prebuild-memory options, reports the selected state and workspace byte count, and propagates malformed values as a configuration/admission failure. The source-bound diagnostic environment-key registry already contained this selector in source195 and is unchanged by this patch.

## Validation

- CPU memory check: compiled with C++23, warnings-as-errors; passed with 'shipping memory preflight checks passed'.
- Allocation preflight consumer: compiled with C++23, warnings-as-errors.
- Objective-C++ shipping probe: compiled using source195 compile-command flags; passed with one pre-existing unused-function warning.
- Old-header control: the updated check source fails against the frozen195 header because the new parser/workspace APIs are absent, as expected.
- Patch replay: applied to a minimal copy of the three base files; every resulting source byte matched the candidate.
- No full source build, Metal runtime execution, GPU dispatch, or shipping-capacity qualification was performed.

Several early compile invocations are retained in cpu-build/ as harness history; only the files under cpu-build/final/ are the final validation receipts. The unchanged allocation-preflight translation unit in candidate/ is included as a compile context file and is excluded from the patch.
