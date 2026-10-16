# Preserve rejected AP constitutive inputs (source163)

The probe writes four arrays already present in its bounded RuntimeStateSnapshot: accepted history at AP linearization, scheduler rate exponents, and material-local F high/low words. The prior writer exported candidate nodes and AP/RAP while dropping these arrays at process exit. New metadata labels accepted history explicitly and marks exact Newton corrections and perturbed projected states unavailable. Old failures cannot be retroactively filled.

No solver, target generation, constitutive program or Metal source change; zero added GPU readback bytes. Runtime already copies and size-validates the vectors under its regional64MiB snapshot bound. The overlay only adds disk export.

Four CPU O0/O3 full/empty fixtures exercise the source-derived actual production writer and verify emitted byte counts and SHA-256 values; O0/O3 output bytes match. Full probe TU syntax and source157-relative patch dry-run pass. No native build, capture or trajectory parity is claimed. Two validation setup failures are retained: ambiguous probe/box-driver compile commands, then a missing include path for an external candidate file. They did not change native evidence or the candidate patch.
