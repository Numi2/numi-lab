# Source 154 build-topology erratum

This separate erratum supplements, and does not modify, the sealed
`analysis/source154-newton-budget-integration-001` package. The original
integration remains byte-for-byte intact.

The source-154 build did not use the same CMake source root or artifact set as
the source-149 and source-153 builds. Its exact command used
`--cmake-source-relative matter`; the build receipt records
`cmake_source = <source154 snapshot>/matter` and produced only the probe plus
the two NumiMatter metallibs. The source-149 command used
`--cmake-source-relative .`; the source-153 receipt records the snapshot root
as `cmake_source` and includes `lib/libmetalrobo.dylib` plus the wider shader
artifact set. Therefore the 154 result is a successful targeted Matter probe
build under the `matter` CMake subtree, not a full-root build with identical
build topology. Its 128-iteration parser/compiler-admission boundary remains
unchanged, and it did not launch a native simulation.

`bindings.json` records the exact source, command, receipt, and comparator
hashes. This is a provenance clarification only. It makes no solver,
convergence, performance, physical, or shipping claim.
