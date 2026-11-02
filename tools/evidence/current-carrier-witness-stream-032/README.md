# Current carrier-witness streaming controls (032)

This source-only Lab increment adds bounded single-member gzip reading and streamed carrier-witness allowlist binding to the published support header. The owner CLI reports the streamed CSV row count through carrierWitnessCSVRowCount; the binding no longer retains the full parsed witness-row vector.

The focused fixtures exercise gzip/plain parity and compare streamed binding against the legacy full-vector binding, including exact CSR/hash parity, row and byte limits, CRC/truncation/trailing-byte rejection, wrong source role/one-hop/distance rejection, and atomic support publication on failure. They compile against the exact apps support header in this publisher checkout. The retained depfiles confirm that the publisher header is selected and no isolated 031/032 trial header is selected.

The compact receipt records source and upstream pins, compiler, fixture outputs, dependency closures, and CLI syntax/link checks. Test binaries and generated fixtures are temporary and are not included.

This package is not a mass-transfer result. It contains no allocator run, native simulation step, or physical admission claim. The out008 allocator attempt is separate and failed before transport flow; it does not change this package's source-control result.
