# Historical raw reader build correction

The optional source370 target compile found five mr_float4 names in the explicit v45 raw-state compatibility struct; this Matter-only tool declares nm_float4. Corrected those five type names to nm_float4. Compile now passes. The 80-byte legacy layout and 96-byte current layout assertions pass. No runtime/Metal/physics bytes changed after the042d tests. No historical full-package execution is claimed. The failed and passing build logs are retained.
