# Midpoint endpoint scratch: native device evidence

The research branch now contains a Metal kernel that reads accepted and
midpoint reference-mode FEM nodes, constructs reflected endpoint nodes in a
separate scratch array, and produces per-node result codes. It checks finite
inputs, fixed-node consistency and conservative interval bounds for accepted
and endpoint clearance against a fixed plane. An invalid or uncertified node
does not write its endpoint slot. The kernel has no writable accepted state,
contact history or runtime status argument. It is **not called by Matter's
runtime**, does not enable an option, and is not a complete midpoint step.

The first native harness run at source 10af8cd7b failed because it requested
the unqualified shader symbol. That failure and the full build output are
retained in run-001.log and build-001.log. The harness was corrected to use
the library's actual namespaced function; run-002.log then passed the 12
predeclared cases.

The follow-up at source cf19c7048 passed those cases again and a separately
planned 257-node mixed batch on Apple M6. It certified 88 nodes and rejected
169. For every certified node, an independent host FP64 evaluation of the
actual Float32 position limbs verified the start and end plane inequalities,
and checked reflected displacement and velocity. Each rejected output
retained its sentinel; the accepted and working inputs remained byte-identical.
This checks these samples and encodings, not every possible Float32 input.

The rebuilt shader passed eight selected CTests, including the existing
native residual/rollback regression, fixed-area policy tests, CPU reference
tests and the new Metal gate. Build and CTest output, a repeated native run,
source and binary hashes, and both plans are retained here. The earlier shader
compile emitted pre-existing warnings in unrelated kernels; the complete log
is retained. These are correctness runs, not performance measurements.

The next owning change must integrate an isolated half-step mechanical solve,
retain the physical clock, check the midpoint trajectory relation, validate
the reflected element Jacobians and material state, convert physical impulses,
rebuild and certify the physical residual, and gate atomic publication and
replay. A per-node success code does not itself validate element inversion,
force accuracy, contact work or a complete transaction. The native contact
force reference remains unqualified. The AL and directional CCD experiments
remain disabled.
