# Existing tissue owner with remeshed source ancestry

The candidate layer builder accepts exact source slots, verified one-hot slots and source-face barycentric samples. It preserves original retained skin deltas and explicit per-node thickness bounds; density times volume determines mass. It constructs existing registered surfaces and paired layers and calls the existing FEM assembly owner. It adds no force or simulation update path.

The maintained test at tests/human_registered_skinfat_candidate_layers.cpp passes, including mixed ancestry, complete 86-body weights, skin/fat bonded assembly, mass closure and rejection cases. The manifest binds the clean staged header, compiler command and output. The separate real 081 fixture passes source reconstruction but is rejected by the existing layer owner for six inner-skin and 93 deep-fat crossings. That rejection is retained; the Human mesh remains unadmitted.

The exact executed wrapper, runner and immutable patch retain their original trailing blank lines so recorded hashes remain valid. Whitespace checks exclude only these three retained evidence artifacts.
