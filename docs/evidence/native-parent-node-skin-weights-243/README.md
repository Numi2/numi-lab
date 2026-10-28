# Exact parent-node binding weights

The optional native source-preparation capture NUMI_HUMAN_SKIN_FAT_PARENT_WEIGHTS_CSV writes the existing parent skin/fat object's full 86-lane reference bindings in exact object-local parent_node_index order. Explicit O/I/D boundary maps determine those rows; the code does not infer them by modulo or conflate them with cell donor labels. Every node is covered once. Finite normalized weights and dominant binding/frame ownership must agree before output.

On SSH Numi the full native executable built, then the source-only command in checks.json wrote 7,233 rows. The maximum weight-sum error was 5.168687611245559e-08. All nine prior geometry, pose, parent mass, source mass and label outputs remained byte-identical. The new CSV is retained as deterministic gzip with its uncompressed SHA-256 in checks.json. Binding lane identities are pinned to the pose snapshot hash.

These are paired-column reference bindings copied from the retained physical parent, not measured deep-tissue biology. The optional capture neither changes mass nor adds a simulation owner. No runtime was created and zero physical steps ran. New layer geometry, conservative child mass transfer, native volume admission and motion qualification remain pending.
