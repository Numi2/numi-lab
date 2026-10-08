# Final native-scene preflight 936, revision 15

Revision 15 is a fresh builder derived from v014; v014 and v009 remain unchanged. It adds a paired caller override for the respiration parameter file:

`--respiration-config PATH --respiration-config-sha SHA256`

Both arguments are required together. Omitting both retains the frozen 761 default file. The builder pins the selected file in its source inventory and owner asset map and passes it through the actual native `--resting-scene CIRCULATION RESPIRATION` argument pair. The comparison reader resolves the respiration path from that assembled owner argv, checks the assembly source hash, and requires the same file/hash in the run invocation. It does not assume the old 761 path.

The selected file remains subject to the existing native owner s area admission check. This wrapper changes no formula, model, area, or runtime. In this revision, both CPU assemblies use the exact current 761 bytes (one by default and one copied to a different path) to exercise default and path-override identity. The custom path fixture is not a changed-area or native-mechanics test.

The default 927-skin/924-anatomy CPU assembly generated 25 native assets, retained all 32 support rows, and reproduced the 907 root pose with zero translation delta. The same-bytes path-override assembly records the override file in the native argv, asset hashes, and source hashes. The v015 comparator also successfully reviewed the retained v014 native attempt3 run: 1,250 q0/COM8 observations compared across the 60 fields, declared endpoint/window-extrema values exactly matched the 931 reference, and all eight accepted body-pose captures passed the existing checks. This is a comparator identity regression and a 20-second viewer regression only.

Twenty-seven focused tests pass and both Python files compile. Tests cover paired override arguments, hash mismatch rejection, comparator path/hash binding, and inclusion of the selected configuration among critical invocation assets. No native/GPU execution or study registration was performed. The final corrected lung candidate is not yet available; after it is, supply its owner-computed respiration file and SHA to this builder. The 924 fixture and these 20-second checks do not qualify updated anatomy, a new diaphragm area, 310-second endurance, or physiology.
