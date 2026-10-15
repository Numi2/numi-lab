# Exact failed contact candidate, native291

This package retains the complete 10.8 MB source088 coupled Newton12 candidate,
140 arrays, and all 18,336 reserved self-contact records and accepted/candidate
histories. It is a read-only failed-state diagnostic, not an accepted physical
checkpoint, a complete nonlinear trajectory, or a shipping-box restart.

The native291 strict JSON metadata and payload are unchanged. Gzip uses mtime0;
its round-trip bytes and SHA256 are verified. The validator resolves the payload
by its basename beside the manifest, so exact metadata can be restored anywhere.

Run `python3 restore.py /absolute/new/directory` from this package. The output
directory must not exist. Every retained package input and decompressed payload
is hash-checked, then all array extents/hashes and active-prefix coverage are
rechecked. Original native289's malformed nonfinite JSON remains separate.
