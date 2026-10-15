# Failed contact candidate291, strict restore v2

The original native291 manifest and all10,787,124 payload bytes remain unchanged.
This version retains packagev1 and replaces its generic metadata admission with
reader-v3: NaN/Infinity metadata tokens reject, raw payload float words remain
unchanged, and only the manifest-adjacent payload is read. Thus restoration
cannot accidentally validate an original absolute file instead of restored bytes.

Run `python3 restore.py /absolute/new/directory` with Python and NumPy installed.
The directory must not exist. Input hashes, gzip round-trip, strict metadata,
all140 array hashes/extents, and full contact arena coverage are checked.
This is a failed Newton candidate, not an accepted restart checkpoint. No solver,
mesh/time, physical assembly, calibration or impact qualification is implied.
