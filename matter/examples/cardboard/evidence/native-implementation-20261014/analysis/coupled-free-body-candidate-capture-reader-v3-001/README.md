# Coupled candidate capture reader v3

This version preserves the v2 strict-JSON behavior and changes payload resolution for relocatable evidence packages. It always opens the payload basename next to the manifest. The manifest's recorded path is retained as provenance but is never used as a fallback, so an extant original capture cannot shadow copied/restored bytes.

Run the portable checks with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v
```

Inspect a capture with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 coupled_capture_reader.py inspect path/to/candidate-capture.coupled.json
```

The reader checks strict JSON constants, payload and per-array hashes, complete byte coverage, bounds, and record layouts. Capture validation does not qualify solver convergence or turn a raw candidate into an accepted checkpoint.
