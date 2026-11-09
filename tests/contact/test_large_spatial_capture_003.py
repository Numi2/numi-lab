"""Exercise the guarded gzip producer with a short, real retained trace."""
import contextlib
import gzip
import io
import json
from pathlib import Path
import tempfile

import run_large_spatial_continuation_003 as run
from large_spatial_reference import read_reference

fixture = run.e / "compact-reference-parity-n2-compact-006.json"
with tempfile.TemporaryDirectory() as d:
    run.e = Path(d)
    run.exe = run.e / "fake_reference"
    run.exe.write_text("#!/bin/sh\nexec cat " + str(fixture) + "\n")
    run.exe.chmod(0o755)
    run.expected_binary = run.digest(run.exe)
    run.free_limit = 0
    run.workers = lambda exclude=(): []
    with contextlib.redirect_stdout(io.StringIO()):
        out = run.run_reference(2, "0.000002", "smoke")
    data = read_reference(out, 2, .000002)
    meta = json.loads((run.e / "rk4-compact-large-003-smoke-execution.json").read_text())
    assert data["mesh_divisions"] == 2 and meta["returncode"] == 0
    assert meta["output_sha256"] == run.digest(out) and not meta["compression_errors"]
    try:
        run.run_reference(2, "0.000002", "smoke")
    except RuntimeError:
        pass
    else:
        raise AssertionError("existing evidence was overwritten")
print("guarded compressed reference capture passed")
