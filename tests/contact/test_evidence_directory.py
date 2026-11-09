#!/usr/bin/env python3
"""Exercise supervisor routing with a failing fake producer, never a GPU run."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import run_be_compressed_ladder_008 as supervisor

with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    chosen = base / "chosen"
    default = base / "default"
    chosen.mkdir()
    default.mkdir()
    executable = base / "producer"
    executable.write_text('#!/usr/bin/env python3\nprint(\'{"traces":[]}\')\n')
    executable.chmod(0o755)
    shader = base / "shader"
    reference = base / "reference"
    shader.write_text("not a GPU shader")
    reference.write_text('{}')
    supervisor.EXE = executable
    supervisor.SHADER = shader
    supervisor.REFERENCE = reference
    supervisor.EVIDENCE = default
    supervisor.LADDER = [("test", "0.1", 1)]
    supervisor.LIMIT_FREE = 0
    sys.argv = ["supervisor", "--evidence-dir", str(chosen)]
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            supervisor.main()
        except SystemExit as error:
            assert "first failure" in str(error)
        else:
            raise AssertionError("invalid producer passed the accuracy gate")
    assert not list(default.iterdir())
    record = json.loads((chosen / "be-precision-008-n2-htest-execution.json").read_text())
    assert not record["all_gates_pass"]
    assert (chosen / "be-precision-008-n2-htest.json.gz").is_file()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            supervisor.main()
        except SystemExit as error:
            assert "refusing to overwrite" in str(error)
        else:
            raise AssertionError("existing evidence was overwritten")
print("custom evidence location, failure record and no-overwrite guard passed")
