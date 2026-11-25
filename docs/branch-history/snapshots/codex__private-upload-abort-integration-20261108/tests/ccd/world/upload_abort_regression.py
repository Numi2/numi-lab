#!/usr/bin/env python3
"""Compare independent successful and pre-submit-rejected owner attempts."""
import subprocess
import sys
import tempfile
from pathlib import Path

exe, metallib = map(Path, sys.argv[1:])
with tempfile.TemporaryDirectory(prefix="numi-upload-abort-") as directory:
    outputs = []
    for arm in ("control", "retry"):
        output = Path(directory) / (arm + ".bin")
        process = subprocess.run(
            [str(exe), str(metallib), str(output), arm, "private"],
            capture_output=True, text=True, timeout=120,
        )
        if process.returncode:
            raise SystemExit(f"{arm} failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")
        outputs.append(output.read_bytes())
    if not outputs[0] or outputs[0] != outputs[1]:
        raise SystemExit("control and retry outputs differ")
print(f"control and retry exactly match ({len(outputs[0])} bytes)")
