#!/usr/bin/env python3
"""Compile the live C API assignment block into a CPU-only synthetic probe."""
import argparse
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--source", required=True, type=Path)
p.add_argument("--headers", required=True, type=Path)
p.add_argument("--libdir", required=True, type=Path)
a = p.parse_args()
source = a.source.resolve()
text = (source / "src/c_api.cpp").read_text()
anchor = text.index("static MRTaskRolloutHandle* createUnitreeG1TeacherRun(")
start = text.index("        if (config->override_interaction_student_authority != 0u)", anchor)
end = text.index("        auto handle = createCompiledRunTaskRollout(", start)
block = text[start:end]
if "<<<<<<<" in block or "compatiblePolicySource = manifest;" not in block:
    raise SystemExit("C API assignment block is incomplete or unresolved")
template = (source / "tests/selector_interaction_contract.cpp.in").read_text()
if template.count("@EXACT_C_API_BLOCK@") != 1:
    raise SystemExit("fixture insertion point invalid")
with tempfile.TemporaryDirectory(prefix="numi-selector-contract-") as temporary:
    root = Path(temporary)
    cpp = root / "fixture.cpp"
    exe = root / "fixture"
    cpp.write_text(template.replace("@EXACT_C_API_BLOCK@\n", block))
    subprocess.run(["clang++", "-std=c++20", "-O1", "-I" + str(a.headers.resolve()),
                    str(cpp), "-L" + str(a.libdir.resolve()), "-lmetalrobo",
                    "-Wl,-rpath," + str(a.libdir.resolve()), "-framework", "Metal",
                    "-framework", "Foundation", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
