import hashlib
import json
import os
import pathlib
import subprocess
import time

base = pathlib.Path("/Users/n/numi-human-separation-20261018/C/wholebody-integration-295")
test_dir = base / "tests/whole-body-mass-reference-001"
evidence = test_dir / "evidence-001"
evidence.mkdir(parents=True, exist_ok=True)
source = base / "tests/whole_body_mass_reference.cpp"
header = base / "source/apps/NumiHumanWholeBodyMassReference.hpp"
library_dir = pathlib.Path("/Users/n/numi-human-separation-20261018/C/contact-legacy-plane-bounded-recovery-279/build-guarded-root-001")
library = library_dir / "libmetalrobo.dylib"
binary = evidence / "whole_body_mass_reference"
argv = [
    "/usr/bin/clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
    f"-I{base / 'source/apps'}",
    "-I/Users/n/numi-lab-native-replay-publish-20261017/include",
    "-I/Users/n/numi-lab-native-replay-publish-20261017/matter/include",
    f"-L{library_dir}", f"-Wl,-rpath,{library_dir}", str(source),
    "-lmetalrobo", "-framework", "Foundation", "-o", str(binary),
]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
t0 = time.monotonic()
cc = subprocess.run(argv, capture_output=True, text=True)
compile_seconds = time.monotonic() - t0
(evidence / "compile.stdout").write_text(cc.stdout)
(evidence / "compile.stderr").write_text(cc.stderr)
if cc.returncode != 0:
    receipt = {"compile_argv": argv, "compile_exit_code": cc.returncode,
               "compile_elapsed_seconds": compile_seconds, "source_sha256": sha(source),
               "header_sha256": sha(header), "library_sha256": sha(library)}
    (evidence / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(cc.returncode)
env = os.environ.copy()
env["DYLD_LIBRARY_PATH"] = str(library_dir)
t1 = time.monotonic()
run = subprocess.run([str(binary)], capture_output=True, text=True, env=env)
run_seconds = time.monotonic() - t1
(evidence / "stdout.log").write_text(run.stdout)
(evidence / "stderr.log").write_text(run.stderr)
receipt = {
    "kind": "CPU-only whole-body mass reference regression; no native simulation",
    "compile_argv": argv, "compile_exit_code": cc.returncode,
    "compile_elapsed_seconds": compile_seconds,
    "compile_stdout_sha256": sha(evidence / "compile.stdout"),
    "compile_stderr_sha256": sha(evidence / "compile.stderr"),
    "source_sha256": sha(source), "header_sha256": sha(header),
    "library_path": str(library), "library_sha256": sha(library),
    "binary_path": str(binary), "binary_sha256": sha(binary),
    "run_argv": [str(binary)], "run_environment": {"DYLD_LIBRARY_PATH": str(library_dir)},
    "exit_code": run.returncode, "elapsed_seconds": run_seconds,
    "stdout_sha256": sha(evidence / "stdout.log"),
    "stderr_sha256": sha(evidence / "stderr.log"),
}
(evidence / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(run.stdout, end="")
print(json.dumps({"exit_code": run.returncode, "result": str(evidence / "result.json")}))
raise SystemExit(run.returncode)
