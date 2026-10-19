import hashlib
import json
import os
import pathlib
import subprocess
import time

base = pathlib.Path("/Users/n/numi-human-separation-20261018/C/wholebody-integration-295")
run_dir = base / "tests/native-dump-stock-replay-001"
run_dir.mkdir(parents=True, exist_ok=True)
source = base / "tests/whole_body_stock_dump_replay.cpp"
header = base / "source/apps/NumiHumanWholeBodyMassReference.hpp"
input_path = base / "native-wholebody-006/tissue-mass-input.txt"
library_dir = pathlib.Path("/Users/n/numi-human-separation-20261018/C/tissue-mass-symmetric-301/build-runtime-001")
library = library_dir / "libmetalrobo.dylib"
binary = run_dir / "whole_body_stock_dump_replay"
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

started = time.monotonic()
cc = subprocess.run(argv, capture_output=True, text=True)
compile_seconds = time.monotonic() - started
(run_dir / "compile.stdout").write_text(cc.stdout)
(run_dir / "compile.stderr").write_text(cc.stderr)
if cc.returncode:
    result = {
        "kind": "CPU-only replay of retained native tissue mass input against the stock partition compiler",
        "compile_argv": argv,
        "compile_exit_code": cc.returncode,
        "compile_elapsed_seconds": compile_seconds,
        "source_sha256": sha(source),
        "header_sha256": sha(header),
        "input_sha256": sha(input_path),
        "library_path": str(library),
        "library_sha256": sha(library),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(cc.returncode)

env = os.environ.copy()
env["DYLD_LIBRARY_PATH"] = str(library_dir)
started = time.monotonic()
run = subprocess.run([str(binary), str(input_path)], capture_output=True, text=True, env=env)
elapsed = time.monotonic() - started
(run_dir / "stdout.log").write_text(run.stdout)
(run_dir / "stderr.log").write_text(run.stderr)
result = {
    "kind": "CPU-only replay of retained native tissue mass input against the stock partition compiler",
    "compile_argv": argv,
    "compile_exit_code": cc.returncode,
    "compile_elapsed_seconds": compile_seconds,
    "compile_stdout_sha256": sha(run_dir / "compile.stdout"),
    "compile_stderr_sha256": sha(run_dir / "compile.stderr"),
    "source_sha256": sha(source),
    "header_sha256": sha(header),
    "input_path": str(input_path),
    "input_sha256": sha(input_path),
    "library_path": str(library),
    "library_sha256": sha(library),
    "binary_path": str(binary),
    "binary_sha256": sha(binary),
    "run_argv": [str(binary), str(input_path)],
    "run_environment": {"DYLD_LIBRARY_PATH": str(library_dir)},
    "exit_code": run.returncode,
    "elapsed_seconds": elapsed,
    "stdout_sha256": sha(run_dir / "stdout.log"),
    "stderr_sha256": sha(run_dir / "stderr.log"),
}
(run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(run.stdout, end="")
print(json.dumps({"exit_code": run.returncode, "result": str(run_dir / "result.json")}))
raise SystemExit(run.returncode)
