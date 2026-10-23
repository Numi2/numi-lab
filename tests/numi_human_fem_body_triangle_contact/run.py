#!/usr/bin/env python3
"""Build and run the bounded Human FEM/bone contact regression on a Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True,
                        help="New private output directory; never overwrites a prior run")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=False)
    test = Path(__file__).resolve().parent
    repo = test.parents[1]
    include = ["-I", str(repo / "matter/include"), "-I", str(repo / "include")]
    host = ["clang++", "-std=c++23", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
    cpu = build / "math-test"
    gpu = build / "gpu-test"
    air = build / "contact.air"
    library = build / "contact.metallib"
    commands = [
        ("cpu-build", host + include + [str(test / "fem_body_triangle_contact_math_test.cpp"), "-o", str(cpu)]),
        ("cpu-run", [str(cpu)]),
        ("metal-build", ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-fno-fast-math"] + include +
         ["-I", str(repo / "matter/src"), "-c", str(test / "fem_body_triangle_contact_coupon.metal"), "-o", str(air)]),
        ("metallib", ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)]),
        ("gpu-build", host + ["-fobjc-arc"] + include + [str(test / "fem_body_triangle_contact_coupon.mm"),
         "-framework", "Foundation", "-framework", "Metal", "-o", str(gpu)]),
        ("gpu-run", [str(gpu), str(library)]),
        ("owner-syntax", host + ["-DNDEBUG", "-arch", "arm64", "-fPIC", "-fobjc-arc",
         '-DNUMI_MATTER_DEFAULT_METALLIB="' + str(library) + '"'] + include +
         ["-x", "objective-c++", "-fsyntax-only", str(repo / "matter/src/numi_human.mm")]),
    ]
    paths = sorted(p for root in (repo / "matter/include", repo / "include", repo / "matter/src", test)
                   for p in root.rglob("*") if p.is_file() and p.suffix in (".h", ".hpp", ".cpp", ".mm", ".metal", ".metalinc", ".py"))
    before = {str(p.relative_to(repo)): sha(p) for p in paths}
    receipt = {"scope": "Prepared single-triangle contact and existing owner-kernel regression; no full-scene qualification",
               "source_sha256": before, "stages": [], "status": "running"}
    output = build / "result.json"
    try:
        for name, argv in commands:
            started = time.monotonic()
            completed = subprocess.run(argv, cwd=repo, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=600)
            (build / (name + ".log")).write_text(completed.stdout)
            receipt["stages"].append({"name": name, "argv": argv, "returncode": completed.returncode,
                                      "wall_seconds": time.monotonic()-started})
            output.write_text(json.dumps(receipt, indent=2)+"\n")
            if completed.returncode:
                raise RuntimeError(name + " failed; see " + str(build / (name + ".log")))
        if any(sha(repo / p) != h for p, h in before.items()):
            raise RuntimeError("source changed during qualification")
        receipt["binary_sha256"] = {p.name: sha(p) for p in (cpu, gpu, library)}
        receipt["status"] = "PASS"
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = str(error)
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"status": receipt["status"], "result": str(output)}))


if __name__ == "__main__":
    main()
