#!/usr/bin/env python3
"""Build the production-owner coupons; keep all outputs outside the checkout."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
TEST = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path)
    ap.add_argument("--native-library", type=Path)
    ap.add_argument("--native-metallib", type=Path)
    args = ap.parse_args()
    if bool(args.native_library) != bool(args.native_metallib):
        ap.error("native library and native metallib must be supplied together")
    out = (args.build_dir or Path(tempfile.mkdtemp(prefix="numi-human-surface-attachment-"))).resolve()
    if out == ROOT or ROOT in out.parents:
        ap.error("build directory must be outside the source checkout")
    out.mkdir(parents=True, exist_ok=True)
    includes = ["-I", str(ROOT / "include"), "-I", str(ROOT / "matter/include"),
                "-I", str(ROOT / "matter/src")]
    stages = []
    def run(name, command):
        started = time.monotonic()
        result = subprocess.run([str(x) for x in command], cwd=ROOT,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / (name + ".log")).write_text(result.stdout)
        stages.append({"name": name, "command": [str(x) for x in command],
                       "returncode": result.returncode, "seconds": time.monotonic() - started,
                       "output": result.stdout})
        print(name, "PASS" if result.returncode == 0 else "FAIL", flush=True)
        if result.returncode:
            raise RuntimeError(name + " failed; see " + str(out / (name + ".log")))
    cpp = ["xcrun", "--sdk", "macosx", "clang++", "-std=c++23", "-O3",
           "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-DNDEBUG", "-arch", "arm64"]
    sources = ["matter/include/numi/matter/numi_human_shared.h",
               "matter/include/numi/matter/numi_human.hpp", "matter/src/numi_human.mm",
               "matter/src/fem_surface_attachment_validation.hpp", "matter/src/metal/numi_human.metalinc"]
    sources += [str(p.relative_to(ROOT)) for p in sorted(TEST.iterdir())
                if p.suffix in (".cpp", ".mm", ".metal", ".py", ".md")]
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"status": "FAIL", "scope": "owner source validation, Metal coupons and optional isolated Runtime regression; no full Human qualification",
               "source_sha256": {p: sha(ROOT / p) for p in sources}, "stages": stages}
    try:
        run("owner-syntax", cpp + ["-fPIC", "-fobjc-arc", *includes, "-fsyntax-only", ROOT / "matter/src/numi_human.mm"])
        run("cpu-build", cpp + includes + [TEST / "fem_surface_attachment_math_test.cpp", "-o", out / "math-test"])
        run("cpu-run", [out / "math-test"])
        run("gpu-build", cpp + ["-fobjc-arc", *includes, TEST / "fem_surface_attachment_coupon.mm",
                                "-framework", "Foundation", "-framework", "Metal", "-o", out / "gpu-test"])
        aggregate = (ROOT / "matter/src/matter_kernels.metal").read_text()
        coupon = (TEST / "surface_attachment_coupon.metal").read_text().replace(
            "#include \"matter_kernels.metal\"\n", "", 1)
        combined = out / "surface_attachment_all.metal"
        combined.write_text(aggregate + "\n" + coupon)
        run("metal-build", ["xcrun", "--sdk", "macosx", "metal", "-c", "-std=metal4.0", "-O3", "-fno-fast-math",
                            *includes, combined, "-o", out / "coupon.air"])
        run("metallib-build", ["xcrun", "--sdk", "macosx", "metallib", out / "coupon.air", "-o", out / "coupon.metallib"])
        run("gpu-run", [out / "gpu-test", out / "coupon.metallib"])
        if args.native_library:
            lib = args.native_library.resolve()
            run("native-build", cpp + ["-fobjc-arc", *includes, TEST / "native_initialize_test.mm", lib,
                                      "-Wl,-rpath," + str(lib.parent), "-framework", "Foundation",
                                      "-framework", "Metal", "-o", out / "native-test"])
            run("native-run", [out / "native-test", ROOT / "matter/materials/silicone.nmatter", args.native_metallib.resolve()])
            run("native-active-contact-run", [out / "native-test", ROOT / "matter/materials/silicone.nmatter", args.native_metallib.resolve(), "active-contact"])
            receipt["native_library_sha256"] = sha(lib)
            receipt["native_metallib_sha256"] = sha(args.native_metallib.resolve())
        receipt["status"] = "PASS"
    finally:
        receipt["binary_sha256"] = {p.name: sha(p) for p in (out / "math-test", out / "gpu-test", out / "coupon.metallib", out / "native-test") if p.is_file()}
        (out / "run-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": receipt["status"], "receipt": str(out / "run-receipt.json")}))

if __name__ == "__main__":
    main()
