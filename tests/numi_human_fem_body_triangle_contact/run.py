#!/usr/bin/env python3
"""Build and run the bounded Human FEM/bone-triangle owner regression on a Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(roots):
    seen = {}
    for root in roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in (
                    ".h", ".hpp", ".cpp", ".mm", ".metal", ".metalinc", ".py", ".md"
                ):
                    seen[str(path.resolve())] = path
    return sorted(seen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True,
                        help="New private output directory; never overwrites a prior run")
    parser.add_argument("--source-overlay", type=Path,
                        help="Optional isolated source overlay for reproducibility checks")
    parser.add_argument("--base-root", type=Path,
                        help="Base checkout used with --source-overlay; defaults to this checkout")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=False)
    test = Path(__file__).resolve().parent
    detected_root = test.parents[1]
    repo = args.base_root.resolve() if args.base_root else detected_root
    primary = args.source_overlay.resolve() if args.source_overlay else repo
    attachment_test = primary / "tests/numi_human_fem_surface_attachment"
    include = ["-I", str(primary / "matter/include"), "-I", str(repo / "matter/include"),
               "-I", str(repo / "include")]
    source_include = include + ["-I", str(primary / "matter/src"),
                                "-I", str(repo / "matter/src")]
    coupon_include = source_include + ["-I", str(test)]
    host = ["clang++", "-std=c++23", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-Werror"]
    cpu = build / "math-test"
    gpu = build / "gpu-test"
    air = build / "contact.air"
    library = build / "contact.metallib"
    owner_source = primary / "matter/src/numi_human.mm"
    coupon_metal = primary / "tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_coupon.metal"
    coupon_mm = primary / "tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_coupon.mm"
    math_test = primary / "tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_math_test.cpp"
    commands = [
        ("cpu-build", host + include + [str(math_test), "-o", str(cpu)]),
        ("cpu-run", [str(cpu)]),
        ("owner-syntax", host + ["-DNDEBUG", "-arch", "arm64", "-fPIC", "-fobjc-arc",
         '-DNUMI_MATTER_DEFAULT_METALLIB="' + str(library) + '"'] + source_include +
         ["-x", "objective-c++", "-fsyntax-only", str(owner_source)]),
        ("surface-attachment-coupon-host-syntax", host + ["-fobjc-arc"] + source_include +
         ["-fsyntax-only", str(attachment_test / "fem_surface_attachment_coupon.mm")]),
        ("metal-build", ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-fno-fast-math"] +
         source_include + ["-c", str(coupon_metal), "-o", str(air)]),
        ("metallib", ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)]),
        ("gpu-build", host + ["-fobjc-arc"] + coupon_include +
         [str(coupon_mm), "-framework", "Foundation", "-framework", "Metal", "-o", str(gpu)]),
        ("gpu-run", [str(gpu), str(library)]),
    ]
    roots = [primary / "matter/include", repo / "matter/include", repo / "include",
             primary / "matter/src", repo / "matter/src",
             primary / "tests/numi_human_fem_body_triangle_contact", attachment_test]
    before = {str(p): sha(p) for p in source_files(roots)}
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    receipt = {"scope": "Prepared bounded body-local bone-triangle banks and legacy fallback; not global nearest-face or anatomical-coverage qualification",
               "source_revision": revision, "source_sha256": before, "stages": [], "status": "running"}
    output = build / "result.json"
    try:
        for name, argv in commands:
            started = time.monotonic()
            completed = subprocess.run(argv, cwd=repo, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=600)
            (build / (name + ".log")).write_text(completed.stdout)
            receipt["stages"].append({"name": name, "argv": argv, "returncode": completed.returncode,
                                      "wall_seconds": time.monotonic()-started,
                                      "output": completed.stdout[-4000:]})
            output.write_text(json.dumps(receipt, indent=2)+"\n")
            if completed.returncode:
                raise RuntimeError(name + " failed; see " + str(build / (name + ".log")))
        if any(sha(Path(p)) != h for p, h in before.items()):
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
