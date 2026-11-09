#!/usr/bin/env python3
"""Verify pinned historical reference sources share the fixture law.

This compares actual source blobs, not a claim that they prove material
calibration or an independent attestation of the historical build process.
"""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
records = [
    ("n32_serial", "b7447ca88", "tests/contact/fem_plane_fp64_reference.cpp",
     "01a8fb10156de380c13ec7df0761f21cb033041d18d22820b043bf47a3bd23db"),
    ("n64_serial", "564855bb1", "tests/contact/fem_plane_fp64_reference.cpp",
     "c4847b86546b85d3c63da91124fd5a5075fd7e3ee2bdc3203a1ff1fe8a6f3daf"),
    ("n64_parallel", "d00717ebe", "tests/contact/fem_plane_fp64_reference_parallel.cpp",
     "01b59053d091879fe4768c704f841d52022ade9f53c7cbf9722519af8184a55a"),
    ("n128_parallel", "968f49d1d", "tests/contact/fem_plane_fp64_reference_parallel.cpp",
     "019371ad99f9cfd0ff0a7c3b08e77d493f698f7b3f7d0ef528dd4695097edc1f"),
]

def method(text, marker):
    start = text.index(marker)
    brace = text.index("{", start)
    depth = 0
    for end in range(brace, len(text)):
        if text[end] == "{": depth += 1
        if text[end] == "}":
            depth -= 1
            if depth == 0: return text[start:end+1]
    raise ValueError("unterminated method: " + marker)

def sections(text):
    constructor = method(text, "explicit Model(int n)")
    # Parallel versions add an incidence index after generating the same mesh.
    if "  incidence.resize(count);" in constructor:
        constructor = constructor[:constructor.index("  incidence.resize(count);")].rstrip() + "\n }"
    return {
        "physical_parameters": text[text.index("double d=1e-4,"):text.index("double d=1e-4,")+len("double d=1e-4,k=1e16,mu=2000,lambda=5000;")],
        "geometry_mass_and_area": constructor,
        "pressure": method(text, "double pressure(double g)const"),
        "potential_and_gravity": method(text, "double energy(const State& s)const"),
    }

baseline = None
out = {}
for label, revision, path, expected in records:
    blob = subprocess.check_output(["git", "show", revision + ":" + path], cwd=root)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected: raise ValueError("source hash mismatch: " + label)
    source = blob.decode()
    selected = sections(source)
    if selected["physical_parameters"] != "double d=1e-4,k=1e16,mu=2000,lambda=5000;":
        raise ValueError("unexpected parameters: " + label)
    if baseline is None: baseline = selected
    if selected != baseline: raise ValueError("physical source differs: " + label)
    # The serial and parallel implementations assemble in different ways.
    # Verify the constitutive formula and retain the separate parity evidence.
    canonical = source.replace("model.mu", "mu").replace("model.lambda", "lambda")
    stress = "P.a[i][j]=mu*F.a[i][j]+(lambda*std::log(J)-mu)*invF.a[j][i];"
    if stress not in canonical: raise ValueError("constitutive expression differs: " + label)
    out[label] = {"revision":revision, "source_sha256":digest}
print(json.dumps({"inputs":out, "identical_source_sections":list(baseline),
                  "mu_Pa":2000, "lambda_Pa":5000, "density_kg_m3":1100,
                  "activation_width_m":1e-4, "area_coefficient_N_m5":1e16,
                  "limitation":"source comparison and retained binary hashes do not independently attest compilation"}, indent=2))
