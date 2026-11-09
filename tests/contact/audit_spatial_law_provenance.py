#!/usr/bin/env python3
"""Audit retained source identity and visible physical-law invariants.

This is a source/provenance audit, not proof that a historical executable was
used when its execution record did not retain a binary hash.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "docs/evidence/midpoint-first-failure-20261109"
OLD = ROOT / "docs/evidence/midpoint-spatial-cost-dynamic-20261108"
CASES = [
    ("n32_serial", "b7447ca88", "tests/contact/fem_plane_fp64_reference.cpp", OLD / "N32_INPUTS.sha256", OLD / "rk4-n32-h025-execution.json"),
    ("n64_serial", "564855bb1", "tests/contact/fem_plane_fp64_reference.cpp", E / "n64-inputs.sha256", E / "rk4-n64-h050-execution.json"),
    ("n64_parallel", "d00717ebe", "tests/contact/fem_plane_fp64_reference_parallel.cpp", E / "fp64-parallel-inputs.sha256", E / "rk4-n64-parallel-h025-execution.json"),
    ("n128_parallel", "968f49d1d", "tests/contact/fem_plane_fp64_reference_parallel.cpp", E / "n128-final-inputs.sha256", E / "rk4-n128-retry1-h050-execution.json"),
]

def body(source, signature):
    start = source.index(signature)
    begin = source.index("{", start)
    depth = 0
    for i in range(begin, len(source)):
        depth += (source[i] == "{") - (source[i] == "}")
        if depth == 0:
            return re.sub(r"\s+", "", source[begin:i + 1])
    raise ValueError("unclosed function " + signature)

def audit():
    result = []
    pressure = []
    energy = []
    for label, revision, path, manifest, execution in CASES:
        source = subprocess.check_output(["git", "show", revision + ":" + path], cwd=ROOT)
        digest = hashlib.sha256(source).hexdigest()
        manifest_sources = [line.split()[0] for line in manifest.read_text().splitlines()
                            if line.split()[-1] == path]
        if manifest_sources != [digest]:
            raise ValueError(label + ": source hash does not match retained manifest")
        text = source.decode()
        compact = re.sub(r"\s+", "", text)
        checks = {
            "law_and_material_parameters": "d=1e-4,k=1e16,mu=2000,lambda=5000;",
            "geometry_and_initial_gap": "initial.x[id(x,y,z)]={.02*x/n,.02*y/n,.0002+.02*z/n};",
            "density": "mass[i]+=1100*e.volume/4;",
            "reference_area_weight": "share=.02*.02/(n*n*6.);",
            "reference_area_triangles": "for(inti:{0,1,3,0,3,2})area[v[i]]+=share;",
            "gravity": "r.v[i].z-=9.81;",
            "normal_force_area_scaling": "doublefz=area[i]*pressure(s.x[i].z);",
        }
        for name, snippet in checks.items():
            if snippet not in compact:
                raise ValueError(label + ": missing invariant " + name)
        pressure.append(body(text, "double pressure(double g)const"))
        energy.append(body(text, "double energy(const State& s)const"))
        record = json.loads(execution.read_text())
        binary_hash = record.get("binary_sha256")
        manifest_binaries = [line.split()[0] for line in manifest.read_text().splitlines()
                             if Path(line.split()[-1]).name.startswith("numi_contact_fem_fp64_reference")]
        if binary_hash is not None and manifest_binaries != [binary_hash]:
            raise ValueError(label + ": execution binary does not match retained manifest")
        result.append({"name": label, "revision": revision, "source_path": path,
                       "source_sha256": digest, "manifest": str(manifest.relative_to(ROOT)),
                       "execution": str(execution.relative_to(ROOT)),
                       "execution_record_contains_binary_sha256": isinstance(binary_hash, str),
                       "execution_record_binary_sha256": binary_hash})
    if len(set(pressure)) != 1 or len(set(energy)) != 1:
        raise ValueError("pressure or energy implementation changed across retained sources")
    return {"schema": "numi.spatial-law-source-audit.v1", "cases": result,
            "parameters": {"activation_width_m": 1e-4, "contact_coefficient": 1e16,
                           "shear_modulus_Pa": 2000, "lame_lambda_Pa": 5000,
                           "density_kg_per_m3": 1100, "cube_side_m": .02,
                           "initial_gap_m": .0002, "gravity_m_per_s2": -9.81,
                           "reference_area_m2": .0004},
            "source_checks_pass": True,
            "scope": "Hash-matched recorded source revisions contain matching pressure and energy functions and the listed input expressions. Historical execution records without binary hashes do not independently bind the executable used. This audit is not a simulation or a spatial-convergence claim."}

if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
