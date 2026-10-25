#!/usr/bin/env python3
"""Reproduce the 039 identity parent-mass formula from frozen Numi inputs.

Uses only Python stdlib so the audit can run on the clean Numi host. This is a
formula reconstruction, not a per-element runtime export or admission receipt.
"""
from __future__ import annotations
import ast, csv, hashlib, json, math, pathlib, struct, zipfile

HERE = pathlib.Path(__file__).resolve().parent
E = HERE.parent
FIT = E / "bone-clearance-fit-040-shared-columns/attempt-004/fit-input.npz"
REPAIR = E / "connected-volume-continuation-039/iteration-005/repair-input.npz"
PRE = E / "connected-volume-continuation-039/iteration-005/native-query/source-preflight"
NODE_CSV = PRE / "initial-skin-fem-nodes.csv"
LOG = PRE / "stdout.log"
COMPILER = pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/src/compiler_impl.cpp")
HELPER = HERE / "apps/NumiHumanParentReferenceMassTransfer.hpp"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path: pathlib.Path, wanted: set[str]) -> dict[str, list[list[float | int]]]:
    result = {}
    fmts = {"<f4":"f", "<f8":"d", "<i4":"i", "<i8":"q", "|i1":"b"}
    with zipfile.ZipFile(path) as archive:
        for name in wanted:
            raw_npy = archive.read(name + ".npy")
            version = raw_npy[6]
            offset = 10 if version == 1 else 12
            header_size = struct.unpack("<H" if version == 1 else "<I", raw_npy[8:offset])[0]
            header = ast.literal_eval(raw_npy[offset:offset+header_size].decode("latin1").strip())
            dtype = header["descr"]
            if dtype not in fmts:
                raise ValueError(f"unsupported dtype {dtype} for {name}")
            width = math.prod(header["shape"][1:]) if len(header["shape"]) > 1 else 1
            values = [v[0] for v in struct.iter_unpack("<" + fmts[dtype], raw_npy[offset+header_size:])]
            if len(values) % width:
                raise ValueError(f"bad shape for {name}")
            result[name] = [values[i:i+width] for i in range(0, len(values), width)]
    return result


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def tet_det(tet, points) -> float:
    p0,p1,p2,p3 = (points[int(i)] for i in tet)
    a = [p1[i]-p0[i] for i in range(3)]
    b = [p2[i]-p0[i] for i in range(3)]
    c = [p3[i]-p0[i] for i in range(3)]
    return (a[0]*(b[1]*c[2]-b[2]*c[1])
          - a[1]*(b[0]*c[2]-b[2]*c[0])
          + a[2]*(b[0]*c[1]-b[1]*c[0]))


def parse_log_fields(line: str) -> dict[str, str]:
    fields = {}
    for token in line.split():
        if "=" in token:
            key,value=token.split("=",1); fields[key]=value
    return fields


def main() -> None:
    fit = read_npz(FIT, {"reference_points_m","tetrahedra","material_ids"})
    repair = read_npz(REPAIR, {"reference_points_m","tetrahedra","material_ids","global_nodes"})
    if fit["tetrahedra"] != repair["tetrahedra"] or fit["material_ids"] != repair["material_ids"]:
        raise ValueError("039 and fit parent topology/material rows differ")
    if fit["reference_points_m"] != repair["reference_points_m"]:
        raise ValueError("039 and fit parent reference-point arrays differ")
    logs = LOG.read_text().splitlines()
    preparation = parse_log_fields(next(x for x in logs if x.startswith("resting_skin_fat_whole_body_preparation=")))
    identity = parse_log_fields(next(x for x in logs if x.startswith("whole_body_skin_fat_unpaired_identity_remap=")))
    # Exported material_ids 1 and 2 are the skin/fat role partition in this input.
    mapping = {1:"skin", 2:"fat"}
    source_volume = {1:float(preparation["skin_volume_m3"]), 2:float(preparation["fat_volume_m3"])}
    source_mass = {1:float(preparation["skin_mass_debit_kg"]), 2:float(preparation["fat_mass_debit_kg"])}
    source_density = {m:source_mass[m]/source_volume[m] for m in mapping}
    density_bulk_f32 = {m:f32(source_density[m]) for m in mapping}
    lower = {1:900.0, 2:850.0}; upper = {1:1100.0, 2:950.0}
    points_f32 = [[f32(float(x)) for x in row] for row in fit["reference_points_m"]]
    reference_volume = {m:0.0 for m in mapping}; compiled_volume = {m:0.0 for m in mapping}
    compiled_mass = {m:0.0 for m in mapping}; cell_volumes=[]; cell_masses=[]
    for tet,mat_row in zip(fit["tetrahedra"],fit["material_ids"]):
        m=int(mat_row[0]); det=tet_det(tet,points_f32)
        if m not in mapping or not math.isfinite(det) or det <= 0.0:
            raise ValueError("invalid material or nonpositive compiled-F32 reference tet")
        f64v=abs(tet_det(tet,fit["reference_points_m"]))/6.0
        f32v=f32(det/6.0)
        if not (f64v>0 and f32v>0): raise ValueError("nonpositive parent measure/compiled volume")
        reference_volume[m]+=f64v; compiled_volume[m]+=f32v
        mass=density_bulk_f32[m]*f32v
        compiled_mass[m]+=mass; cell_volumes.append(f32v); cell_masses.append(mass)
    node_mass_pred=[0.0]*len(points_f32)
    for tet,mass in zip(fit["tetrahedra"],cell_masses):
        for node in tet: node_mass_pred[int(node)]+=mass*.25
    csv_mass={}
    with NODE_CSV.open(newline="") as stream:
        for row in csv.DictReader(stream): csv_mass[int(row["global_node"])]=float(row["mass_kg"])
    gids=[int(row[0]) for row in repair["global_nodes"]]
    if len(gids)!=len(points_f32) or set(gids)!=set(csv_mass):
        raise ValueError("node identity domain does not match compiled CSV")
    residual=[node_mass_pred[i]-csv_mass[gids[i]] for i in range(len(gids))]
    f64_total=sum(reference_volume.values()); f32_total=sum(compiled_volume.values())
    report={
      "schema":"numi.human.actual-039-identity-parent-mass-formula-receipt.v1",
      "host":"numi.local via SSH alias numi; all calculation executed remotely",
      "status":"identity-formula-reconstruction-only; donor attribution and exact per-cell runtime export are unavailable",
      "inputs":{"fit_input_npz":{"path":str(FIT),"sha256":sha(FIT)},"repair_input_npz":{"path":str(REPAIR),"sha256":sha(REPAIR)},"compiled_node_csv":{"path":str(NODE_CSV),"sha256":sha(NODE_CSV)},"source_preflight_log":{"path":str(LOG),"sha256":sha(LOG)},"compiler_impl":{"path":str(COMPILER),"sha256":sha(COMPILER)},"owner_helper":{"path":str(HELPER),"sha256":sha(HELPER)}},
      "identity_receipt":{"nodes":int(identity["nodes"]),"tets":int(identity["tets"]),"f64_reference_volume_m3":float(identity["reference_m3"]),"source_mass_debit_kg":float(identity["mass_kg"]),"receipt_sha256":identity["exact_identity_receipt_sha256"],"source_world_mutated":identity["source_world_mutated"]},
      "parent_child_identity":{"topology_and_material_rows_byte_equal":True,"pullback_fraction_each_child_to_parent":1.0,"f64_reference_volume_m3_by_material":{mapping[m]:reference_volume[m] for m in mapping},"compiled_f32_parent_volume_m3_by_material":{mapping[m]:compiled_volume[m] for m in mapping},"compiler_f32_bulk_density_kg_m3_by_material":{mapping[m]:density_bulk_f32[m] for m in mapping},"parent_mass_formula":"double(F32 bulk.x) * double(F32 compiler rest volume); child identity mass uses F64 pullback fraction=1","parent_compiler_mass_kg_by_material":{mapping[m]:compiled_mass[m] for m in mapping},"child_emitted_f32_volume_equals_parent_compiled_f32_volume":True,"child_implied_density_kg_m3_by_material":{mapping[m]:compiled_mass[m]/compiled_volume[m] for m in mapping},"density_bounds_kg_m3_by_material":{mapping[m]:[lower[m],upper[m]] for m in mapping},"cell_volume_sha256_le_f32":hashlib.sha256(b"".join(struct.pack("<f",v) for v in cell_volumes)).hexdigest(),"cell_mass_sha256_le_f64":hashlib.sha256(b"".join(struct.pack("<d",v) for v in cell_masses)).hexdigest()},
      "precision_crosscheck":{"formula_parent_mass_total_kg":sum(compiled_mass.values()),"source_identity_mass_total_kg":float(identity["mass_kg"]),"formula_minus_source_identity_mass_kg":sum(compiled_mass.values())-float(identity["mass_kg"]),"compiled_node_csv_total_kg":sum(csv_mass.values()),"formula_nodal_lump_max_abs_difference_kg":max(map(abs,residual)),"formula_nodal_lump_rms_difference_kg":(sum(x*x for x in residual)/len(residual))**.5,"direct_per_tet_runtime_mass_export_present":False,"interpretation":"Per-tet masses are reconstructed from the pinned compiler formula, source material density inputs, and F32 reference coordinates. The available runtime CSV is nodal only and does not match this reconstructed nodal lumping at every node, so it is not accepted as direct per-cell compiler verification."},
      "limits":["No donor/origin IDs are present in this identity input; only material-level closure is reported.","The identity case verifies row identity and algebraic F64 pullback fractions, not source ancestry for changed topology.","No new refined child candidate is mass-qualified by this receipt.","This does not qualify geometry, anatomy, native evolution, or physical admission."]
    }
    out=HERE/"actual-039-identity-parent-mass-receipt.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    report["audit_script"]={"path":str(pathlib.Path(__file__).resolve()),"sha256":sha(pathlib.Path(__file__).resolve())}
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(out),"script_sha256":sha(pathlib.Path(__file__).resolve()),"sha256":sha(out),"reference_volume":f64_total,"compiled_f32_volume":f32_total,"parent_mass":sum(compiled_mass.values()),"identity_mass":float(identity["mass_kg"]),"csv_mass":sum(csv_mass.values()),"max_node_residual":max(map(abs,residual))},sort_keys=True))

if __name__ == "__main__": main()
