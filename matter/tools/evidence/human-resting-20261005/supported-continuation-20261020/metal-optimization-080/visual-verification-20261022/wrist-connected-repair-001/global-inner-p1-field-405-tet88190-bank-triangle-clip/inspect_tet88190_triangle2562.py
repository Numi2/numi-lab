#!/usr/bin/env python3
import hashlib, importlib.util, json, pathlib
import numpy as np

BASE = pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = BASE / "global-inner-p1-field-405-tet88190-bank-triangle-clip"
OUT.mkdir(parents=True, exist_ok=True)
CAND403 = BASE / "global-inner-p1-edge-support-403/build-002/candidate.npz"
CAND398 = BASE / "global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz"
BANKS = BASE / "registered-surrounding-tissue-native013-191/registered-tissue-banks.npz"
INV = BASE / "registered-surrounding-tissue-native013-191/inventory.json"
HELPER = BASE / "robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py"
EXPECTED = {
    str(CAND403): "9c8c23c8229c46494d19f6fc12d6131667b901864dfab21cd1ef7de6a2fc6744",
    str(CAND398): "8abe15f941dbc01659aee693a07e3edf66f32617c784cf0334410ca11a357890",
    str(BANKS): "39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792",
    str(INV): "b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7",
    str(HELPER): "e1b5a246e531a16d8d2438b8569d31a27c9a10712f0de063084bed87db58c436",
}
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
for p,want in EXPECTED.items():
    got=sha(p)
    if got != want: raise RuntimeError(f"input hash mismatch {p}: {got} != {want}")
spec=importlib.util.spec_from_file_location("p1clip302",HELPER)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
clip=mod.triangle_tet_barycentric_polygon

z3=np.load(CAND403); z8=np.load(CAND398); banks=np.load(BANKS); inventory=json.load(open(INV))
T3=z3["tetrahedra"]; X3=z3["points_world_m"].astype(np.float64)
parent398=z3["parent_398_tet_row"]
T8=z8["tetrahedra"]; X8=z8["points_world_m"].astype(np.float64)
parent_row=88190
parent_nodes=T8[parent_row]
expected_parent_nodes=np.array([9451,9471,9473,27269],dtype=np.int64)
if not np.array_equal(parent_nodes,expected_parent_nodes):
    raise RuntimeError(f"398 parent row mismatch: {parent_nodes.tolist()}")
desc=np.flatnonzero(parent398==parent_row)
expected_desc=np.array([222172,222192,222193],dtype=np.int64)
if not np.array_equal(desc,expected_desc):
    raise RuntimeError(f"403 descendants mismatch: {desc.tolist()}")
outer=set(map(int,z3["physical_outer_node_ids"]))
def tetvol(x):
    return abs(float(np.linalg.det((x[1:]-x[0]).T)))/6.0
parent_vol=tetvol(X8[parent_nodes])
child_vols=[tetvol(X3[T3[int(r)]]) for r in desc]
volratio=sum(child_vols)/parent_vol
if not (np.isfinite(volratio) and abs(volratio-1.0)<5e-5):
    raise RuntimeError(f"descendant volume mismatch: ratio={volratio}")
def poly_area(P,C):
    if len(C)<3: return 0.0
    q=np.asarray(C,dtype=np.float64)@np.asarray(P,dtype=np.float64)
    s=np.zeros(3)
    for i in range(1,len(q)-1): s += np.cross(q[i]-q[0],q[i+1]-q[0])
    return float(0.5*np.linalg.norm(s))
def cap_rows(T, row, B):
    ids=T[int(row)]
    free=np.array([int(n) not in outer for n in ids],dtype=bool)
    caps=np.asarray(B,dtype=np.float64)[:,free].sum(axis=1) if np.any(free) else np.zeros(len(B))
    return ids.tolist(),int(np.count_nonzero(free)),caps
records=[]
for idx,meta in enumerate(inventory["banks"]):
    if int(meta["stable_id"]) != 5: continue
    key=meta["key"]
    F=np.asarray(banks[f"{key}_faces"],dtype=np.int64)
    V=np.asarray(banks[f"{key}_vertices"],dtype=np.float64)
    face_row=2562
    if face_row>=len(F): raise RuntimeError(f"{key}: face row {face_row} unavailable")
    tri_ids=F[face_row]
    tri=V[tri_ids]
    Bp,Cp=clip(X8[parent_nodes],tri)
    parent_area=poly_area(tri,Cp)
    child_rows=[]
    total_child_area=0.0
    all_caps=[]
    for r in desc:
        tetrow=int(r); ids=T3[tetrow]
        B,C=clip(X3[ids],tri)
        area=poly_area(tri,C); total_child_area+=area
        node_ids,nfree,caps=cap_rows(T3,tetrow,B)
        all_caps.extend(caps.tolist())
        child_rows.append({
            "tet_row_403":tetrow,
            "tet_node_rows":node_ids,
            "free_nonphysical_O_nodes":nfree,
            "intersection_polygon_vertices":int(len(B)),
            "clipped_area_m2":area,
            "p1_capacity_phi_free_max1":caps.tolist(),
            "min_capacity":float(caps.min()) if len(caps) else None,
            "max_capacity":float(caps.max()) if len(caps) else None,
            "vertices_world_m":(B@X3[ids]).tolist() if len(B) else [],
        })
    records.append({
        "inventory_bank_index":idx,"bank_key":key,"kind":meta["kind"],"stable_id":meta["stable_id"],
        "body_indices":meta.get("body_indices"),"bank_face_row_zero_based":face_row,
        "bank_face_vertex_rows":tri_ids.tolist(),"triangle_world_vertices_m":tri.tolist(),
        "parent_398_tet_row":parent_row,"parent_398_nodes":parent_nodes.tolist(),
        "parent_intersection_polygon_vertices":int(len(Bp)),"parent_clipped_area_m2":parent_area,
        "sum_descendant_clipped_area_m2":total_child_area,
        "child_over_parent_area_ratio":total_child_area/parent_area if parent_area else None,
        "all_descendant_capacity_values":all_caps,
        "minimum_p1_capacity_phi_free_max1":float(min(all_caps)) if all_caps else None,
        "capacity_rows_below_0p341":int(sum(float(x)<0.341 for x in all_caps)),
        "children":child_rows,
    })
report={
    "schema":"human-global-p1-field-405-tet88190-stable5-face2562-clip.v1",
    "status":"bounded exact-construction clip diagnostic; not a solved/admitted material field",
    "pins":{p:sha(p) for p in EXPECTED},
    "parent_398_tet_row":parent_row,
    "parent_398_node_rows":parent_nodes.tolist(),
    "parent_398_volume_m3":parent_vol,
    "build403_descendant_rows":desc.tolist(),
    "build403_descendant_volume_m3":child_vols,
    "descendant_over_parent_volume_ratio":volratio,
    "physical_outer_node_count":len(outer),
    "target_phi_for_capacity_screen":0.341,
    "capacity_definition":"At each clipped polygon vertex, sum barycentric weights on non-physical-O nodes, assuming phi=0 on physical O and phi<=1 on free nodes. Necessary P1 support capacity only; no solved-field claim.",
    "bank_triangle_clips":records,
    "limitations":["Both typed inventory entries with stable_id=5 are reported separately; they are not conflated.","302 Float64 clipper with exact-binary-rational fallback is construction arithmetic, not final exact collision authority.","Only the designated triangle row and parent/three descendants were queried; no whole-bank or global collision gate was run.","Build-002 remains unadmitted; its recorded six quality failures and parent-volume drift are not resolved here."],
}
path=OUT/"clip-receipt.json"
path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(path),"report_sha256":sha(path),"build_script_sha256":sha(__file__),"records":[{k:r[k] for k in ("bank_key","kind","bank_face_row_zero_based","parent_intersection_polygon_vertices","parent_clipped_area_m2","sum_descendant_clipped_area_m2","child_over_parent_area_ratio","minimum_p1_capacity_phi_free_max1","capacity_rows_below_0p341")} for r in records],"descendants":desc.tolist(),"volume_ratio":volratio},indent=2))
