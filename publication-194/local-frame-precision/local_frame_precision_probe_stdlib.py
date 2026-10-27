from __future__ import annotations
import ast, csv, hashlib, json, math, struct, zipfile
from fractions import Fraction
from pathlib import Path

S = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
C = S / "partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
OUT = S / "per-bank-workers-20261027/common-sub-float32-root-local-frame-001"
PINS = {
    "candidate": "7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af",
    "exact-emitted.triangles.csv": "0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac",
    "exact-emitted.nodes.csv": "99e58d6afb68e0939670f69d4132bfa730cf3e988a7f70a179a41a89eac3802c",
    "exact-emitted.bodies.csv": "dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
    "all-bank-containment.csv.banks.csv": "46568d191d84dd68f1fbcfe02b7938b15d4a7b3b786f268f46bcfe916070c6df",
}

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def f32(x): return struct.unpack("<f",struct.pack("<f",float(x)))[0]
def bits(v): return [struct.unpack("<I",struct.pack("<f",f32(x)))[0] for x in v]
def parse_frac_vec(v): return tuple(Fraction(x) for x in v)
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def mul(a,s): return tuple(x*s for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def orient2(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def axes(n): return tuple(i for i in range(3) if n[i])[:2]
def point_on_tri(p,t):
    n=cross(sub(t[1],t[0]),sub(t[2],t[0]))
    if not any(n) or dot(n,sub(p,t[0])) != 0: return False
    ax=axes(n); q=tuple(p[k] for k in ax)
    signs=[orient2(tuple(a[k] for k in ax),tuple(b[k] for k in ax),q) for a,b in zip(t,t[1:]+t[:1])]
    return min(signs)>=0 or max(signs)<=0

def tri_intersections(a,b):
    out=[]
    def edges_against(src,dst):
        n=cross(sub(dst[1],dst[0]),sub(dst[2],dst[0]))
        for i in range(3):
            p,q=src[i],src[(i+1)%3]
            dp,dq=dot(n,sub(p,dst[0])),dot(n,sub(q,dst[0]))
            if dp==0 and point_on_tri(p,dst): out.append(p)
            if dq==0 and point_on_tri(q,dst): out.append(q)
            if (dp<0<dq) or (dq<0<dp):
                t=dp/(dp-dq); x=add(p,mul(sub(q,p),t))
                if point_on_tri(x,dst): out.append(x)
    edges_against(a,b); edges_against(b,a)
    return sorted(set(out))

def npy_array(zf, name):
    b=zf.read(name+".npy")
    if b[:6]!=b"\x93NUMPY": raise ValueError("bad NPY magic")
    major,minor=b[6],b[7]
    if major==1: hlen=struct.unpack_from("<H",b,8)[0]; start=10
    else: hlen=struct.unpack_from("<I",b,8)[0]; start=12
    h=ast.literal_eval(b[start:start+hlen].decode("latin1").strip())
    desc,shape,fortran=h["descr"],h["shape"],h["fortran_order"]
    if fortran: raise ValueError("Fortran NPY array unsupported")
    code=desc[-2]; size=int(desc[-1]); fmt={"f":"f","i":"i","u":"I","q":"q"}.get(code)
    if fmt is None: raise ValueError("unsupported NPY dtype "+desc)
    fmt="<"+fmt
    count=math.prod(shape); data=b[start+hlen:]
    values=[struct.unpack_from(fmt,data,i*size)[0] for i in range(count)]
    return shape,values

def vec_len(a,b): return math.sqrt(sum((float(x-y))**2 for x,y in zip(a,b)))
def tri_from_rows(rows): return tuple(tuple(Fraction.from_float(float(x)) for x in r) for r in rows)

def read_body(body_rows, body):
    r=next(r for r in body_rows if int(r["body_index"])==body)
    pos=tuple(f32(r["position_"+c+"_m"]) for c in "xyz")
    mat=tuple(tuple(float(r[f"world_to_body_m{i}{j}"]) for j in range(3)) for i in range(3))
    off=tuple(float(r["com_offset_"+c+"_m"]) for c in "xyz")
    return pos,mat,off

def to_local(w,pos,mat,off):
    d=tuple(float(w[i])-pos[i] for i in range(3))
    return tuple(sum(d[i]*mat[j][i] for i in range(3))-off[j] for j in range(3))
def to_world(x,pos,mat,off):
    y=tuple(float(x[i])+off[i] for i in range(3))
    return tuple(sum(y[i]*mat[i][j] for i in range(3))+pos[j] for j in range(3))

for name,path in [("candidate",C)]+[(k,BANKDIR/k) for k in PINS if k!="candidate"]:
    if sha(path)!=PINS[name]: raise RuntimeError("pinned input changed: "+name)
with zipfile.ZipFile(C) as zf:
    vshape,vflat=npy_array(zf,"points_world_m")
    fshape,fflat=npy_array(zf,"source_O_faces")
if vshape[1]!=3 or fshape[1]!=3: raise ValueError("unexpected candidate array shape")
vertex_count=vshape[0]
point_rows=[tuple(float(x) for x in vflat[i*3:i*3+3]) for i in range(vertex_count)]
face_rows=[tuple(int(x) for x in fflat[i*3:i*3+3]) for i in range(fshape[0])]
with (BANKDIR/"exact-emitted.bodies.csv").open(newline="") as f: body_rows=list(csv.DictReader(f))
tri_rows={}
with (BANKDIR/"exact-emitted.triangles.csv").open(newline="") as f:
    for r in csv.DictReader(f):
        key=(int(r["body_index"]),int(r["bone_stable_id"]),int(r["triangle_stable_id"]))
        if key in {(43,16,145),(65,99,174)}:
            tri_rows[key]=tuple(tuple(f32(r[f"v{j}{c}_m"]) for c in "xyz") for j in range(3))

cases=[
    ("43/16",43,16,19841,145,"b43-16/attempt-005-exact-endpoint-identity/receipt.json","b43-16/attempt-005-exact-endpoint-identity/facewise-failure-witness.json"),
    ("65/99",65,99,14173,174,"b65-99/attempt-003-exact-endpoint-identity/receipt.json","b65-99/attempt-003-exact-endpoint-identity/facewise-failure-witness.json"),
]
results=[]
for label,body,bone,face,btri,receipt_rel,witness_rel in cases:
    receipt=json.loads((S/"per-bank-workers-20261027"/receipt_rel).read_text())
    witness=json.loads((S/"per-bank-workers-20261027"/witness_rel).read_text())
    status=next(x for x in receipt["exact_intersections"]["pair_status_and_exact_local_witnesses"] if x["skin_face"]==face and x["bone_triangle"]==btri)
    originals=[parse_frac_vec(p) for p in status["exact_local_endpoints_m"]]
    face_nodes=face_rows[face]
    pos,mat,off=read_body(body_rows,body)
    skin_world=tuple(point_rows[i] for i in face_nodes)
    skin_local=tuple(to_local(p,pos,mat,off) for p in skin_world)
    skin_tri_exact=tri_from_rows(skin_local)
    bone_rows=tri_rows[(body,bone,btri)]
    bone_tri_exact=tri_from_rows(bone_rows)
    exact_original=tri_intersections(skin_tri_exact,bone_tri_exact)
    if len(exact_original)!=2: raise RuntimeError(f"exact reconstructed original pair did not produce segment for {label}: {len(exact_original)}")
    original_length=vec_len(exact_original[0],exact_original[1])
    local_skin32=tuple(tuple(f32(x) for x in p) for p in skin_local)
    local_bone32=tuple(tuple(f32(x) for x in p) for p in bone_rows)
    local_skin_q=tri_from_rows(local_skin32); local_bone_q=tri_from_rows(local_bone32)
    qroots=tri_intersections(local_skin_q,local_bone_q)
    qroots32=tuple(tuple(f32(x) for x in p) for p in qroots)
    local_bits=[bits(p) for p in qroots32]
    world64=tuple(to_world(p,pos,mat,off) for p in qroots32)
    world32=tuple(tuple(f32(x) for x in p) for p in world64)
    world_bits=[bits(p) for p in world32]
    witness_roots=witness["exact_sub_ulp_witnesses"]
    world_exact_by_id={int(x["canonical_exact_root"]):parse_frac_vec(x["exact_world_endpoint_m"]) for x in witness_roots}
    emitted_by_id={int(x["canonical_exact_root"]):x["emitted_world_f32_m"] for x in witness_roots}
    root_ids=[int(x["canonical_exact_root"]) for x in witness_roots]
    expected_world_bits=[bits(tuple(float(x) for x in emitted_by_id[i])) for i in root_ids]
    world_exact=[world_exact_by_id[i] for i in root_ids]
    original_local_f32=tuple(tuple(f32(x) for x in p) for p in exact_original)
    local_rounded_incidence=[]
    for p in qroots32:
        ep=tuple(Fraction.from_float(x) for x in p)
        local_rounded_incidence.append({"skin":point_on_tri(ep,local_skin_q),"bone":point_on_tri(ep,local_bone_q)})
    roundtrip_vertex64=tuple(to_world(p,pos,mat,off) for p in local_skin32)
    roundtrip_vertex32=tuple(tuple(f32(x) for x in p) for p in roundtrip_vertex64)
    vertex_deltas=tuple(tuple(roundtrip_vertex64[i][k]-skin_world[i][k] for k in range(3)) for i in range(3))
    results.append({
        "bank":label,"body":body,"stable_bone_id":bone,"skin_face":face,"bone_triangle":btri,
        "skin_source_vertex_ids":[int(x) for x in face_nodes],
        "retained_exact_local_segment_length_m":status["segment_length_m_for_reporting_only"],
        "reconstructed_exact_local_segment_length_m":original_length,
        "reconstructed_exact_local_root_count":len(exact_original),
        "body_local_float32_reintersection_root_count":len(qroots),
        "body_local_float32_reintersection_segment_length_m":vec_len(qroots[0],qroots[1]) if len(qroots)==2 else None,
        "body_local_float32_roots_m":[[float(x) for x in p] for p in qroots],
        "body_local_float32_root_bit_patterns":local_bits,
        "body_local_float32_roots_distinct":len(set(tuple(x) for x in local_bits))==len(qroots),
        "rounded_local_roots_exactly_on_quantized_skin_and_bone_triangles":local_rounded_incidence,
        "body_local_root_round_trip_world_f32_bits":world_bits,
        "body_local_root_round_trip_world_f32_distinct":len(set(tuple(x) for x in world_bits))==len(qroots),
        "pinned_world_f32_root_bits_from_retained_witness":expected_world_bits,
        "body_local_f32_source_vertex_round_trip_world_max_abs_delta_m":max(abs(x) for r in vertex_deltas for x in r),
        "body_local_f32_source_vertex_round_trip_world_max_norm_delta_m":max(vec_len((0,0,0),r) for r in vertex_deltas),
        "body_local_f32_source_vertex_round_trip_world_bits":[bits(p) for p in roundtrip_vertex32],
        "pinned_source_vertex_world_f32_bits":[bits(p) for p in skin_world],
        "body_local_source_vertex_changed_world_f32_component_count":sum(bits(roundtrip_vertex32[i])[k]!=bits(skin_world[i])[k] for i in range(3) for k in range(3)),
        "body_local_source_vertex_total_world_f32_components":9,
        "world_root_reconstruction_delta_from_exact_m":[vec_len(tuple(float(x) for x in world64[i]),tuple(float(x) for x in world_exact[i])) for i in range(min(len(world64),len(world_exact)))],
    })

report={
    "schema":"numi.human.common-sub-float32-root-local-frame-probe.v1",
    "status":"LOCAL_FLOAT32_DISTINCT_WORLD_FLOAT32_COLLAPSES",
    "candidate_sha256":PINS["candidate"],
    "emitted_input_sha256":{k:v for k,v in PINS.items() if k!="candidate"},
    "results":results,
    "owner_code_review":{
        "double_registered_surface_source":"/Users/numi/Work/unpaired-skin-fat-mass-ownership-070/apps/NumiHumanCoarseTissuePreparation.hpp:21; RegisteredSurface uses V3=std::array<double,3> for donorLocalPoints and worldPoints.",
        "world_f32_fem_gate":"/Users/numi/Work/unpaired-skin-fat-mass-ownership-070/apps/NumiHumanCoarseTissuePreparation.hpp:1163-1173; accepted shell builder casts each world FEM coordinate to Float32 for reference quality admission.",
        "local_point_abi":"/Users/numi/Work/unpaired-skin-fat-mass-ownership-070/include/metalrobo/numi_human_resting_visual_gpu.h:14-18 and /Users/numi/Work/unpaired-skin-fat-mass-ownership-070/src/metal/NumiHumanRestingSupportGeometry.metalinc:79-103; support ABI stores per-body local Float32 influence points and reconstructs compensated world query positions.",
        "mesh_topology_gap":"The local influence ABI carries point/weight/body records, not skin triangle connectivity or a per-patch origin; accepted candidate points_world_m and matched-query parity remain world Float32.",
    },
    "conclusion":"Body-relative Float32 can distinguish the witness endpoints in the local point representation, but transforming those local endpoints through the pinned pose and casting back to the accepted world Float32 geometry collapses them again. The available local Metal ABI is for support point queries and cannot carry the required conforming triangle topology. No safe owner-integrated remesh can be represented without a new local-frame topology/transform contract; this probe changes no geometry or public format.",
    "geometry_changed":False,"candidate_written":False,"full_or_all_bank_audit_run":False,"native_admission":False,"accepted_physical_steps":0,
}
(OUT/"local_frame_precision_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
