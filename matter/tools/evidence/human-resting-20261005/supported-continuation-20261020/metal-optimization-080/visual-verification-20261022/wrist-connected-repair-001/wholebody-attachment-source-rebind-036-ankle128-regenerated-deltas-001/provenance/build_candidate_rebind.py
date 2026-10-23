#!/usr/bin/env python3
"""Regenerate D35 candidate-specific C, donor frames, normals, and locals from candidate."""
import argparse, csv, hashlib, json, math, re, shutil, struct
from collections import defaultdict
from pathlib import Path

parser=argparse.ArgumentParser(description="Candidate-bound skin donor/frame remap using the D28 registered source topology")
parser.add_argument("--config",required=True)
args=parser.parse_args()
CFG_PATH=Path(args.config)
CFG=json.loads(CFG_PATH.read_text())
ROOT=Path(CFG["root"])
D28=Path(CFG["base_candidate_root"])
OUT=Path(CFG["output_root"])
PAYLOAD=Path(CFG["candidate_payload"])
BASEPAYLOAD=Path(CFG["base_payload"])
REGCSV=Path(CFG["registered_candidate_csv"])
CAPTURE=Path(CFG["native_capture_csv"])
CAPTURE_MANIFEST=Path(CFG["capture_manifest"])
CAPTURE_RECEIPT=Path(CFG["capture_receipt"])
D28_PAYLOAD=Path(CFG["parent_payload"])
D28_REGISTERED_CSV=Path(CFG["parent_registered_csv"])
FRAMES=Path(CFG["body_frames"])
CAPTURE_DIAG=Path(CFG["placement_diagnostic"])
PARENT_IDENTITY=json.loads((D28/"D28-candidate-identity.json").read_text())
OLD_ID=PARENT_IDENTITY["candidate_id"]
CANDIDATE_ID=CFG["candidate_id"]
CANDIDATE_TAG=CFG["candidate_tag"]
RUN_TAG=CFG["run_tag"]
EXPECTED_INPUTS=CFG["expected_input_hashes"]
N=2411
SKIN_DEBIT=3.4525707761420956
FAT_DEBIT=10.487108964465744

SOURCE_C_TOL=2.5e-6
ROOT_LIFT_Z=0.35
N=2411
SKIN_DEBIT=3.4525707761420956
FAT_DEBIT=10.487108964465744

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with open(p,newline="") as f: return list(csv.DictReader(f))
def norm(v):
    n=math.sqrt(sum(float(x)*float(x) for x in v))
    if not math.isfinite(n) or n<=1e-12: raise RuntimeError(f"degenerate vector {v}")
    return [float(x)/n for x in v]
def sub(a,b): return [a[i]-b[i] for i in range(3)]
def add(a,b): return [a[i]+b[i] for i in range(3)]
def mv(A,x): return [sum(A[i][j]*x[j] for j in range(3)) for i in range(3)]
def tr(A): return [[A[j][i] for j in range(3)] for i in range(3)]
def dot(a,b): return sum(a[i]*b[i] for i in range(3))
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dist(a,b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))
def det3(a,b,c): return dot(a,cross(b,c))
def quatrot(q):
    x,y,z,w=map(float,q); s=math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(s) or s<=1e-12: raise RuntimeError("invalid body quaternion")
    x,y,z,w=(x/s,y/s,z/s,w/s)
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
def f32(x): return struct.unpack("<f",struct.pack("<f",float(x)))[0]
def f32lit(x):
    s=format(f32(x),".9g")
    if "." not in s and "e" not in s and "E" not in s: s += ".0"
    return s+"f"
def dlit(x): return format(float(x),".17g")
def replace_one(s,old,new,label):
    n=s.count(old)
    if n!=1: raise RuntimeError(f"{label}: expected one occurrence, found {n}")
    return s.replace(old,new)
def replace_decl(s,name,newdecl):
    m=re.search(r"inline constexpr [^\n]*\b"+re.escape(name)+r"\s*=",s)
    if not m: raise RuntimeError(f"missing declaration {name}")
    end=s.find("\ninline constexpr ",m.end())
    if end>=0:
        return s[:m.start()]+newdecl+s[end+1:]
    # The candidate reference-point array is the final constexpr in its header.
    close=s.find("\n}};",m.start())
    if close<0: raise RuntimeError(f"cannot find end of final declaration {name}")
    return s[:m.start()]+newdecl+s[close+4:]
def array3_decl(name,rows):
    return "inline constexpr std::array<std::array<double,3>,2411u> "+name+"={{\n"+",\n".join("    {"+",".join(dlit(x) for x in r)+"}" for r in rows)+"\n}};"
def donor_decl(rows):
    body=["inline constexpr std::array<WholeBodySkinDonorFrameV1,2411u> kWholeBodySkinDonorFrames={"]
    for r in rows:
        body.append("    WholeBodySkinDonorFrameV1{"+str(int(r["donor"]))+"u,{"+",".join(f32lit(r[k]) for k in ("frame_lx","frame_ly","frame_lz"))+"},{"+",".join(f32lit(r["m"+str(i)+str(j)]) for i in range(3) for j in range(3))+"},{"+",".join(f32lit(r[k]) for k in ("cx","cy","cz"))+"},{"+",".join(f32lit(r[k]) for k in ("nx","ny","nz"))+"}},")
    body.append("};")
    return "\n".join(body)
def uint_decl(name,vals):
    lines=["inline constexpr std::array<std::uint32_t,2411u> "+name+"={{"]
    for i in range(0,len(vals),20): lines.append("    "+",".join(str(int(v))+"u" for v in vals[i:i+20])+",")
    lines.append("}};")
    return "\n".join(lines)
def source_points_decl(rows):
    lines=["inline constexpr std::array<std::array<double,3>,2411u> kWholeBodySkinCandidateReferenceWorldPoints={{"]
    for i in range(0,len(rows),3):
        lines.append("    "+",".join("{"+",".join(dlit(v) for v in p)+"}" for p in rows[i:i+3])+",")
    lines.append("}};")
    return "\n".join(lines)

if OUT.name != CANDIDATE_ID: raise RuntimeError("candidate directory identity mismatch")
if not CFG_PATH.is_file(): raise RuntimeError("missing build input configuration")
if (OUT/"generated").exists() or (OUT/"source/apps").exists(): raise RuntimeError("refusing to overwrite an existing candidate generated/source tree")
expected=EXPECTED_INPUTS
actual={"payload":sha(PAYLOAD),"registered_csv":sha(REGCSV),"native_candidate_capture":sha(CAPTURE),
        "D28_payload":sha(D28_PAYLOAD),"D28_registered_csv":sha(D28_REGISTERED_CSV),
        "D28_nodes":sha(D28/"generated/nodes.csv"),"D28_faces":sha(D28/"generated/faces.csv"),
        "D28_tets":sha(D28/"generated/tetrahedra.csv"),"D28_embedding":sha(D28/"generated/embedding.csv"),
        "body_frames":sha(FRAMES),"base_payload":sha(BASEPAYLOAD),
        "D28_transfer_header":sha(D28/"source/apps/NumiHumanWholeBodySkinTransferAncestry.hpp"),
        "capture_manifest":sha(CAPTURE_MANIFEST),"capture_receipt":sha(CAPTURE_RECEIPT),
        "placement_diagnostic":sha(CAPTURE_DIAG)}
for k,v in expected.items():
    if actual[k]!=v: raise RuntimeError(f"immutable input hash mismatch {k}: {actual[k]} != {v}")

# Decode ABI5 source and certify the candidate retains the registered topology/bind table.
raw=PAYLOAD.read_bytes(); magic,abi,nb,nv,ni,flags=struct.unpack_from("<8s5I",raw,0)
if (magic,abi,nb,nv,ni)!=(b"NHSKIN1\0",5,86,61950,326961): raise RuntimeError("unexpected candidate ABI/counts")
bindoff=60; vertoff=bindoff+nb*36; faceoff=vertoff+nv*56; weightoff=faceoff+ni*4
base=BASEPAYLOAD.read_bytes()
if raw[:bindoff+nb*36] != base[:bindoff+nb*36]: raise RuntimeError("candidate source binding table differs from registered source contract")
if raw[faceoff:weightoff] != base[faceoff:weightoff]: raise RuntimeError("candidate detailed face topology differs from registered source ancestry")
bind_bodies=[struct.unpack_from("<I",raw,bindoff+i*36)[0] for i in range(nb)]
faces=[struct.unpack_from("<3I",raw,faceoff+12*i) for i in range(ni//3)]
weights_detail=[struct.unpack_from("<86f",raw,weightoff+i*nb*4) for i in range(nv)]

emb=readcsv(D28/"generated/embedding.csv")
old_donor=readcsv(D28/"generated/donor-frames-fullweights.csv")
nodes=readcsv(D28/"generated/nodes.csv")
tets=readcsv(D28/"generated/tetrahedra.csv")
export=readcsv(REGCSV); capture=readcsv(CAPTURE)
if not (len(emb)==len(old_donor)==len(nodes)==N and len(export)==len(capture)==nv and len(tets)==28632 and len(faces)==108987):
    raise RuntimeError("row cardinality mismatch")
if any(int(r["row"])!=i for i,r in enumerate(emb)) or any(int(r["row"])!=i for i,r in enumerate(old_donor)): raise RuntimeError("reduction row order mismatch")
if any(int(r["vertex"])!=i for i,r in enumerate(export)) or any(int(r["vertex"])!=i for i,r in enumerate(capture)): raise RuntimeError("detailed source row order mismatch")
# Native current-pose capture was made with this candidate payload. Its XYZ
# is in placed capture coordinates; the paired export CSV is in registered reference coordinates.
cap_normals=[[float(r[k]) for k in ("nx","ny","nz")] for r in capture]
norm_errors=[abs(math.sqrt(dot(v,v))-1.0) for v in cap_normals]
if max(norm_errors)>3.0e-6: raise RuntimeError(f"native candidate normal capture is nonunit: {max(norm_errors)}")

# The exact source-to-reference placement maps reference normals into capture normals by Q.
diag=json.loads(CAPTURE_DIAG.read_text()); Q=[[float(diag["placement_rotation"][i][j]) for j in range(3)] for i in range(3)]
capture_t=[float(x) for x in diag["placement_translation_m"]]
QT=tr(Q)
orth=max(abs(sum(Q[k][i]*Q[k][j] for k in range(3))-(1.0 if i==j else 0.0)) for i in range(3) for j in range(3))
if orth>2e-8: raise RuntimeError(f"common placement rotation not orthogonal: {orth}")
capture_xyz_error=0.0
for a,b in zip(export,capture):
    capture_p=[float(b["w"+axis]) for axis in "xyz"]
    reference_p=mv(QT,sub(capture_p,capture_t))
    exporter_p=[float(a["w"+axis]) for axis in "xyz"]
    capture_xyz_error=max(capture_xyz_error,max(abs(reference_p[i]-exporter_p[i]) for i in range(3)))
if capture_xyz_error>2.5e-6: raise RuntimeError(f"native candidate capture transformed back to reference disagrees with candidate exporter: {capture_xyz_error}")

# Load the frozen D28 source rows, candidate body frames, and unchanged tissue columns.
frame_json=json.loads(FRAMES.read_text())
body_R={}; body_t={}
for f in frame_json["source_poses"]:
    b=int(f["core_body"]); R=quatrot(f["quaternion_xyzw"])
    body_R[b]=R
    p=[float(f["position"][j])+(ROOT_LIFT_Z if j==2 else 0.0) for j in range(3)]
    body_t[b]=p
# Verify the exact FK frame convention against all D28 rows before applying any replacement.
old_frame_max=0.0; old_rot_max=0.0
for i,r in enumerate(old_donor):
    b=int(r["donor"])
    if b not in body_R: raise RuntimeError(f"missing old donor FK body {b}")
    M=tr(body_R[b]); C=[float(r["c"+a]) for a in "xyz"]
    local=mv(M,sub(C,body_t[b]))
    oldframe=[float(r[k]) for k in ("frame_lx","frame_ly","frame_lz")]
    old_frame_max=max(old_frame_max,dist(local,oldframe))
    oldM=[[float(r["m"+str(x)+str(y)]) for y in range(3)] for x in range(3)]
    old_rot_max=max(old_rot_max,max(abs(M[x][y]-oldM[x][y]) for x in range(3) for y in range(3)))
if old_frame_max>1e-6 or old_rot_max>1e-6: raise RuntimeError(f"D28 frame calibration mismatch: local={old_frame_max} R={old_rot_max}")

new_weights=[]; source_C=[]; candidate_donors=[]; source_normals=[]; part_errors=[]
raw_part_errors=[]; dominant_float_residuals=[]
for i,e in enumerate(emb):
    sv=int(e["source_vertex"]); fi=int(e["face"]); beta=[float(e["b"+str(j)]) for j in range(3)]
    if fi==4294967295:
        if not 0<=sv<nv: raise RuntimeError(f"invalid direct source row {i}")
        ids=(sv,); bw=(1.0,)
    else:
        if not 0<=fi<len(faces): raise RuntimeError(f"invalid source face row {i}")
        if min(beta)<-1e-8 or max(beta)>1+1e-8 or abs(sum(beta)-1.0)>2e-8: raise RuntimeError(f"invalid source barycentrics row {i}")
        ids=faces[fi]; bw=beta
    w=[sum(bw[j]*weights_detail[ids[j]][k] for j in range(len(ids))) for k in range(86)]
    if not all(math.isfinite(x) and x>=0.0 for x in w): raise RuntimeError(f"invalid 86-weight row {i}")
    raw_sum=math.fsum(w)
    if not math.isfinite(raw_sum) or raw_sum<=0.0: raise RuntimeError(f"invalid 86-weight partition row {i}: {raw_sum}")
    raw_part_errors.append(abs(raw_sum-1.0))
    # Barycentric interpolation preserves partition unity algebraically; normalize
    # its accumulated Float32-source roundoff in double, then encode as Float32.
    w=[x/raw_sum for x in w]
    wf=[f32(x) for x in w]
    lane_for_residual=max(range(86),key=lambda k:(wf[k],-k))
    residual=1.0-math.fsum(wf)
    adjusted=f32(wf[lane_for_residual]+residual)
    if adjusted>=0.0 and math.isfinite(adjusted): wf[lane_for_residual]=adjusted
    dominant_float_residuals.append(residual)
    w=[float(x) for x in wf]
    if not all(math.isfinite(x) and x>=0.0 for x in w): raise RuntimeError(f"invalid encoded 86-weight row {i}")
    part_errors.append(abs(math.fsum(w)-1.0))
    p=[sum(bw[j]*float(export[ids[j]]["w"+a]) for j in range(len(ids))) for a in "xyz"]
    ncap=[sum(bw[j]*cap_normals[ids[j]][k] for j in range(len(ids))) for k in range(3)]
    nref=norm(mv(QT,ncap))
    lane=max(range(86),key=lambda k:(w[k],-k)); donor=bind_bodies[lane]
    if donor not in body_R: raise RuntimeError(f"new dominant donor body lacks FK frame row={i} donor={donor}")
    new_weights.append(w); source_C.append(p); source_normals.append(nref); candidate_donors.append(donor)
if max(part_errors)>1.0e-4: raise RuntimeError(f"candidate Float32 weight partition exceeds native owner bound: {max(part_errors)}")

# Preserve the established source-to-C residual, then form C and O-relative donor deltas once.
Cnew=[]; source_motion=[]; C_motion=[]; donor_changed=[]
for i,r in enumerate(old_donor):
    old_source=[float(r["source_c"+a]) for a in "xyz"]
    old_C=[float(r["c"+a]) for a in "xyz"]
    reg=[float(r["reg_r"+a]) for a in "xyz"]
    C=add(source_C[i],reg)
    Cnew.append(C); source_motion.append(dist(source_C[i],old_source)); C_motion.append(dist(C,old_C))
    if int(r["donor"])!=candidate_donors[i]: donor_changed.append(i)
if max(source_motion)>SOURCE_C_TOL or max(C_motion)>SOURCE_C_TOL:
    raise RuntimeError(f"candidate source/C changed beyond existing tolerance: source={max(source_motion)} C={max(C_motion)}")

# Copy only the immutable generated tables and private source snapshot; never touch D28.
shutil.copytree(D28/"generated",OUT/"generated")
shutil.copytree(D28/"source/apps",OUT/"source/apps")
# Regenerate full row-aligned source weights, C, candidate frame, O/I/D local positions, and reference normals.
outrows=[]; inner_locals=[]; deep_locals=[]
for i,(old,nrow) in enumerate(zip(old_donor,nodes)):
    r=dict(old); w=new_weights[i]; b=candidate_donors[i]; R=body_R[b]; M=tr(R); t=body_t[b]
    O=[float(nrow["o"+a]) for a in "xyz"]
    I=[float(nrow["i"+a]) for a in "xyz"]
    D=[float(nrow["d"+a]) for a in "xyz"]
    C=Cnew[i]; source=source_C[i]
    frame_l=mv(M,sub(C,t)); mech_w=sub(O,C); mech_l=mv(M,mech_w); outer_l=add(frame_l,mech_l)
    inner_l=add(outer_l,mv(M,sub(I,O))); deep_l=add(outer_l,mv(M,sub(D,O)))
    inner_locals.append(inner_l); deep_locals.append(deep_l)
    r["donor"]=str(b)
    for j,a in enumerate("xyz"):
        r["source_c"+a]=dlit(source[j]); r["c"+a]=dlit(C[j]); r["reg_r"+a]=old["reg_r"+a]
        r["mech_w"+a]=dlit(mech_w[j]); r["mech_l"+a]=dlit(mech_l[j]); r["frame_l"+a]=dlit(frame_l[j]); r["l"+a]=dlit(outer_l[j])
        r["nx"] if False else None
    for j,a in enumerate("xyz"):
        r["n"+a]=dlit(source_normals[i][j])
    # Local layer arrays are stored in the reduction header; retain matching row values in donor table too.
    for a in range(3):
        for c in range(3): r["m"+str(a)+str(c)]=dlit(M[a][c])
    for k in range(86): r["w"+str(k)]=dlit(w[k])
    outrows.append(r)
fields=list(old_donor[0].keys())
donor_csv=OUT/"generated/donor-frames-fullweights.csv"
with donor_csv.open("w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); wr.writeheader(); wr.writerows(outrows)

# Read actual output donor-local fields back and check all reference O/I/D layer relations.
max_O_roundtrip=max_I_rel=max_D_rel=0.0
for i,r in enumerate(outrows):
    b=int(r["donor"]); M=[[float(r["m"+str(x)+str(y)]) for y in range(3)] for x in range(3)]
    O=[float(nodes[i]["o"+a]) for a in "xyz"]; I=[float(nodes[i]["i"+a]) for a in "xyz"]; D=[float(nodes[i]["d"+a]) for a in "xyz"]
    local=[float(r["l"+a]) for a in "xyz"]
    wO=add(mv(body_R[b],local),body_t[b]); max_O_roundtrip=max(max_O_roundtrip,dist(wO,O))
    il=inner_locals[i]; dl=deep_locals[i]
    max_I_rel=max(max_I_rel,dist(sub(il,local),mv(M,sub(I,O))))
    max_D_rel=max(max_D_rel,dist(sub(dl,local),mv(M,sub(D,O))))
if max_O_roundtrip>2.5e-6 or max_I_rel>2e-6 or max_D_rel>2e-6: raise RuntimeError(f"candidate donor layer frames fail: O={max_O_roundtrip} I={max_I_rel} D={max_D_rel}")

# Rebuild 2411 local layer constants and full donor-frame table.
reduction_path=OUT/"source/apps/NumiHumanWholeBodySkinReduction.hpp"
text=reduction_path.read_text()
old_payload=PARENT_IDENTITY["source_payload_sha256"]
old_csv=PARENT_IDENTITY["registered_csv_sha256"]
new_payload=sha(PAYLOAD); new_csv=sha(REGCSV)
text=replace_one(text,old_payload,new_payload,"reduction source payload hash")
text=replace_one(text,old_csv,new_csv,"reduction registered CSV hash")
text=replace_decl(text,"kWholeBodySkinDonorFrames",donor_decl(outrows))
text=replace_decl(text,"kWholeBodySkinInnerLocal",array3_decl("kWholeBodySkinInnerLocal",inner_locals))
text=replace_decl(text,"kWholeBodyFatDeepLocal",array3_decl("kWholeBodyFatDeepLocal",deep_locals))
mechanical_world_deltas=[[float(r["mech_w"+a]) for a in "xyz"] for r in outrows]
mechanical_local_deltas=[[float(r["mech_l"+a]) for a in "xyz"] for r in outrows]
world_delta_decl=array3_decl("kWholeBodySkinMechanicalWorldDeltas",mechanical_world_deltas)
local_delta_decl=array3_decl("kWholeBodySkinMechanicalDonorLocalDeltas",mechanical_local_deltas)
def replace_array3_decl(source,name,newdecl):
    pattern=re.compile(r"inline constexpr\s+std::array<std::array<double,3>,2411u>\s+"+re.escape(name)+r"\s*=\s*\{\{.*?\}\};",re.S)
    replaced,count=pattern.subn(lambda _match:newdecl,source,count=1)
    if count!=1: raise RuntimeError(f"{name}: expected exactly one complete array declaration, replaced {count}")
    return replaced
text=replace_array3_decl(text,"kWholeBodySkinMechanicalWorldDeltas",world_delta_decl)
text=replace_array3_decl(text,"kWholeBodySkinMechanicalDonorLocalDeltas",local_delta_decl)
# Compare every emitted C++ row to its candidate donor-frame CSV row before freezing.
def parse_array3_decl(decl):
    rows=re.findall(r"\{([^{}]+)\}",decl[decl.index("={{")+3:decl.rfind("}};")])
    return [[float(x) for x in re.findall(r"[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?",row)] for row in rows]
parsed_world=parse_array3_decl(world_delta_decl)
parsed_local=parse_array3_decl(local_delta_decl)
if len(parsed_world)!=N or len(parsed_local)!=N: raise RuntimeError(f"mechanical declaration parser counts {len(parsed_world)}/{len(parsed_local)} != {N}")
if text.count(world_delta_decl)!=1 or text.count(local_delta_decl)!=1: raise RuntimeError("mechanical declarations are not uniquely present in the emitted header")
mechanical_world_header_csv_max=max(abs(parsed_world[i][j]-mechanical_world_deltas[i][j]) for i in range(N) for j in range(3))
mechanical_local_header_csv_max=max(abs(parsed_local[i][j]-mechanical_local_deltas[i][j]) for i in range(N) for j in range(3))
max_f32_donor_frame_world_to_local_delta_residual=max(
    dist(mv([[f32(float(outrows[i]["m"+str(r)+str(c)])) for c in range(3)] for r in range(3)],mechanical_world_deltas[i]),mechanical_local_deltas[i])
    for i in range(N))
if mechanical_world_header_csv_max>1e-15 or mechanical_local_header_csv_max>1e-15:
    raise RuntimeError("emitted mechanical constexpr arrays differ from candidate CSV")
if max_f32_donor_frame_world_to_local_delta_residual>2.0e-6:
    raise RuntimeError(f"Float32 pinned donor-frame mechanical delta mapping exceeds existing owner tolerance: {max_f32_donor_frame_world_to_local_delta_residual}")
reduction_path.write_text(text)
reduction_sha=sha(reduction_path)

reg_path=OUT/"source/apps/NumiHumanFinal080CandidateRegistration.hpp"
reg_text=reg_path.read_text()
reg_text=replace_one(reg_text,old_payload,new_payload,"registration candidate payload hash")
reg_text=replace_decl(reg_text,"kWholeBodySkinCandidateDonorBodies",uint_decl("kWholeBodySkinCandidateDonorBodies",candidate_donors))
reg_text=replace_decl(reg_text,"kWholeBodySkinCandidateReferenceWorldPoints",source_points_decl(Cnew))
# D35 has an explicit finalcandidate header name; update app's include below.
new_reg_path=OUT/f"source/apps/NumiHumanFinal{CANDIDATE_TAG}CandidateRegistration.hpp"
reg_path.rename(new_reg_path); new_reg_path.write_text(reg_text)
app_path=OUT/"source/apps/numilab_human_myosim_visual_probe.mm"
app=app_path.read_text()
app=app.replace(f'"{reg_path.name}"',f'"{new_reg_path.name}"')
# Update duplicated source-bound identity constants after the candidate identity file is frozen.
old_id=OLD_ID
if old_id not in app: raise RuntimeError("D28 candidate ID absent from app")
app=app.replace(old_id,CANDIDATE_ID)
# Keep the transfer ancestry bound to the new payload/candidate CSV and exact new reduction header.
transfer_path=OUT/"source/apps/NumiHumanWholeBodySkinTransferAncestry.hpp"
transfer=transfer_path.read_text()
transfer=replace_one(transfer,old_payload,new_payload,"transfer source payload hash")
transfer=replace_one(transfer,old_csv,new_csv,"transfer registered CSV hash")
m_transfer_reduction=re.search(r"kWholeBodySkinTransferReductionHeaderSHA256\s*=\s*\x22([a-f0-9]{64})\x22",transfer)
if not m_transfer_reduction: raise RuntimeError("missing transfer reduction-header hash field")
old_reduction_header=m_transfer_reduction.group(1)
transfer=replace_one(transfer,old_reduction_header,reduction_sha,"transfer reduction header hash")
transfer_path.write_text(transfer)

# Candidate source ancestry and all physical geometry/material source tables remain unchanged.
for rel in ("nodes.csv","faces.csv","tetrahedra.csv","embedding.csv","skin-thickness.csv","fat-thickness.csv","parent-tetrahedra.csv","midpoint-ancestry.csv"):
    a=D28/"generated"/rel; b=OUT/"generated"/rel
    if a.exists() and (not b.exists() or sha(a)!=sha(b)): raise RuntimeError(f"physical/ancestry table changed: {rel}")

# Recheck the exact 14 retained inferred attachment sites under the existing policy.
attach_pat=re.compile(r"\{\s*(\d+)u,\s*(\d+)u,\s*(\d+)u\s*\}")
header=(OUT/"source/apps/NumiHumanWholeBodySkinReduction.hpp").read_text()
block=header[header.index("kWholeBodySkinDeepAttachmentSites"):header.index("inline constexpr char kWholeBodySkinSourcePayloadSHA256")]
attachments=[tuple(map(int,m)) for m in attach_pat.findall(block)]
if len(attachments)!=14: raise RuntimeError(f"expected 14 retained sites, got {len(attachments)}")
sm=re.search(r"kWholeBodySkinSourceVertices=\{(.*?)\};",header,re.S)
if not sm: raise RuntimeError("source vertex list missing")
source_ids=[int(x) for x in re.findall(r"\d+",sm.group(1))]
if len(source_ids)!=N: raise RuntimeError("wrong source ID count")
# Per-node deep-layer altitudes from exact unchanged tetrahedra and node geometry.
coords=[]
for keys in (("ox","oy","oz"),("ix","iy","iz"),("dx","dy","dz")):
    coords.extend([[float(r[k]) for k in keys] for r in nodes])
incident=[[] for _ in range(3*N)]
for t in tets:
    ids=[int(t[k]) for k in "abcd"]; a,b,c,d=[coords[x] for x in ids]
    vol6=abs(det3(sub(b,a),sub(c,a),sub(d,a)))
    for omit,v in enumerate(ids):
        tri=[coords[ids[j]] for j in range(4) if j!=omit]
        area2=math.sqrt(dot(cross(sub(tri[1],tri[0]),sub(tri[2],tri[0])),cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]))))
        if vol6<=0 or area2<=0: raise RuntimeError("degenerate unchanged tetrahedron")
        incident[v].append(vol6/area2)
att_rows=[]; failed=[]
for row,source,body in attachments:
    r=outrows[row]; maxw=max(float(r["w"+str(k)]) for k in range(86)); got=int(r["donor"])
    deep_alt=min(incident[2*N+row])*1000.0
    ok=(source_ids[row]==source and got==body and maxw>=0.85 and deep_alt>=0.05)
    item={"reduced_row":row,"source_vertex":source,"expected_donor":body,"actual_donor":got,"maximum_full86_weight":maxw,"minimum_deep_fat_incident_altitude_mm":deep_alt,"passed_identity_weight_altitude":ok}
    att_rows.append(item)
    if not ok: failed.append(item)
for i,a in enumerate(att_rows):
    for b in att_rows[i+1:]:
        if a["actual_donor"]==b["actual_donor"]:
            pa=[float(nodes[a["reduced_row"]]["o"+k]) for k in "xyz"]; pb=[float(nodes[b["reduced_row"]]["o"+k]) for k in "xyz"]
            if dist(pa,pb)<0.025:
                failed.append({"reason":"same-donor attachment spacing","rows":[a["reduced_row"],b["reduced_row"]]})
counts=defaultdict(int)
for a in att_rows: counts[a["actual_donor"]]+=1
for donor,count in counts.items():
    if count>3: failed.append({"reason":"per-donor attachment capacity","donor":donor,"count":count})
if failed: raise RuntimeError("one or more D28 retained attachment sites fail D35 exact donor/weight/altitude/spacing policy")

# Report source donor mass redistribution using the unchanged D28 lumped-volume policy; native mass dump is authoritative.
mat=defaultdict(lambda:{"volume":0.0,"nodes":defaultdict(float),"count":0})
for t in tets:
    ids=[int(t[k]) for k in "abcd"]; a,b,c,d=[coords[x] for x in ids]
    vol=abs(det3(sub(b,a),sub(c,a),sub(d,a)))/6.0
    group="skin" if max(v//N for v in ids)<=1 else "fat" if min(v//N for v in ids)>=1 else "mixed"
    if group=="mixed" or vol<=0: raise RuntimeError(f"invalid tet group/volume {ids}")
    mat[group]["volume"]+=vol; mat[group]["count"]+=1
    for v in ids: mat[group]["nodes"][v]+=vol/4.0
old_donors=[int(r["donor"]) for r in old_donor]
new_donors=candidate_donors
alloc={}
for group,debit in (("skin",SKIN_DEBIT),("fat",FAT_DEBIT)):
    density=debit/mat[group]["volume"]; alloc[group]={"density":density,"old":defaultdict(float),"new":defaultdict(float)}
    for v,vol in mat[group]["nodes"].items():
        row=v%N; mass=vol*density; alloc[group]["old"][old_donors[row]]+=mass; alloc[group]["new"][new_donors[row]]+=mass
    if abs(sum(alloc[group]["old"].values())-debit)>1e-8*debit or abs(sum(alloc[group]["new"].values())-debit)>1e-8*debit: raise RuntimeError(f"{group} total debit not conserved")
bodyids=sorted(set(old_donors)|set(new_donors))
with (OUT/"donor-mass-allocation-delta.csv").open("w",newline="") as f:
    wr=csv.writer(f,lineterminator="\n")
    wr.writerow(["core_body","skin_before_kg","skin_after_kg","skin_delta_kg","fat_before_kg","fat_after_kg","fat_delta_kg","total_delta_kg"])
    for b in bodyids:
        sb=alloc["skin"]["old"].get(b,0.); sa=alloc["skin"]["new"].get(b,0.); fb=alloc["fat"]["old"].get(b,0.); fa=alloc["fat"]["new"].get(b,0.)
        wr.writerow([b,*[dlit(x) for x in (sb,sa,sa-sb,fb,fa,fa-fb,sa+fa-sb-fb)]])

# Update the one private candidate identity, then bind the app diagnostic guard to its exact hash.
label_path=OUT/"source/apps/NumiHumanSkinFatReductionLabels.hpp"
old_identity=json.loads((D28/"D28-candidate-identity.json").read_text())
identity={
 "schema":"numi.human.skin.attachment-candidate-identity.d35.v1","candidate_id":CANDIDATE_ID,
 "base_candidate_id":OLD_ID,"source_payload_sha256":new_payload,"registered_csv_sha256":new_csv,
 "native_registered_normal_capture_sha256":sha(CAPTURE),
 "mechanical_csv_sha256":old_identity["mechanical_csv_sha256"],
 "source_reduction_header_sha256":sha(reduction_path),"source_label_owner_sha256":sha(label_path),
 "candidate_geometry":{"nodes_csv_sha256":sha(OUT/"generated/nodes.csv"),"faces_csv_sha256":sha(OUT/"generated/faces.csv"),"tetrahedra_csv_sha256":sha(OUT/"generated/tetrahedra.csv")},
 "attachment_count":14,"omitted_stale_inferred_attachment_count":6,"physical_qualification":False
}
identity_path=OUT/f"{RUN_TAG}-candidate-identity.json"
identity_path.write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n")
identity_sha=sha(identity_path)
oldsha=sha(D28/"D28-candidate-identity.json")
if oldsha not in app: raise RuntimeError("D28 identity hash missing from app")
app=app.replace(oldsha,identity_sha)
app_path.write_text(app)

# Candidate-specific D35 receipt: identity and remap only; no mechanical qualification asserted.
weight_changed=[i for i,(a,b) in enumerate(zip(old_donor,outrows)) if max(abs(float(a["w"+str(k)])-float(b["w"+str(k)])) for k in range(86))>1e-7]
point_changed=[i for i,(a,b) in enumerate(zip(old_donor,outrows)) if dist([float(a["source_c"+k]) for k in "xyz"],[float(b["source_c"+k]) for k in "xyz"])>1e-12]
report={
 "schema":"numi.human.skin.source-rebind.d35.report.v1","candidate_id":CANDIDATE_ID,"status":"PRIVATE_UNADMITTED_DIAGNOSTIC","physical_qualification":False,
 "inputs":{"candidate_payload_sha256":new_payload,"candidate_export_csv_sha256":new_csv,"native_candidate_capture_sha256":sha(CAPTURE),"D28_identity_sha256":sha(D28/"D28-candidate-identity.json"),"body_frames_sha256":sha(FRAMES),"normal_transform_diagnostic_sha256":sha(CAPTURE_DIAG)},
 "native_preflight":{"binary_sha256":json.loads(CAPTURE_MANIFEST.read_text())["binary_sha256"],"returncode":json.loads(CAPTURE_RECEIPT.read_text())["returncode"],"strict_rejection":"candidate source payload hash does not match its declaration","capture_written_before_rejection":json.loads(CAPTURE_RECEIPT.read_text())["registered_csv_exists"],"capture_rows":len(capture),"capture_coordinates":"placed capture frame; transformed to reference by verified Q^T*(x-t)","max_export_vs_native_reference_xyz_component_error_m":capture_xyz_error,"capture_sha256":sha(CAPTURE),"capture_manifest_sha256":sha(CAPTURE_MANIFEST),"capture_receipt_sha256":sha(CAPTURE_RECEIPT)},
 "counts":{"reduced_rows":N,"faces":4777,"tets":28632,"candidate_weight_changed_rows":len(weight_changed),"candidate_source_C_changed_rows":len(point_changed),"dominant_donor_changed_rows":len(donor_changed),"retained_deep_attachment_sites":len(attachments),"omitted_stale_inferred_hand_sites":6},
 "reference_checks":{"source_C_max_movement_vs_D28":max(source_motion),"registered_C_max_movement_vs_D28":max(C_motion),"source_C_tolerance_m":SOURCE_C_TOL,"max_raw_barycentric_weight_partition_error":max(raw_part_errors),"max_emitted_Float32_weight_partition_error":max(part_errors),"native_Float32_weight_partition_bound":1.0e-4,"max_dominant_lane_Float32_residual_adjustment":max(abs(x) for x in dominant_float_residuals),"old_D28_frame_calibration_max_m":old_frame_max,"old_D28_rotation_calibration_max":old_rot_max,"max_outer_local_reference_roundtrip_m":max_O_roundtrip,"max_inner_relative_frame_error_m":max_I_rel,"max_deep_relative_frame_error_m":max_D_rel,"placement_rotation_orthogonality_error":orth,"mechanical_world_delta_header_csv_max_abs_m":mechanical_world_header_csv_max,"mechanical_local_delta_header_csv_max_abs_m":mechanical_local_header_csv_max,"max_f32_donor_frame_world_to_local_delta_residual_m":max_f32_donor_frame_world_to_local_delta_residual,"native_normal_rows_changed_vs_D28":sum(dist(source_normals[i],[float(old_donor[i]["n"+a]) for a in "xyz"])>1e-10 for i in range(N)),"max_native_source_normal_delta_vs_D28":max(dist(source_normals[i],[float(old_donor[i]["n"+a]) for a in "xyz"]) for i in range(N))},
 "attachment_policy":{"minimum_max_source_weight":.85,"minimum_deep_fat_incident_altitude_mm":.05,"maximum_per_donor":3,"minimum_same_donor_spacing_mm":25.0,"sites":att_rows,"all_pass":True,"omitted_six_not_replaced":True},
 "geometry_mass":{"nodes_faces_tets_unchanged":True,"skin_tets":mat["skin"]["count"],"fat_tets":mat["fat"]["count"],"skin_volume_m3":mat["skin"]["volume"],"fat_volume_m3":mat["fat"]["volume"],"fixed_skin_debit_kg":SKIN_DEBIT,"fixed_fat_debit_kg":FAT_DEBIT,"per_dominant_donor_mass_delta_csv":"donor-mass-allocation-delta.csv","runtime_mass_and_capacity_owner_required":True},
 "limitations":["This D35 derivative regenerates both mechanical delta arrays from the candidate candidate donor-frame CSV; earlier candidates are retained as failed evidence. No physical qualification is claimed.","D35 preserves D28 O/I/D, topology, thickness, and total tissue mass. It changes candidate source weights, source points, dominant donor rows, and their local frames.","The 14 attachment sites are revalidated as inferred sites under the unchanged policy; six superseded hand sites remain omitted and no replacement entheses are asserted.","A successful source-only initialization and short physical diagnostic, if reached, would remain unadmitted evidence, not ROM or anatomical qualification."]
}
report_path=OUT/f"{RUN_TAG}-remap-report.json"; report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
builder_copy=OUT/"provenance/build_candidate_rebind.py"
if Path(__file__).resolve()!=builder_copy.resolve(): shutil.copy2(Path(__file__),builder_copy)
config_copy=OUT/"provenance/build_candidate_rebind_inputs.json"
if CFG_PATH.resolve()!=config_copy.resolve(): shutil.copy2(CFG_PATH,config_copy)
manifest={"schema":f"numi.human.skin.source-rebind.{RUN_TAG.lower()}.manifest.v1","candidate_id":CANDIDATE_ID,"status":"PRIVATE_UNADMITTED_DIAGNOSTIC","input_hashes":actual,"output_hashes":{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in (f"{RUN_TAG}-manifest.json",f"{RUN_TAG}-manifest.sha256") and "failed-generator-attempt-" not in p.relative_to(OUT).as_posix()},
 "candidate_identity_sha256":identity_sha,"source_reduction_header_sha256":sha(reduction_path),"source_registration_header_sha256":sha(new_reg_path),"transfer_ancestry_header_sha256":sha(transfer_path),"donor_table_sha256":sha(donor_csv),"report_sha256":sha(report_path),"physical_qualification":False}
mp=OUT/f"{RUN_TAG}-manifest.json"; mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
(OUT/f"{RUN_TAG}-manifest.sha256").write_text(sha(mp)+f"  {RUN_TAG}-manifest.json\n")
print(json.dumps({"out":str(OUT),"candidate_identity_sha256":identity_sha,"reduction_header_sha256":sha(reduction_path),"registration_header_sha256":sha(new_reg_path),"transfer_header_sha256":sha(transfer_path),"app_sha256":sha(app_path),"manifest_sha256":sha(mp),"report_sha256":sha(report_path),"counts":report["counts"],"C_source_max_movement":max(source_motion),"C_registered_max_movement":max(C_motion),"normal_rows_changed":report["reference_checks"]["native_normal_rows_changed_vs_D28"],"max_outer_roundtrip_m":max_O_roundtrip,"max_inner_relative_m":max_I_rel,"max_deep_relative_m":max_D_rel,"attachment_policy_pass":True,"fixed_mass_kg":[SKIN_DEBIT,FAT_DEBIT]},indent=2))

