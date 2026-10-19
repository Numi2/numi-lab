#!/usr/bin/env python3
import csv,hashlib,json,math,pathlib,re,struct
from collections import defaultdict
BASE=pathlib.Path("/Users/numi/numi-human-separation-20261018/C")
C=BASE/"wholebody-skin-layer-fitting-296"
cand=C/"infeasible-cone-remesh-024/candidate-024-safe"
solve=C/"column-direction-trial-001/direct-coordinate-repair-025/solve-032-local-fat-separation"
out=solve
header_in=cand/"NumiHumanWholeBodySkinReduction.hpp"
orig_csv=BASE/"wholebody-skin-weight-repair-293/native-native-root-pose-003/registered-skin.csv"
mechanical_csv=cand/"registered-skin-mechanical.csv"
coords=solve/"admitted-layer-coordinates.csv"
audit=solve/"float32-geometry-audit-rebased.json"
boundary_audit=BASE/"wholebody-closed-boundary-audit-300/run-solve032c/result.json"
direction_csv=C/"column-direction-trial-001/max-margin-inferred-fit-028/directions.csv"
repair_receipt=C/"infeasible-cone-remesh-024/topology-ancestry-quality-receipt-safe.json"
manifest=BASE/"publication-source-001/docs/media/numi-human-whole-body-all-dof-v1/source-manifest.json"
full_gate=BASE/"wholebody-integration-295/from-layers-wholebody-test-005"
api_gate=BASE/"wholebody-integration-295/from-layers-test-007"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p,key):
 with p.open(newline="") as f:
  rr=list(csv.DictReader(f))
 return {int(r[key]):r for r in rr}
orig=readrows(orig_csv,"vertex"); mech=readrows(mechanical_csv,"vertex")
with coords.open(newline="") as f:
 rows=list(csv.DictReader(f))
assert len(rows)==2332
# Preserve the frozen Float32 world geometry while rebasing all frame-relative values
# to the exact registered source frame consumed by the app.
cols=list(rows[0].keys())
def f32(x): return struct.unpack("<f",struct.pack("<f",float(x)))[0]
def frame(a):
 return [[float(a[f"m{r}{c}"]) for c in range(3)] for r in range(3)]
def mv(M,v):
 return [sum(M[r][c]*v[c] for c in range(3)) for r in range(3)]
for r in rows:
 sid=int(r["source_vertex"]); a=orig[sid]; M=frame(a)
 O=[f32(r[f"outer_{x}"]) for x in "xyz"]
 S=[f32(r[f"skin_{x}"]) for x in "xyz"]
 T=[f32(r[f"fat_{x}"]) for x in "xyz"]
 OW=[float(a[k]) for k in ("wx","wy","wz")]
 LL=[float(a[k]) for k in ("lx","ly","lz")]
 for prefix,P in (("skin",S),("fat",T)):
  off=[P[k]-OW[k] for k in range(3)]
  local=mv(M,off)
  for k,x in zip("xyz",local): r[f"{prefix}_local_{k}"]=format(LL["xyz".index(k)]+x,".17g")
 r["skin_m"]=format(math.sqrt(sum((O[k]-S[k])**2 for k in range(3))),".17g")
 r["fat_m"]=format(math.sqrt(sum((S[k]-T[k])**2 for k in range(3))),".17g")
with coords.open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=cols,lineterminator="\n");w.writeheader();w.writerows(rows)
# Verify the prepared outer coordinates are exactly the registered mechanical surface after source-relative deltas.
src_raw=list(map(int,(cand/"candidate.txt").read_text().split())); n,nf=src_raw[:2]; src=src_raw[2:2+n]
assert n==2332 and nf==4619 and len(rows)==n
assert [int(r["source_vertex"]) for r in rows]==src
with manifest.open() as f: body_records=json.load(f)["core_tree"]["source_body_records"]
body_names={int(x["core_body_index"]):str(x["name"]) for x in body_records}
hand_terms=("capitate","trapezium","trapezoid","hamate","lunate","scaphoid","pisiform","triquetrum","mc_","proxph","midph","distph","thumb")
def capclass(body):
 name=body_names.get(int(body),"").lower()
 return 1 if name=="head" or any(t in name for t in hand_terms) else 0
worldD=[];localD=[];caps=[]
for i,sid in enumerate(src):
 a=orig[sid]
 dw=[float(rows[i][f"outer_{axis}"])-float(a[k]) for axis,k in zip("xyz",("wx","wy","wz"))]
 dl=mv(frame(a),dw)
 worldD.append(dw);localD.append(dl);caps.append(capclass(int(a["donor"])))
 cap=0.002 if caps[-1] else 0.003
 if math.sqrt(sum(x*x for x in dw))>cap+1e-7: raise ValueError(f"source-relative displacement exceeds cap at {sid}")
 # App double addition must reconstruct the frozen emitted world point.
 for axis,k in zip("xyz",("wx","wy","wz")):
  if float(a[k])+dw["xyz".index(axis)] != float(rows[i][f"outer_{axis}"]):
   if abs((float(a[k])+dw["xyz".index(axis)])-float(rows[i][f"outer_{axis}"]))>2e-16:
    raise ValueError(f"mechanical delta does not reconstruct frozen outer point at {sid}")
# Direction values remain tied to the exact 028 max-margin smoothed LP field.
directions=[[float(r[f"outward_{a}"]) for a in "xyz"] for r in rows]
for d in directions:
 if abs(math.sqrt(sum(x*x for x in d))-1.0)>1e-5: raise ValueError("nonunit layer direction")
skin=[ [f32(r[f"skin_{a}"]) for a in "xyz"] for r in rows]
fat=[[f32(r[f"fat_{a}"]) for a in "xyz"] for r in rows]
outer=[[f32(r[f"outer_{a}"]) for a in "xyz"] for r in rows]
skinL=[[float(r[f"skin_local_{a}"]) for a in "xyz"] for r in rows]
fatL=[[float(r[f"fat_local_{a}"]) for a in "xyz"] for r in rows]
sd=[float(r["skin_m"]) for r in rows]; fd=[float(r["fat_m"]) for r in rows]
# The exact production prism decomposition, with deterministic sorted source rows and absolute volume.
raw=src_raw; off=2+n; faces=[raw[off+3*i:off+3*i+3] for i in range(nf)]
def sub(a,b):return [a[i]-b[i] for i in range(3)]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b):return sum(a[i]*b[i] for i in range(3))
def vol(a,b,c,d):return dot(sub(b,a),cross(sub(c,a),sub(d,a)))/6.0
volskin=volfat=0.0
for face in faces:
 a,b,c=sorted(face)
 volskin+=sum(abs(x) for x in (vol(outer[a],outer[b],outer[c],skin[c]),vol(outer[a],outer[b],skin[b],skin[c]),vol(outer[a],skin[a],skin[b],skin[c])))
 volfat+=sum(abs(x) for x in (vol(skin[a],skin[b],skin[c],fat[c]),vol(skin[a],skin[b],fat[b],fat[c]),vol(skin[a],fat[a],fat[b],fat[c])))
assert 0.0005<=min(sd) and max(sd)<=0.0025 and 0.0005<=min(fd) and max(fd)<=0.010
audit_data=json.loads(audit.read_text())
boundary_data=json.loads(boundary_audit.read_text())
full_stdout=full_gate/"stdout.log"; full_stderr=full_gate/"stderr.log"; full_binary=full_gate/"full_size_from_layers"
api_stdout=api_gate/"stdout.log"; api_stderr=api_gate/"stderr.log"; api_binary=api_gate/"from_layers_test"
assert audit_data["overall_float32_geometry_ok"] == 1
assert boundary_data["overall_ok"] is True
assert full_stdout.exists() and full_binary.exists() and api_stdout.exists()
full_text=full_stdout.read_text()
api_text=api_stdout.read_text()
assert "wholebody_FromLayers=PASS" in full_text and "vertices=2332 faces=4619 nodes=6996 tets=27714" in full_text
assert "reversed_winding_rejected=PASS" in api_text and "local_prism_orientation_rejected=PASS" in api_text and "variable_fields=PASS" in api_text
tested_header_sha=sha(out/"NumiHumanWholeBodySkinReduction.hpp")

receipt={
 "scope":"Whole-body coarse layered-surface geometry passed exact Float32 geometry, closed exposed-boundary, full-size FromLayers assembly, and negative API regression gates. Thickness fields remain bounded geometric construction parameters, not subject-specific measurements; this is not native evolution, anatomical, physiological, or clinical qualification.",
 "source_payload_sha256":"d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04",
 "registered_source_csv_sha256":sha(orig_csv),
 "mechanical_source_csv_sha256":sha(mechanical_csv),
 "topology_reduction_sha256":sha(cand/"candidate.txt"),
 "topology_header_sha256":sha(header_in),
 "topology_repair_receipt_sha256":sha(repair_receipt),
 "direction_csv_sha256":sha(direction_csv),
 "layer_coordinates_csv_sha256":sha(coords),
 "float32_geometry_audit_sha256":sha(audit),
 "compact_closed_boundary_audit_sha256":sha(boundary_audit),
 "compact_closed_boundary_audit":boundary_data,
 "full_size_from_layers_gate":{
   "status":"PASS","tested_header_sha256":tested_header_sha,
   "stdout_path":str(full_stdout),"stdout_sha256":sha(full_stdout),
   "stderr_path":str(full_stderr),"stderr_sha256":sha(full_stderr),
   "binary_path":str(full_binary),"binary_sha256":sha(full_binary),
   "nodes":6996,"tetrahedra":27714
 },
 "from_layers_negative_api_regression":{
   "status":"PASS","stdout_path":str(api_stdout),"stdout_sha256":sha(api_stdout),
   "stderr_path":str(api_stderr),"stderr_sha256":sha(api_stderr),
   "binary_path":str(api_binary),"binary_sha256":sha(api_binary),
   "coverage":"NaN summary, reversed winding, local prism orientation rejected transactionally; exact fit reuse, compile-bind, and variable fields passed"
 },
 "vertices":n,"faces":nf,"layer_tets_per_shell":3*nf,
 "float32_audit":audit_data,
 "thickness_bounds_m":{"skin":[0.0005,0.0025],"fat":[0.0005,0.010]},
 "thickness_actual_minmax_m":{"skin":[min(sd),max(sd)],"fat":[min(fd),max(fd)]},
 "exact_prism_volume_m3":{"skin":volskin,"fat":volfat},
 "layer_fit_method":"bounded direct-coordinate optimization from frozen 029 interior-target solution; 028 max-margin smoothed outward field; targeted donor-49 fat self-separation correction with 50 um SAT projection margin; outer and skin coordinates fixed during correction; exact Float32 source orientation and prism audit",
 "layer_provenance":{"skin":"Per-vertex geometric construction field constrained to 0.5-2.5 mm; inferred bounded geometry parameter, not subject-specific measured anatomy.","fat":"Per-vertex geometric construction field constrained to 0.5-10 mm; inferred bounded geometry parameter, not subject-specific measured anatomy."},
 "mechanical_delta_caps_m":{"head_hand":0.002,"other_body":0.003},
 "mechanical_delta_max_by_class_m":{"head_hand":0.0,"other_body":0.0},
 "admitted":True
}
for i,d in enumerate(worldD):
 value=math.sqrt(sum(x*x for x in d))
 k="head_hand" if caps[i] else "other_body"
 receipt["mechanical_delta_max_by_class_m"][k]=max(receipt["mechanical_delta_max_by_class_m"][k],value)
receipt_path=out/"layer-geometry-receipt.json";receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
receipt_sha=sha(receipt_path)
def fmt(x): return format(float(x),".17g")
def arr3(name,typ,vals):
 lines=[f"inline constexpr std::array<std::array<{typ},3>,{n}> {name}={{{{"]
 for i in range(0,n,3):
  chunk=vals[i:i+3]
  lines.append("    "+", ".join(f"std::array<{typ},3>{{"+",".join(fmt(v)+("f" if typ=="float" else "") for v in row)+"}" for row in chunk)+",")
 lines.append("}};")
 return "\n".join(lines)
def arr1(name,typ,vals):
 lines=[f"inline constexpr std::array<{typ},{n}> {name}={{"]
 for i in range(0,n,10):
  chunk=vals[i:i+10]
  lines.append("    "+", ".join((str(int(v))+"u") if typ=="std::uint32_t" else str(int(v)) if typ=="std::uint8_t" else fmt(v) for v in chunk)+",")
 lines.append("};")
 return "\n".join(lines)
h=header_in.read_text()
h=h.replace("#include <cstdint>","#include <cstdint>\n#include <string_view>",1)
# Replace only the two source-relative delta arrays with precise double arrays.
for name,vals in (("kWholeBodySkinMechanicalWorldDeltas",worldD),("kWholeBodySkinMechanicalDonorLocalDeltas",localD)):
 pattern=rf"inline constexpr std::array<std::array<float,3>,{n}> {name}=\{{.*?\}};"
 replacement=arr3(name,"double",vals)
 h2,count=re.subn(pattern,lambda _:replacement,h,flags=re.S)
 if count!=1: raise ValueError(f"expected one {name} definition, found {count}")
 h=h2
if "kWholeBodySkinMechanicalCapClass" in h: raise ValueError("cap class already present")
# Candidate header already has the reduction ancestry and pinned donor frames.
decls=[]
decls += [arr1("kWholeBodySkinMechanicalCapClass","std::uint8_t",caps)]
decls += [arr3("kWholeBodySkinOutwardDirections","double",directions)]
decls += [arr3("kWholeBodySkinInnerWorld","double",skin)]
decls += [arr3("kWholeBodySkinInnerLocal","double",skinL)]
decls += [arr3("kWholeBodyFatDeepWorld","double",fat)]
decls += [arr3("kWholeBodyFatDeepLocal","double",fatL)]
decls += [arr1("kWholeBodySkinThicknessM","double",sd)]
decls += [arr1("kWholeBodyFatThicknessM","double",fd)]
meta=f'''
inline constexpr bool kWholeBodySkinLayersAdmitted=true;
inline constexpr double kWholeBodySkinBodyDisplacementCapM=0.003;
inline constexpr double kWholeBodySkinHeadHandDisplacementCapM=0.002;
inline constexpr double kWholeBodySkinMinThicknessM=0.0005;
inline constexpr double kWholeBodySkinMaxThicknessM=0.0025;
inline constexpr double kWholeBodyFatMinThicknessM=0.0005;
inline constexpr double kWholeBodyFatMaxThicknessM=0.01;
inline constexpr double kWholeBodySkinLayerVolumeM3={fmt(volskin)};
inline constexpr double kWholeBodyFatLayerVolumeM3={fmt(volfat)};
inline constexpr std::string_view kWholeBodySkinSourceCSV_SHA256=kWholeBodySkinRegisteredCSV_SHA256;
inline constexpr std::string_view kWholeBodySkinReductionSHA256="{sha(cand/'candidate.txt')}";
inline constexpr std::string_view kWholeBodySkinMechanicalRepairSHA256="{sha(repair_receipt)}";
inline constexpr std::string_view kWholeBodySkinDirectionFieldSHA256="{sha(direction_csv)}";
inline constexpr std::string_view kWholeBodySkinLayerCoordinatesSHA256="{sha(coords)}";
inline constexpr std::string_view kWholeBodySkinLayerReceiptSHA256="{receipt_sha}";
inline constexpr std::string_view kWholeBodySkinLayerFitMethod="frozen bounded direct-coordinate fit; passed exact Float32 geometry, closed boundary, full-size FromLayers and negative API regression gates; no native evolution or anatomical qualification";
inline constexpr std::string_view kWholeBodySkinThicknessProvenance="geometric construction field constrained to 0.5-2.5 mm; not subject-specific measured anatomy";
inline constexpr std::string_view kWholeBodyFatThicknessProvenance="geometric construction field constrained to 0.5-10 mm; not subject-specific measured anatomy";
'''
decls.append(meta)
decls_text="\n".join(decls)+"\n"
marker="} // namespace metalrobo::human::coarse_tissue"
if marker not in h: raise ValueError("namespace marker missing")
h=h.replace(marker,decls_text+marker)
header_out=out/"NumiHumanWholeBodySkinReduction.hpp"
header_out.write_text(h)
# Keep the receipt hash non-circular; bind its and the header's hashes in a sidecar.
header_sha=sha(header_out)
manifest_out=out/"final-header-manifest.json"
manifest_out.write_text(json.dumps({"header_path":str(header_out),"header_sha256":header_sha,
 "layer_receipt_path":str(receipt_path),"layer_receipt_sha256":receipt_sha,
 "layer_coordinates_sha256":sha(coords)},indent=2)+"\n")
assert sha(receipt_path)==receipt_sha
print(json.dumps({"header":str(header_out),"header_sha256":header_sha,"manifest":str(manifest_out),"receipt":str(receipt_path),"receipt_sha256":receipt_sha,"coordinates_sha256":sha(coords),"vertices":n,"faces":nf,"skin_volume_m3":volskin,"fat_volume_m3":volfat,"skin_depth_minmax_m":[min(sd),max(sd)],"fat_depth_minmax_m":[min(fd),max(fd)],"capclass_counts":{"head_hand":sum(caps),"other":n-sum(caps)},"delta_max_by_class_m":receipt["mechanical_delta_max_by_class_m"]},indent=2))
