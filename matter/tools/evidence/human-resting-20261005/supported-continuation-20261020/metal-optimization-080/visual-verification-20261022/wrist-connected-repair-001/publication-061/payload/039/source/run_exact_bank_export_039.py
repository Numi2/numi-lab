#!/usr/bin/env python3
"""Export exact in-memory D37 banks for the frozen 038 source-only candidate audit."""
import csv,hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=ROOT/"bone-clearance-fit-038-exact-bank-export-039"
BASE=ROOT/"native-candidate020-all-bank-containment-035/source-preflight-003"
CANDIDATE=ROOT/"bone-clearance-fit-038-candidate-query/run-002/candidate-node-override.csv"
CHECKED=ROOT/"native-candidate020-bone-witness-028/run_native_contact_checked.py"
APP=D/"build-001/numi-human-native"
RUNTIME=ROOT/"native-app-main-header-integration-151/build-runtime-004/libmetalrobo.dylib"
METALLIB=Path("/tmp/numi-human-joint-owner-build-033/matter/shaders/NumiMatter.metallib")
APPS=D/"apps"
EXPECTED_CANDIDATE="058db5c144a48d84670963081e2b1a6cfab826d79ec098e19eefd07d14841681"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def f32bits(value):
 import struct
 return struct.unpack("<I",struct.pack("<f",float(value)))[0]
def rotate(q,p):
 axis=q[:3]
 c=(axis[1]*p[2]-axis[2]*p[1],axis[2]*p[0]-axis[0]*p[2],axis[0]*p[1]-axis[1]*p[0])
 t=tuple(2*x for x in c)
 d=(axis[1]*t[2]-axis[2]*t[1],axis[2]*t[0]-axis[0]*t[2],axis[0]*t[1]-axis[1]*t[0])
 return tuple(p[i]+q[3]*t[i]+d[i] for i in range(3))
def check_outputs(run):
 tri=run/"exact-emitted.triangles.csv"; nodes=run/"exact-emitted.nodes.csv"; bodies=run/"exact-emitted.bodies.csv"
 bankcsv=run/"all-bank-containment.csv.banks.csv"; nodecsv=run/"all-bank-containment.csv.nodes.csv"
 for p in (tri,nodes,bodies,bankcsv,nodecsv): require(p.is_file() and p.stat().st_size>0,f"missing/empty: {p}")
 expected={}
 with bankcsv.open(newline="") as f:
  for r in csv.DictReader(f): expected[(int(r["body"]),int(r["stable_bone_id"]))]=int(r["triangle_count"])
 observed={};count=0
 with tri.open(newline="") as f:
  reader=csv.DictReader(f)
  required={"body_index","bone_stable_id","source_record_index","triangle_stable_id","v0x_bits_u32","v2z_bits_u32"}
  require(required.issubset(reader.fieldnames or []),"triangle export schema mismatch")
  for r in reader:
   key=(int(r["body_index"]),int(r["bone_stable_id"]))
   require(int(r["source_record_index"])>=0 and int(r["triangle_stable_id"])>=0,"missing native triangle identity")
   for name in ("v0x_m","v0y_m","v0z_m","v1x_m","v1y_m","v1z_m","v2x_m","v2y_m","v2z_m"):
    require(f32bits(r[name])==int(r[name.replace("_m","_bits_u32")]),f"F32 triangle value/bits mismatch {name}")
   observed[key]=observed.get(key,0)+1;count+=1
 require(observed==expected,f"emitted triangles differ from exact 038 queried inventory: {len(observed)} vs {len(expected)} banks")
 require(len(observed)==185,"expected all 185 complete per-bone banks")
 base=BASE/"initial-skin-fem-nodes.csv"
 with base.open(newline="") as f: base_rows={int(r["global_node"]):r for r in csv.DictReader(f)}
 with CANDIDATE.open(newline="") as f: cand_rows={int(r["global_node"]):r for r in csv.DictReader(f)}
 require(len(cand_rows)==7233,"candidate override is not complete 7,233 node set")
 rowcount=0
 with nodes.open(newline="") as f:
  reader=csv.DictReader(f);require(len(reader.fieldnames or [])>=13,"node export schema mismatch")
  for r in reader:
   g=int(r["global_node"]);require(g in base_rows and g in cand_rows,f"unexpected node ID {g}")
   c=cand_rows[g];b=base_rows[g]
   require((int(r["layer"]),int(r["row"]))==(int(c["layer"]),int(c["row"])),f"node ancestry mismatch {g}")
   for axis in "xyz": require(int(r[f"{axis}_bits_u32"])==f32bits(c[f"{axis}_m"]),f"node candidate value mismatch {g}:{axis}")
   require(int(r["mass_bits_u32"])==f32bits(b["mass_kg"]),f"node mass changed {g}")
   rowcount+=1
 require(rowcount==7233,"expected 7,233 native candidate-node rows")
 with bodies.open(newline="") as f:
  reader=csv.DictReader(f);body_rows=list(reader)
 require(body_rows and len(body_rows)==len({r["body_index"] for r in body_rows}),"body transform table is empty or duplicated")
 for r in body_rows:
  q=[float(r[f"orientation_q{k}"]) for k in ("x","y","z","w")]
  qi=(-q[0],-q[1],-q[2],q[3])
  mat=[float(r[f"world_to_body_m{i}{j}"]) for i in range(3) for j in range(3)]
  columns=[rotate(qi,tuple(float(c==axis) for c in range(3))) for axis in range(3)]
  for i in range(3):
   for j in range(3): require(abs(mat[3*i+j]-columns[j][i])<=2e-15,"double inverse quaternion matrix mismatch")
 return {"triangles":count,"banks":len(observed),"nodes":rowcount,"body_transforms":len(body_rows),
         "triangle_sha256":sha(tri),"node_sha256":sha(nodes),"body_sha256":sha(bodies),
         "bank_inventory_sha256":sha(bankcsv),"node_classification_sha256":sha(nodecsv)}
def main():
 for p in (BASE/"launch-manifest.snapshot.json",BASE/"initial-skin-fem-nodes.csv",CANDIDATE,CHECKED,APP,RUNTIME,METALLIB):
  require(p.is_file() and not p.is_symlink(),f"required input missing or redirected: {p}")
 require(sha(CANDIDATE)==EXPECTED_CANDIDATE,"038 candidate override hash changed")
 run=D/"run-002/source-preflight"
 require(not run.exists(),"run output must be new")
 base_sha=sha(BASE/"launch-manifest.snapshot.json");candidate_sha=sha(CANDIDATE)
 base=json.loads((BASE/"launch-manifest.snapshot.json").read_text())
 base_env=dict(base["environment"])
 base_env.pop("NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV",None)
 base["environment"]=base_env
 base["scope"]="derived source-only all-bank audit; legacy nearest-only witness disabled; no simulation"
 derived_base=D/"run-002/base-launch-manifest.json"
 derived_base.parent.mkdir(exist_ok=True)
 derived_base.write_text(json.dumps(base,indent=2,sort_keys=True)+"\n")
 derived_base_sha=sha(derived_base)
 env=[f"NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_CSV={CANDIDATE}",
      f"NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_SHA256={candidate_sha}",
      f"NUMI_HUMAN_INITIAL_SKIN_BONE_ALL_BANK_CSV={run/'all-bank-containment.csv'}",
      f"NUMI_HUMAN_D37_EXACT_EMITTED_BANK_DUMP_PREFIX={run/'exact-emitted'}"]
 command=[sys.executable,str(CHECKED),"--base-manifest",str(derived_base),
  "--app",str(APP),"--runtime",str(RUNTIME),"--metallib",str(METALLIB),"--run-dir",str(run),
  "--source-root",str(APPS),"--mode","source-preflight","--static-bvh"]
 for item in env: command += ["--env",item]
 parent=run.parent;parent.mkdir(exist_ok=True)
 proc=subprocess.run(command,text=True,capture_output=True)
 (parent/"checked-runner.stdout.txt").write_text(proc.stdout)
 (parent/"checked-runner.stderr.txt").write_text(proc.stderr)
 require(proc.returncode==0,f"checked source-only runner failed {proc.returncode}; see preserved logs")
 receipt=json.loads((run/"execution-receipt.json").read_text())
 require(receipt.get("status")=="SOURCE_ONLY_INTENTIONAL_STOP" and receipt.get("physical_qualification") is False,
         "run is not a source-only intentional stop")
 stdout=(run/"stdout.log").read_text()
 require("d37_candidate_node_override=hash_verified" in stdout,"candidate override not verified by app")
 require("d37_initial_skin_bone_all_bank=written" in stdout,"all-bank owner did not execute")
 require("d37_exact_emitted_bank_dump=written" in stdout,"exact dump hook did not execute")
 result=check_outputs(run)
 result.update({"schema":"numi.human.d37-exact-emitted-bank-export.v1",
  "source_app_sha256":sha(APP),"source_header_sha256":sha(APPS/"NumiHumanD37ExactBankDump.hpp"),
  "candidate_override_sha256":candidate_sha,"base_manifest_sha256":base_sha,
  "derived_base_manifest_sha256":derived_base_sha,
  "pose_snapshot_sha256":sha(run/"registered-skin-poses.json"),
  "query_triangle_frame":"native body-local Float32 vertices after exact COM-offset shift; these exact records feed classifyClosedBank",
  "query_point_frame":"double R(q_F32 conjugate)*(world_F32 - body_position_F32) - exact COM offset",
  "body_matrix":"double inverse-quaternion matrix built with the same witness::rotate routine from raw promoted Float32 quaternion; raw Float32 pose bits also exported",
  "source_record_mapping":"body_index + bone_stable_id to unique verified BoneOwner sourceRecordIndex; member string is not present in this owner record",
  "native_simulation_steps":0,"physical_qualification":False})
 out=run/"export-report.json";out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
