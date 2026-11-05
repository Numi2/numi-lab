"""Rebuild current P1 support using the existing owner and its exact-F32 cache path."""
from pathlib import Path
import json,sys,subprocess,hashlib,time
W=Path("/Users/numi/Work/human-integrated-gaps-20261104")
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
C=W/"joint-boundary-rebuild-64c/compiler"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {"path":str(Path(p).resolve()),"sha256":sha(p)}
def checked(p):
 assert sha(p["path"])==p["sha256"],p
 return p
owner=H/"current-role-p1-support-938-1027-pinned-230-dependency/build_current_role_p1_143_support_1027_pinned_230_dependency.py"
for role in ("Skin","Fat"):
 start=time.time()
 olddir=H/f"current-role-p1-support-938-1027-pinned-230-dependency/actual-001/p1-{role}"
 oldpath=olddir/"receipt.json";r=json.loads(oldpath.read_text())
 assert r["schema"]=="numi.human.current-role-p1-semantic-onehop-support.v1"
 assert r["status"]==f"PASS_SOURCE_ONLY_CURRENT_{role.upper()}_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT"
 assert r["physical_admission"] is False and r["native_steps"]==0 and r["mass_transport"]=="NOT_RUN"
 # Adapt only the old owner's receipt spelling to the existing cache spelling.
 # Every derived field is authenticated against its source receipt and real file.
 volume=checked(r["current_volume"]);support=checked(r["outputs"]["support_npz"])
 origin=checked(r["source330_mass"]);semantic=checked(r["source230_owner"])
 assert r["source330_parent_identity"]["source_owner_sha256"]==semantic["sha256"]
 assert r["source330_parent_identity"]["mass_csv_sha256"]==origin["sha256"]
 sidecars={"massSHA256":"","nodeSHA256":".nodes.csv","tetsSHA256":".tets.csv","labelsSHA256":".labels.csv"}
 hashes={semantic["path"]:semantic["sha256"]}
 for key,suffix in sidecars.items():
  p=Path(origin["path"]+suffix);assert sha(p)==r["source330"][key]
  hashes[str(p)]=r["source330"][key]
 cache={"schema":r["schema"],"physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN",
 "source_receipt":pin(oldpath),"geometry_target":volume,"outputs":{"p1_143":support},
 "inputs":{"semantic_owner":semantic,"source330_mass":origin},"input_sha256":hashes}
 cp=C/f"{role.lower()}-cache-receipt.json";cp.write_text(json.dumps(cache,indent=2)+"\n")
 cmd=[sys.executable,str(owner),"--role",role,
 "--current-state",str(C/"volume-prefix/verified-state.json"),
 "--volume-npz",str(C/f"canonical/{role.lower()}-canonical-parent-union.npz"),
 "--canonical-834-report",str(C/"canonical834-report.json"),
 "--role-ele",str(C/f"handoff938/{role.lower()}.ele"),
 "--handoff-receipt",str(C/"handoff938/adapter-receipt.json"),
 "--cache-support-npz",support["path"],"--cache-volume-npz",volume["path"],"--cache-receipt",str(cp),
 "--output-dir",str(C/f"p1-{role}")]
 print("START",role,flush=True)
 with (C/f"p1-{role}.stdout.log").open("w") as o,(C/f"p1-{role}.stderr.log").open("w") as e:done=subprocess.run(cmd,stdout=o,stderr=e)
 (C/f"p1-{role}.command.json").write_text(json.dumps({"command":cmd,"returncode":done.returncode,"wall_s":time.time()-start,"producer":pin(__file__),"owner":pin(owner)},indent=2)+"\n")
 print("END",role,done.returncode,flush=True)
 if done.returncode:raise RuntimeError((C/f"p1-{role}.stderr.log").read_text()[-4500:])
