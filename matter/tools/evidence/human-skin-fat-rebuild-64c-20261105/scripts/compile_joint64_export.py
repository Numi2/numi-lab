"""Continue the existing current-domain P1 and rich-433 compiler for candidate64c."""
from pathlib import Path
import json,sys,subprocess,hashlib,time
W=Path("/Users/numi/Work/human-integrated-gaps-20261104")
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=W/"joint-boundary-rebuild-64c";C=D/"compiler";hand=C/"handoff938";out=C/"p1-export"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {"path":str(Path(p).resolve()),"sha256":sha(p)}
def load(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+"\n")
def run(name,args):
 cmd=[sys.executable,*map(str,args)];start=time.time();print("START",name,flush=True)
 with (C/(name+".stdout.log")).open("w") as o,(C/(name+".stderr.log")).open("w") as e:r=subprocess.run(cmd,stdout=o,stderr=e)
 save(C/(name+".command.json"),{"command":cmd,"returncode":r.returncode,"wall_s":time.time()-start})
 print("END",name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError((name,(C/(name+".stderr.log")).read_text()[-4500:]))
config=load(H/"current-role-p1-support-938-1027-pinned-230-dependency/actual-001/current-domain-pins.json")
config["current_state"]=pin(C/"volume-prefix/verified-state.json")
files={"combined_node":"skin-fat-combined.node","combined_ele":"skin-fat-combined.ele","combined_node_map":"skin-fat-combined-node-map.csv.gz","element_map":"skin-fat-combined-element-map.csv.gz","handoff_receipt":"adapter-receipt.json"}
config["combined"]={k:pin(hand/f) for k,f in files.items()}
for role in ("Skin","Fat"):
 lr=role.lower();r=load(C/f"p1-{role}/receipt.json");assert r["status"].startswith("PASS_")
 e=config["roles"][lr]
 e.update(current_volume=pin(C/f"canonical/{lr}-canonical-parent-union.npz"),cell_count=r["counts"]["cells"],node=pin(hand/f"{lr}.node"),ele=pin(hand/f"{lr}.ele"),support_npz=r["outputs"]["support_npz"],support_receipt=pin(C/f"p1-{role}/receipt.json"))
 config["counts"][f"{lr}_cells"]=r["counts"]["cells"]
config["counts"]["combined_cells"]=sum(config["counts"][f"{r}_cells"] for r in ("skin","fat"))
cp=C/"current-domain-pins.json";save(cp,config)
src=config["source"];args=[
 H/"current-role-p1-support-938-1028-active-node-domain/current_domain_p1_export_937_active_node_domain_1028.py",
 "--current-domain-pins",cp,"--current-domain-pins-sha256",sha(cp),"--",
 "--parent-mass-csv",src["parent_mass_csv"]["path"],"--parent-full86-weights-csv",src["full86_weights"]["path"],
 "--child-node",hand/"skin-fat-combined.node","--child-ele",hand/"skin-fat-combined.ele",
 "--output-labels-csv",out/"child.labels.csv","--source-map-csv",out/"child.labels.csv.source.csv","--receipt-json",out/"receipt.json",
 "--capacity-profile","full-reference-500k-nodes-2m-cells","--reference-registration-max-distance-m","0.02","--reference-registration-ambiguity-tolerance-m","1e-8","--p1-supported-reference-anchor"]
for role in ("Skin","Fat"):
 lr=role.lower();args+=["--p1-"+lr+"-support-npz",C/f"p1-{role}/{lr}-current-p1-143-support.npz","--p1-"+lr+"-support-receipt-json",C/f"p1-{role}/receipt.json"]
args+=["--combined-element-map-csv-gz",hand/"skin-fat-combined-element-map.csv.gz","--combined-node-map-csv-gz",hand/"skin-fat-combined-node-map.csv.gz","--combined-handoff-receipt-json",hand/"adapter-receipt.json","--anchor-csr-npz",out/"p1-anchor.npz","--anchor-projection-csv",out/"p1-anchor-projection.csv","--projection-helper-path",src["projection_helper"]["path"],"--projection-helper-sha256",src["projection_helper"]["sha256"],"--collect-rejection-census","--rejection-witness-json",out/"rejection-witness.json"]
run("p1-export",args)
run("labels850",[H/"canonical-volume-parent-unions-849-nativepreflight/p1_projection_labels_850.py","--projection",out/"p1-anchor-projection.csv","--receipt",out/"receipt.json","--output-labels",out/"child.labels-850.csv","--report",out/"labels850-report.json"])
old=load(H/"current-native-1024-root/current-433-1037/launch-inputs.json")
args=old["argv"][3:] # interpreter and run_geometry_serial.py removed; run() supplies interpreter.
replace={
 "--current-surface-node-map":[str(hand/"skin-fat-combined-registered-surface-node-map.npz"),sha(hand/"skin-fat-combined-registered-surface-node-map.npz")],
 "--child-node":[str(hand/"skin-fat-combined.node")],"--child-ele":[str(hand/"skin-fat-combined.ele")],
 "--child-labels":[str(out/"child.labels-850.csv")],"--child-label-source":[str(out/"child.labels.csv.source.csv")],
 "--p1-anchor-csr":[str(out/"p1-anchor.npz")],"--p1-anchor-receipt":[str(out/"receipt.json")],"--p1-anchor-projection-csv":[str(out/"p1-anchor-projection.csv")],
 "--child-node-map":[str(hand/"skin-fat-combined-node-map.csv.gz")],
 "--output-npz":[str(C/"rich433/candidate-current-typed-volume.npz")],
 "--output-node-map":[str(C/"rich433/candidate-node-map.csv")],
 "--output-ancestry":[str(C/"rich433/inserted-node-ancestry.csv")],"--output-receipt":[str(C/"rich433/receipt.json")]}
for flag,values in replace.items():
 i=args.index(flag)+1;args[i:i+len(values)]=values
for role in "OID":
 i=next(i for i,x in enumerate(args) if x=="--current-typed-role" and args[i+1]==role)
 args[i+2:i+4]=[str(D/f"candidate-{role}-typed-current.npz"),sha(D/f"candidate-{role}-typed-current.npz")]
(C/"rich433").mkdir(exist_ok=False)
run("rich433",args)
print("RICH_VOLUME_COMPILER_COMPLETE_MASS_NOT_RUN",flush=True)
