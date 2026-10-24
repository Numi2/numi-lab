from pathlib import Path
import json,hashlib,subprocess,time,shlex,difflib,sys,shutil
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";A=N/"apps"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
number=sys.argv[1];B=N/("build-"+number);B.mkdir()
if len(sys.argv)>2:
 P=Path(sys.argv[2])
 for name in ("NumiHumanSkinFatReductionLabels.hpp","NumiHumanStableDonorBasis.hpp"):
  if (P/name).exists():shutil.copy2(P/name,A/name)
ip=N/"candidate-identity.json";oldsha=sha(ip);identity=json.loads(ip.read_text())
identity["parent_region_policy"]="exact baseline paired source labels; changed child inherits maximum-overlap parent's donor and basis ID"
identity["region_label_owner_sha256"]=sha(A/"NumiHumanSkinFatReductionLabels.hpp")
if (A/"NumiHumanStableDonorBasis.hpp").exists():identity["stable_donor_basis_helper_sha256"]=sha(A/"NumiHumanStableDonorBasis.hpp")
ip.write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n")
p=A/"numilab_human_myosim_visual_probe.mm";s=p.read_text()
assert s.count(oldsha)==1
s=s.replace(oldsha,sha(ip));p.write_text(s)
base=(E/"native-candidate020-mechanical-source-024/apps"/p.name).read_text()
(N/"app-integration.diff").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="024/app.mm",tofile="030/app.mm")))
for label in ("compile","link"):
 cmd=json.loads((N/"build-003"/(label+"-command.json")).read_text())
 cmd=[str(B/Path(x).name) if x.startswith(str(N/"build-003")+"/") else x for x in cmd]
 (B/(label+"-command.json")).write_text(json.dumps(cmd,indent=2))
 now=time.monotonic()
 with(B/(label+".log")).open("wb") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (B/(label+"-result.json")).write_text(json.dumps({"returncode":r.returncode,"wall_seconds":time.monotonic()-now}))
 print(label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
deps=shlex.split((B/"candidate.d").read_text().split(":",1)[1].replace("\\\n",""))
for p in A.iterdir():
 if p.is_file() and p.suffix==".hpp":assert [Path(z) for z in deps if z.endswith("/"+p.name)]==[p],p
(B/"dependency-certification.json").write_text(json.dumps({"status":"PASS_EXACT_OVERLAY","input_hashes":{z:sha(z) for z in deps if Path(z).is_file() and z.startswith("/Users/")},"binary_sha256":sha(B/"numi-human-native"),"physical_qualification":False},indent=2))
s=(N/"run_source_preflight.py").read_text().replace("build-001","build-"+number).replace("source-preflight-001","source-preflight-"+number)
rp=N/("run_source_preflight_"+number+".py");rp.write_text(s)
raise SystemExit(subprocess.run(["python3",str(rp)]).returncode)
