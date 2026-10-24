from pathlib import Path
import json,hashlib,shutil,subprocess,time,shlex
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");N=E/"native-candidate020-conforming-prism-030";A=N/"apps"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copytree(A,N/"apps-attempt-001")
shutil.copy2(N/"app-integration.diff",N/"app-integration-attempt-001.diff")
shutil.copy2(N/"candidate-identity.json",N/"candidate-identity-attempt-001.json")
p=A/"numilab_human_myosim_visual_probe.mm";s=p.read_text()
old='                identityTopology.childDonorBodyIndices=identityTetDonor;'
new='''                // Element region provenance follows its measured dominant old-volume
                // parent. Nodal donor frames and physical lumped masses remain separate.
                for(std::size_t ti=0;ti<tetCount;++ti)
                    identityTetDonor[ti]=parentTetDonor[prism030::dominantParent[ti]];
                identityTopology.childDonorBodyIndices=identityTetDonor;'''
assert s.count(old)==1;s=s.replace(old,new)
ip=N/"candidate-identity.json";oldsha=sha(ip);identity=json.loads(ip.read_text())
identity["dominant_cell_region_policy"]="largest exact-overlap parent; node donors/mass remain existing source mapping"
ip.write_text(json.dumps(identity,indent=2,sort_keys=True)+"\n");s=s.replace(oldsha,sha(ip));p.write_text(s)
(N/"dominant-parent-donor-fix.json").write_text(json.dumps({"reason":"source gate correctly rejected new-node-majority region inconsistent with dominant parent at11789","old_run":"source-preflight-001","new_policy":identity["dominant_cell_region_policy"],"old_source_snapshot":"apps-attempt-001","physical_nodal_donors_unchanged":True},indent=2))
base=(E/"native-candidate020-mechanical-source-024/apps"/p.name).read_text()
import difflib
(N/"app-integration.diff").write_text("".join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile="024/app.mm",tofile="030/app.mm")))
B=N/"build-002";B.mkdir()
for label in ("compile","link"):
 cmd=json.loads((N/"build-001"/(label+"-command.json")).read_text())
 cmd=[str(B/Path(x).name) if x.startswith(str(N/"build-001")+"/") else x for x in cmd]
 (B/(label+"-command.json")).write_text(json.dumps(cmd,indent=2))
 t=time.monotonic()
 with(B/(label+".log")).open("wb") as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (B/(label+"-result.json")).write_text(json.dumps({"returncode":r.returncode,"wall_seconds":time.monotonic()-t}))
 print(label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
deps=shlex.split((B/"candidate.d").read_text().split(":",1)[1].replace("\\\n",""))
for p in A.iterdir():
 if p.is_file() and p.suffix==".hpp":
  assert [Path(x) for x in deps if x.endswith("/"+p.name)]==[p],p
(B/"dependency-certification.json").write_text(json.dumps({"status":"PASS_EXACT_OVERLAY","input_hashes":{x:sha(x) for x in deps if Path(x).is_file() and x.startswith("/Users/")},"binary_sha256":sha(B/"numi-human-native"),"physical_qualification":False},indent=2))
run=(N/"run_source_preflight.py").read_text().replace("build-001","build-002").replace("source-preflight-001","source-preflight-002")
(N/"run_source_preflight_002.py").write_text(run)
raise SystemExit(subprocess.run(["python3",str(N/"run_source_preflight_002.py")]).returncode)
