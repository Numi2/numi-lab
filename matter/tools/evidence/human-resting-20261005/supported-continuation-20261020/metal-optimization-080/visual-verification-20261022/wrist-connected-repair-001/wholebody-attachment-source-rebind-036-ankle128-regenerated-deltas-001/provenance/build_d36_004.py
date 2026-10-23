from pathlib import Path
import subprocess,json,hashlib,time
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";D35=E/"wholebody-attachment-source-rebind-035-knee100-regenerated-deltas-001";D=E/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001"
old=D35/"build-main-de561-002";build=D/"build-main-0a693-004";build.mkdir(exist_ok=False);base=Path("/tmp/numi-main-0a693-d36-base");overlay=D/"source-overlay/apps";sha=lambda b:hashlib.sha256(b).hexdigest();results={}
for stage in ["compile","link"]:
 cmd=json.loads((old/(stage+"-command.json")).read_text());out=[]
 for arg in cmd:
  for a,b in [(str(old),str(build)),(str(D35/"source-overlay/apps"),str(overlay)),("/private/tmp/numi-main-de561-d35-base",str(base))]:arg=str(arg).replace(a,b)
  out.append(arg)
 (build/(stage+"-command.json")).write_text(json.dumps(out,indent=2)+"\n")
 t=time.monotonic();run=subprocess.run(out,cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (build/(stage+".stdout.log")).write_bytes(run.stdout);(build/(stage+".stderr.log")).write_bytes(run.stderr);rr={"returncode":run.returncode,"seconds":time.monotonic()-t,"stdout_sha256":sha(run.stdout),"stderr_sha256":sha(run.stderr)};results[stage]=rr
 (build/(stage+"-result.json")).write_text(json.dumps(rr,indent=2)+"\n");print(stage,json.dumps(rr),flush=True)
 if run.returncode:raise SystemExit(run.stderr.decode()[-10000:])
manifest={"base_commit":"0a693f20c0d22eadb993b1fe8e2f627126445f1c","binary_sha256":sha((build/"numi-human-native").read_bytes()),"candidate_identity_sha256":sha((D/"D36-candidate-identity.json").read_bytes()),"candidate_payload_sha256":sha((E/"left-ankle-rest-preserving-weight-repair-128/candidate.nhskin").read_bytes()),"strict_werror":True,"source_overlay_hashes":{p.name:sha(p.read_bytes()) for p in sorted(overlay.iterdir()) if p.is_file()},"host_trace_sha256":sha((base/"src/metal/HumanHostPhaseTrace.hpp").read_bytes()),"results":results,"physical_qualification":False}
(D/"D36-build-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest),flush=True)
