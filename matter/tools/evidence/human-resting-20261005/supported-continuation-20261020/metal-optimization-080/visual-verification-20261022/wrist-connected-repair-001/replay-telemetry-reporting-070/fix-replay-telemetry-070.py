from pathlib import Path
import subprocess,json,hashlib,time
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";d=E/"elbow-refinement-current-main-019-dense-execution-face-identities-001";out=E/"replay-telemetry-reporting-070";out.mkdir(exist_ok=False)
old='<< (result.deterministicReplayVerified ? "bitwise" : "failed")';new='<< (result.deterministicReplayVerified ? "bitwise" : (verifyDeterminism ? "failed" : "not_requested"))'
def patch(s):
 assert s.count(old)==1
 return s.replace(old,new)
p=R/"apps/numilab_human_myosim_visual_probe.mm";before=p.read_bytes();p.write_text(patch(before.decode()))
base=subprocess.check_output(["git","-C",str(R),"show","4fdd8a2168616b0164e6e3dd76074da8c516e13f:apps/numilab_human_myosim_visual_probe.mm"],text=True)
(out/"main-source").mkdir();(out/"main-source/numilab_human_myosim_visual_probe.mm").write_text(patch(base))
(out/"compile").mkdir();src=out/"compile/numilab_human_myosim_visual_probe.mm";src.write_text(patch((d/"apps/numilab_human_myosim_visual_probe.mm").read_text()))
argv=json.loads((d/"compile-019-attempt-01/compile-command.json").read_text());argv=[str(src) if x==str(d/"apps/numilab_human_myosim_visual_probe.mm") else x.replace(str(d/"compile-019-attempt-01"),str(out/"compile")) for x in argv]
(out/"compile/command.json").write_text(json.dumps(argv,indent=2)+"\n");t=time.monotonic()
with (out/"compile/stdout.txt").open("w") as stdout,(out/"compile/stderr.txt").open("w") as stderr:r=subprocess.run(argv,cwd=R,stdout=stdout,stderr=stderr)
receipt={"returncode":r.returncode,"elapsed_s":time.monotonic()-t,"reporting_change_only":True,"not_requested_run_did_not_execute_replay":True,"verified_run_status":"bitwise","requested_unverified_status":"failed","unrequested_unverified_status":"not_requested","native_run_not_repeated":True,"working_source_before_sha256":hashlib.sha256(before).hexdigest(),"working_source_after_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest()}
(out/"result.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt));assert r.returncode==0

