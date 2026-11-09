from pathlib import Path
import hashlib,json,subprocess,tempfile
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report=[]
with tempfile.TemporaryDirectory(prefix='crow-shader-reconstruct-',dir=B) as t:
 prev=R/'src/metal/LocomotionTask.metal'
 for patch_name,existing_name in [('feedback_live.patch','feedback_live.metal'),('walk-handoff.patch','walk-handoff.metal'),('boundary-handoff.patch','boundary-handoff.metal')]:
  out=Path(t)/existing_name;subprocess.run(['patch','--batch','-o',str(out),str(prev),str(A/patch_name)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert h(out)==h(B/existing_name),(patch_name,h(out),h(B/existing_name));report.append({'patch':patch_name,'reconstructed_sha256':h(out),'matches_built_source':True});prev=out
(A/'shader-source-reconstruction.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
