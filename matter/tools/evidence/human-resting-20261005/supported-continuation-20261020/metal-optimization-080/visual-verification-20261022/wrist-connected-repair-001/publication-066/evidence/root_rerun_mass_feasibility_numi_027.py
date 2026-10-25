from pathlib import Path
import subprocess,json,hashlib,platform
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");P=E/'o12-outward-repair-027-f32-gated/mass-feasibility-001'
frozen=json.loads((P/'audit-report027-frozen.json').read_text());cmd=["/Users/n/numi-human-prep-venv-20261005/bin/python",str(P/'audit_mass_027.py')]
for key,row in frozen['pinned_inputs'].items():
 p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],key
 cmd+=['--'+key,str(p)]
out=P/'root-ssh-numi-report.json';assert not out.exists();cmd+=['--output',str(out)]
r=subprocess.run(cmd,capture_output=True,text=True)
(P/'root-ssh-numi-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(P/'root-ssh-numi.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
actual=json.loads(out.read_text())
for key in ['totals','material_geometry','row_mapping','donor_capacity','configured_limits']:assert actual[key]==frozen[key],key
receipt={'status':'SSH_NUMI_RERUN_EXACT_SUMMARY_PARITY_PASS','hostname':platform.node(),'device':platform.platform(),'input_count':len(frozen['pinned_inputs']),'physical_qualification':False,'accepted_physical_steps':0,'report_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((P/'audit_mass_027.py').read_bytes()).hexdigest(),'J_violations':actual['totals']['J_violations']}
(P/'root-ssh-numi-parity.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

