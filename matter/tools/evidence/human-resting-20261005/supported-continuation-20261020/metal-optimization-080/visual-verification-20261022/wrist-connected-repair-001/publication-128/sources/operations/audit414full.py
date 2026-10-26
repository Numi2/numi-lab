from pathlib import Path
import json,subprocess,time,hashlib
d=Path('/Users/numi/Work/numi-human-connected-surface-069');p=d/'D/local-two-parent-retriangulation-414/run-001-pass002';c=p/'search/selected/candidate.npz';out=p/'full-audits';out.mkdir(exist_ok=False)
py='/Users/n/numi-human-prep-venv-20261005/bin/python';sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert sha(d/'audit_connected_outer_geometry_130.py')=='52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9'
assert sha(d/'audit_skin_bone_surfaces_159.py')=='de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1'
start=time.monotonic();rs=[]
for script,dest in [(d/'audit_connected_outer_geometry_130.py',out/'audit-130'),(d/'audit_skin_bone_surfaces_159.py',out/'audit-159')]:
 a=[py,'/tmp/numi-human-geometry-slot.py',py,str(script),str(c),str(dest)]
 with(out/(dest.name+'.log')).open('x')as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
 rs.append(dict(argv=a,returncode=r.returncode));assert r.returncode==0
 if dest.name=='audit-130':
  j=json.load(open(dest/'summary.json'));assert j['complete_surface_audit'] and j['exact_PLC_surface_pairs']==0
j=json.load(open(out/'audit-130/summary.json'));s=json.load(open(out/'audit-159/report.json'));assert s['complete'] and len(s['banks'])==185
v=dict(candidate=str(c),candidate_sha256=sha(c),runs=rs,wall_seconds=time.monotonic()-start,self_pairs=j['exact_PLC_surface_pairs'],inside_pairs=j['inside_node_bank_pairs'],surface_pairs=sum((b.get('result')or{}).get('count',0)for b in s['banks']),point_summary_sha256=sha(out/'audit-130/summary.json'),surface_report_sha256=sha(out/'audit-159/report.json'),native_admission=False,accepted_physical_steps=0)
(out/'execution.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))

