from pathlib import Path
import json,hashlib,subprocess,time,numpy as np,importlib.util
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';r=H/'bone-aware-remesh-370/run-002-self-safe-subset';c=r/'candidate.npz';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(c)=='d60b58c1babc739a4a0bc53ad43c4c0623d0181a8b80dc2ac00a02d6ba11b630'
z=np.load(c,allow_pickle=False);b=np.load(H/'same-parent-remesh-368/run-001/candidate.npz',allow_pickle=False)
assert np.array_equal(z['points_world_m'].view(np.uint32),b['points_world_m'].view(np.uint32));assert np.array_equal(z['source_O_faces'],z['area_reference_037_source_O_faces'])
f=D/'D/connected-local-witness-proposal-365/propose_joint_clearance_witnesses_365.py';sp=importlib.util.spec_from_file_location('gate368',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
area=m.full_f32_area_gate(z['points_world_m'],z['points_world_m'].astype(float)*1000,z['area_reference_037_points_world_m_f64']*1000,z['source_O_faces'],.1);assert area['pass'],area
(r/'root-preparation-gates.json').write_text(json.dumps(dict(candidate_sha256=sha(c),points_bitwise_unchanged=True,matched_reference_topology=True,area=area,native_admission=False,accepted_physical_steps=0),indent=2)+'\n')
for name,out in [('audit_connected_outer_geometry_130.py','audit-130'),('audit_skin_bone_surfaces_159.py','audit-159')]:
 cmd=['/Users/n/numi-human-prep-venv-20261005/bin/python','/tmp/numi-human-geometry-slot.py','/Users/n/numi-human-prep-venv-20261005/bin/python',str(D/name),str(c),str(r/out)]
 if out=='audit-159':cmd+=['--max-seconds','600']
 with(r/(out+'.log')).open('x')as log:cp=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert cp.returncode==0,cp.returncode
 if out=='audit-130':
  j=json.loads((r/out/'summary.json').read_text())
  if j['exact_PLC_surface_pairs']!=0: print(json.dumps(j));raise SystemExit(0)
j=json.loads((r/'audit-159/report.json').read_text());print(json.dumps(dict(surface_pairs=sum(b['result']['count']for b in j['banks']if b.get('result')),complete=j['complete'])))
