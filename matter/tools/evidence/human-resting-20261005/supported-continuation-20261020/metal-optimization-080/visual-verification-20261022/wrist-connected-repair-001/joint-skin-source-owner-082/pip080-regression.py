from pathlib import Path
import json,subprocess,hashlib,re,importlib.util,numpy as np,time
r=Path('/Users/n/numi-human-aortic-contact-opt-20261019');e=r/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
old=json.loads((e/'middle-pip-local-weight-repair-074-rigid-mid-001/all-finger-poses-001/report.json').read_text())
o=e/'middle-pip-singular-strain-repair-080/all-finger-poses-001';o.mkdir(exist_ok=False)
asset=e/'middle-pip-singular-strain-repair-080/iteration-0/candidate.nhskin'
spec=importlib.util.spec_from_file_location('m',r/'tools/rebind_human_skin_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cand=m.decode(asset.read_bytes());base=m.decode((Path(old['poses']['neutral']['path']).parent/'restored-pre020-weights-current-reference-001.nhskin').read_bytes())
a,b=m.reference_transforms(base,json.loads(Path(old['poses']['neutral']['path']).read_text()),[0,0,.35])
changed=np.flatnonzero(np.any(base['weights']!=cand['weights'],axis=1))
outside=np.setdiff1d(np.arange(base['nv']),changed)
record={'scope':'Detailed source-surface pose audit and numerical invariants only; not physical or natural ROM qualification.','candidate_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'changed_vs_restored':changed.tolist(),'reference_max_m':float(np.linalg.norm(m.positions(cand,a,b)-m.positions(base,a,b),axis=1).max()),'outside_vertex_records_exact':bool(np.array_equal(cand['verts'][outside],base['verts'][outside])),'faces_exact':bool(np.array_equal(cand['inds'],base['inds'])),'normal_bytes_exact':bool(np.array_equal(cand['verts'][:,3:6],base['verts'][:,3:6])),'weight_sum_max':float(np.abs(cand['weights'].sum(axis=1,dtype=float)-1).max()),'poses':[]}
for name,pose in old['poses'].items():
 f=o/(name+'.csv')
 cmd=[old['inputs']['pose_exporter']['path'],str(asset),pose['path'],old['inputs']['reference_registered_csv']['path'],str(f)]
 z=subprocess.run(cmd,capture_output=True,text=True);assert z.returncode==0,z.stderr
 z=subprocess.run([old['inputs']['probe']['path'],str(asset),old['inputs']['fixture']['path'],str(f)],capture_output=True,text=True)
 hit=re.search(r'source_crossing_count=(\d+)',z.stdout)
 row=dict(name=name,pose=pose,returncode=z.returncode,crossings=int(hit.group(1)) if hit else None,stdout=z.stdout,stderr=z.stderr)
 record['poses'].append(row);f.unlink()
 print(name,row['crossings'],flush=True)
record['all_pass']=all(t['returncode']==0 and t['crossings']==0 for t in record['poses'])
(o/'report.json').write_text(json.dumps(record,indent=2)+'\n')
print('all_pass',record['all_pass'],'changed_rows',len(changed),'reference_max_m',record['reference_max_m'],flush=True)
