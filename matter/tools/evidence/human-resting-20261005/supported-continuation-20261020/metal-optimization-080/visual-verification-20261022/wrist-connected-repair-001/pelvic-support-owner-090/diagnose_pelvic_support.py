from pathlib import Path
import importlib.util,json,hashlib,numpy as np,subprocess,re,time
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
P=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
O=E/'pelvic-distal-skin-influence-repair-089';O.mkdir(exist_ok=False)
sp=importlib.util.spec_from_file_location('rb',R/'tools/rebind_human_skin_reference.py');rb=importlib.util.module_from_spec(sp);sp.loader.exec_module(rb)
src=E/'middle-pip-singular-strain-repair-080/iteration-0/candidate.nhskin';payload=rb.decode(src.read_bytes())
a,b=rb.reference_transforms(payload,json.loads((P/'neutral.json').read_text()),[0,0,.35])
W=payload['weights'];by={int(z['body']):i for i,z in enumerate(payload['binds'])}
g=np.clip((W[:,by[128]].astype(float)-.2)/.2,0,1);g=g*g*(3-2*g)
distal=[136,137,138,139,142,150,151,152,153,156];lanes=[by[x] for x in distal]
old=json.loads((P/'joint-surface-audit-final074-001/report.json').read_text())
poses={name:r for name,r in old['poses'].items() if name.startswith('knee') or name.startswith('ankle') or name.startswith('mtp')}
for name in ['knee_0p5','knee_1p0','knee_1p5','knee_2p0']:
 path=E/'joint-regional-quality-088-expanded-poses'/(name+'.json');poses[name]={'path':str(path)}
reports=[]
for strength in [.5,1.]:
 out=O/str(strength);out.mkdir();w=W.astype(float).copy();removed=w[:,lanes]*(g*strength)[:,None];w[:,lanes]-=removed;w[:,by[128]]+=removed.sum(axis=1)
 result=rb.rebind(payload,w,a,b);print(type(result),flush=True)
 if isinstance(result,tuple):blob=result[0];metadata=result[1:]
 else:blob=result;metadata=None
 candidate=out/'candidate.nhskin';candidate.write_bytes(blob)
 new=rb.decode(blob);changed=np.flatnonzero(np.any(new['weights']!=W,axis=1));outside=np.ones(payload['nv'],bool);outside[changed]=False
 assert np.array_equal(new['verts'][outside],payload['verts'][outside]);assert new['inds'].tobytes()==payload['inds'].tobytes()
 reference_err=np.linalg.norm(rb.positions(new,a,b)-rb.positions(payload,a,b),axis=1)
 report={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(blob).hexdigest(),'strength':strength,'policy':'Continuous pelvis-weight smoothstep .2-.4 removes distal lower-leg influence, transfers exactly to pelvis; reference positions rebound once','distal_bodies':distal,'changed_vertices':changed.tolist(),'maximum_reference_error_m':float(reference_err.max()),'outside_vertex_records_exact':True,'faces_exact':True,'poses':{}}
 for name,p in poses.items():
  csv=out/(name+'.csv');z=subprocess.run([old['inputs']['pose_exporter']['path'],str(candidate),p['path'],str(P/'final074-neutral-reference-002/registered-skin.csv'),str(csv)],text=True,capture_output=True);assert z.returncode==0,z.stderr
  z=subprocess.run([old['inputs']['probe']['path'],str(candidate),old['inputs']['fixture']['path'],str(csv)],text=True,capture_output=True)
  hit=re.search(r'source_crossing_count=(\d+)',z.stdout);report['poses'][name]={'crossings':int(hit.group(1)) if hit else None,'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr};csv.unlink()
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
 print(json.dumps({'strength':strength,'changed':len(changed),'reference_error':float(reference_err.max()),'crossings':{n:r['crossings'] for n,r in report['poses'].items()}}),flush=True)
(O/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
