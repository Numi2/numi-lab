from pathlib import Path
import importlib.util,json,hashlib,numpy as np,subprocess,csv,re,time
from scipy.optimize import minimize
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001';P=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001');O=E/'forefoot-connected-weight-repair-091';O.mkdir(exist_ok=False)
sp=importlib.util.spec_from_file_location('rb',R/'tools/rebind_human_skin_reference.py');rb=importlib.util.module_from_spec(sp);sp.loader.exec_module(rb)
src=E/'pelvic-support-owner-090/candidate.nhskin';p=rb.decode(src.read_bytes());F=p['inds'];W=p['weights'].astype(float);by={int(b['body']):i for i,b in enumerate(p['binds'])}
seed=set(map(int,F[60121]));selected=set(seed)
for f in F:
 if any(int(i) in seed for i in f):selected.update(map(int,f))
sel=np.array(sorted(selected));faceids=np.flatnonzero(np.any(np.isin(F,sel),axis=1));localF=F[faceids]
a0,b0=rb.reference_transforms(p,json.loads((P/'neutral.json').read_text()),[0,0,.35]);X=rb.positions(p,a0,b0)
v=X[localF];A=np.stack((v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=2);_,RR=np.linalg.qr(A,mode='reduced');invR=np.linalg.inv(RR);area2=np.linalg.norm(np.cross(A[:,:,0],A[:,:,1]),axis=1)
poses={}
for name in ['mtp_m0p5','mtp_0p5']:
 path=E/'joint-regional-quality-088-expanded-poses'/(name+'.json');poses[name]=path
old=json.loads((P/'joint-surface-audit-final074-001/report.json').read_text())
for name,r in old['poses'].items():
 if name.startswith('mtp') or name.startswith('knee'):poses[name]=Path(r['path'])
cache={}
for name,path in poses.items():
 a,b=rb.reference_transforms(p,json.loads(path.read_text()),[0,0,.35]);Y=rb.positions(p,a,b);cache[name]=(a,b,Y)
def emit_local(d,a,b):
 w=W[sel].copy();w[:,by[152]]+=d;w[:,by[153]]-=d
 M=np.einsum('nk,kij->nij',w,a0);t=w@b0
 x=np.linalg.solve(M,(X[sel]-t)[...,None])[...,0]
 return np.einsum('nk,kij,nj->ni',w,a,x)+w@b
def metrics(d):
 rows=[]
 for name,(a,b,Y0) in cache.items():
  Y=Y0.copy();Y[sel]=emit_local(d,a,b);v=Y[localF];B=np.stack((v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=2)
  D=B@invR;ss=np.linalg.svd(D,compute_uv=False);ar=np.linalg.norm(np.cross(B[:,:,0],B[:,:,1]),axis=1)/area2
  rows.extend([ss[:,-1]-.355,2.4-ss[:,0],ar-.25])
 return np.concatenate(rows)
d0=np.zeros(len(sel));bounds=[(max(-.03,-W[i,by[152]]),min(.03,W[i,by[153]])) for i in sel]
t=time.perf_counter();sol=minimize(lambda d:float(d@d),d0,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':metrics}],options={'maxiter':80,'ftol':1e-12,'disp':False})
np.savez(O/'solution.npz',delta=sol.x,selected=sel,source_faces=faceids)
report={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'seed_face_local_row':60121,'selected_source_vertices':sel.tolist(),'touched_source_face_rows':faceids.tolist(),'algorithm':'minimal adjacent calcn_l/toes_l redistribution, neutral reference rebind, allincidenttriangle singular/area constraints in5poses','body_ids':[152,153],'max_abs_weight_change':float(np.max(np.abs(sol.x))),'optimizer_success':bool(sol.success),'message':str(sol.message),'iterations':int(sol.nit),'seconds':time.perf_counter()-t,'minimum_constraint_before':float(metrics(d0).min()),'minimum_constraint_after':float(metrics(sol.x).min()),'status':'UNADMITTED_SOURCE_CANDIDATE'}
if sol.success and metrics(sol.x).min()>-1e-7:
 w=W.copy();w[sel,by[152]]+=sol.x;w[sel,by[153]]-=sol.x;blob,rr=rb.rebind(p,w,a0,b0);asset=O/'candidate.nhskin';asset.write_bytes(blob);report['rebind']=rr;report['candidate_sha256']=hashlib.sha256(blob).hexdigest();report['poses']={}
 for name,path in poses.items():
  out=O/(name+'.csv');z=subprocess.run([old['inputs']['pose_exporter']['path'],str(asset),str(path),str(P/'final074-neutral-reference-002/registered-skin.csv'),str(out)],capture_output=True,text=True);assert z.returncode==0,z.stderr
  z=subprocess.run([old['inputs']['probe']['path'],str(asset),old['inputs']['fixture']['path'],str(out)],capture_output=True,text=True);m=re.search(r'source_crossing_count=(\d+)',z.stdout)
  report['poses'][name]={'crossings':int(m.group(1)) if m else None,'stdout':z.stdout,'stderr':z.stderr,'returncode':z.returncode};out.unlink()
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
