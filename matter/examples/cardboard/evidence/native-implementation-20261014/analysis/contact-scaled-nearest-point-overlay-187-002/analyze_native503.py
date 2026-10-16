from pathlib import Path
import datetime,hashlib,json,numpy as np
B=Path('/Users/home/cardboard-implementation-20261014');P=B/'analysis/contact-scaled-nearest-point-overlay-187-002';O=B/'analysis/native503-nearest-point-adjudication-001';N=B/'native/contact-scaled-nearest-point-503';assert not O.exists();O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((N/'receipt.json').read_text());assert r['returncode']==0 and sha(N/'output.bin')==r['outputs_sha256']['output.bin']=='8d8675f642fa6449c0dc68d280a2b09826a7f07df68d40763eb4200421d7a482'
a=np.fromfile(N/'output.bin',dtype='<f4').reshape(48,16,4).astype(np.float64);assert np.isfinite(a).all();rows=[]
for i,x in enumerate(a):
 kind=int(x[0,0]);q=x[1,:3];p0,p1,q0,q1=x[2:6,:3];s=x[0,1]
 if kind==0:
  want=np.array([q[0],q[1],p0[2]]);v=(q[0]-p0[0])/(p1[0]-p0[0]);w=(q[1]-p0[1])/(q0[1]-p0[1]);bary=np.array([1-v-w,v,w])
 elif kind==1:
  want=np.array([q[0],p0[1],p0[2]]);t=(q[0]-p0[0])/(p1[0]-p0[0]);bary=np.array([1-t,t,0.])
 elif kind==2:
  want=q0;bary=np.array([0.,0.,1.])
 if kind<3:
  olderr=max(np.max(np.abs(x[6,:3]-bary)),np.linalg.norm(x[8,:3]-want)/s);newerr=max(np.max(np.abs(x[7,:3]-bary)),np.linalg.norm(x[9,:3]-want)/s)
 else:
  if kind==3:ws=(q0[0]-p0[0])/(p1[0]-p0[0]);wt=(p0[1]-q0[1])/(q1[1]-q0[1])
  elif kind==4:ws=(q0[0]-p0[0])/(p1[0]-p0[0]);wt=0.
  else:ws=0.;wt=(p0[0]-q0[0])/(q1[0]-q0[0])
  wantp=p0+ws*(p1-p0);wantq=q0+wt*(q1-q0);bary=np.array([ws,wt,0.])
  olderr=max(np.max(np.abs(x[6,:3]-bary)),np.linalg.norm(x[8,:3]-wantp)/s,np.linalg.norm(x[10,:3]-wantq)/s);newerr=max(np.max(np.abs(x[7,:3]-bary)),np.linalg.norm(x[9,:3]-wantp)/s,np.linalg.norm(x[11,:3]-wantq)/s)
 rows.append({'row':i,'kind':kind,'scale_m':s,'shift_m':x[0,2],'baseline_error':float(olderr),'candidate_error':float(newerr),'candidate_pass_1e3':bool(newerr<=1e-3)})
assert all(x['candidate_pass_1e3']for x in rows)
report={'schema':'numi.cardboard.native503-nearest-point-analytical-audit.v1','native_receipt_sha256':sha(N/'receipt.json'),'native_output_sha256':sha(N/'output.bin'),'candidate_sha256':sha(P/'candidate/matter/src/metal/contact.metalinc'),'nearest_functions_sha256':sha(P/'nearest-functions.metalinc'),'rows':rows,'baseline_failures':sum(x['baseline_error']>1e-3 for x in rows),'candidate_passes':len(rows),'max_candidate_error':max(x['candidate_error']for x in rows),'oracle':'Analytical axis-aligned face, edge, vertex, crossing/parallel/degenerate segment nearest points derived independently from exact emitted FP32 input coordinates. Error=max(parameter absolute, point absolute/declared physical scale).','gate':1e-3,'physical_transactions':0,'production_integration':False,'trajectory_qualification':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'analyze.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({'files_sha256':{p.name:sha(p)for p in sorted(O.iterdir())if p.is_file()}},indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='rows'}))
