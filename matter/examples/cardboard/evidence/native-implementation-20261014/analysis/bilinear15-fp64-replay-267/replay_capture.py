from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import bilinear15_reference as ref
b=Path('/Users/home/cardboard-implementation-20261014');cap=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False,parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def words(a):return np.asarray(a,dtype=np.float32).astype(float)
def pair(s,k):return words(s[k])+words(s.get(k+'_low',np.zeros_like(s[k])))
t0=time.monotonic();records=[json.loads(s) for s in cap.read_text().splitlines() if s.strip()];header=records[0]
assert records[-1]['record']=='integrity' and records[-1]['production_inputs_bitwise_unchanged']
tets=[r for r in records if r['record']=='tetrahedron' and r['material_index'] in [0,1]]
pars=words(header['parameters_by_environment']);p=np.array([pars[t['environment'],28*t['material_index']:28*(t['material_index']+1)] for t in tets]);old=np.array([words(t['center']['accepted_state']) for t in tets]);f=np.array([pair(t['center'],'F_local') for t in tets]);zcap=np.array([pair(t['center'],'projected_state') for t in tets]);h=np.array([words(t['center']['F_direction_local_per_s']) for t in tets]);z,it,norm=ref.project(p,f,old)
pref=ref.stress(p,f,z);pcap=np.array([words(t['center']['P_local_Pa']) for t in tets]);tref=ref.tangent(p,f,old,zcap,h);tcap=np.array([words(t['center']['tangent_local_Pa_per_s']) for t in tets]);tprojected=ref.tangent(p,f,old,z,h)
def metric(a,r):
 a=a.reshape(len(tets),-1);r=r.reshape(len(tets),-1);d=np.max(np.abs(a-r),axis=1);s=np.max(np.abs(r),axis=1);return {'absolute_max':float(d.max()),'relative_max_nonzero':float(np.max(d[s>0]/s[s>0])) if (s>0).any() else None,'worst_index':int(np.argmax(d/np.maximum(s,1e-300))),'global_l2_relative':float(np.linalg.norm(a-r)/np.linalg.norm(r)) if np.linalg.norm(r)>0 else None}
report={'schema':'numi.cardboard.bilinear15-independent-fp64-replay.v1','capture':str(cap),'capture_sha256':sha(cap),'source_sha256':{x.name:sha(x) for x in [Path(__file__),Path(ref.__file__)]},'selection':{k:header[k] for k in ['root','newton_iteration','environment_count']},'paper_tet_count':len(tets),'equations':'Independent handwritten15stateHill/tensile-hardening residuals, FP64 safeguardedNewton, complex-step partialJacobian and implicit derivative. No compiler/VM/Metal projection or derivative call. All physical parameters and accepted states roundtrip exactFP32; paired F and candidate words widened independently.','candidate_projection':metric(zcap,z),'stress_reprojected':metric(pcap,pref),'tangent_at_captured_candidate':metric(tcap,tref),'tangent_after_reference_projection':metric(tcap,tprojected),'reference_max_scaled_residual':float(norm.max()),'max_reference_iterations':int(it.max()),'limits':['Nonzero-direction tangent agreement is separate from complete smooth finite-difference gates.','Zero directional actions do not qualify a relative error.','No complete coupon/convergence/physical validation claim.'],'wall_seconds':time.monotonic()-t0}
np.savez_compressed(out/'arrays.npz',p=p,F=f,old=old,captured_candidate=zcap,projected_candidate=z,direction=h,nativeP=pcap,referenceP=pref,nativeT=tcap,referenceT=tref,referenceProjectedT=tprojected,iterations=it)
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
