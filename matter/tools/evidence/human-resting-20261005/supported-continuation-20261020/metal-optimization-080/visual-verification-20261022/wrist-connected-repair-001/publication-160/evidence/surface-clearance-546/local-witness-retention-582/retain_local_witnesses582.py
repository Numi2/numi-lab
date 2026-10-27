"""Witness-local retention of a saved coordinate target. Offline construction only.
Each trial is checked globally. No independence, LP optimality, swept-motion or
native-physics acceptance is inferred from local damping.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
N=Path(__file__).resolve().parent.parent
ROOT=N.parent.parent
V567=N/'local-witness-retention-567/retain_local_witnesses567.py'
V567_SHA='422667ee0099145013ae39a706f64cf20e16c8c411d6b69e05d096264c7d9e35'
OWNER=ROOT/'D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
OWNER_SHA='5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
PRED=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
PRED_SHA='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def area_witnesses(x,base,ref,f,floor=.1):
 for a in [x,base,ref]:
  if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():raise ValueError('invalid point field')
 a=x.astype(np.float32).astype(np.float64)*1000.;b=base.astype(np.float64)*1000.;r=ref.astype(np.float64)*1000.
 c0=np.cross(b[f[:,1]]-b[f[:,0]],b[f[:,2]]-b[f[:,0]])
 c1=np.cross(a[f[:,1]]-a[f[:,0]],a[f[:,2]]-a[f[:,0]])
 cr=np.cross(r[f[:,1]]-r[f[:,0]],r[f[:,2]]-r[f[:,0]])
 n0=np.linalg.norm(c0,axis=1);nr=np.linalg.norm(cr,axis=1)
 if np.any(n0<=0)or np.any(nr<=0):raise ValueError('degenerate baseline/reference')
 oriented=np.einsum('ij,ij->i',c1,c0)/(n0*n0);ratio=np.linalg.norm(c1,axis=1)/nr
 return np.flatnonzero((oriented<floor)|(ratio<floor)),oriented,ratio
def damp(alpha,nodes,mutable):
 ids=np.unique(np.asarray(nodes,dtype=np.int64))
 if np.any(ids<0)or np.any(ids>=len(alpha)):raise ValueError('witness node outside domain')
 ids=ids[mutable[ids]]
 if not len(ids):raise ValueError('failing witness has no mutable nodes')
 out=alpha.copy();out[ids]*=.5
 return out,ids
def validate_config(max_edge_gradient,trust_mm):
 if not np.isfinite(max_edge_gradient) or max_edge_gradient<=0:raise ValueError('max-edge-gradient must be finite and positive')
 if not np.isfinite(trust_mm) or trust_mm<=0:raise ValueError('trust-mm must be finite and positive')
 return float(max_edge_gradient),float(trust_mm)
def edge_constraints(owner,base_mm,faces,free,max_edge_gradient):
 max_edge_gradient,_=validate_config(max_edge_gradient,4.)
 EG,rhs,_=owner.build_edge_gradient_constraints(base_mm,faces,free,max_edge_gradient)
 if EG.shape[1]!=3*len(free) or rhs.shape!=(EG.shape[0],):raise ValueError('edge-constraint domain mismatch')
 return EG,rhs
def trial_metrics(trial,base,source,free,EG,rhs,trust_mm):
 _,trust_mm=validate_config(.5,trust_mm)
 if trial.shape!=base.shape or source.shape!=base.shape or trial.ndim!=2 or trial.shape[1]!=3:raise ValueError('trial/source domain mismatch')
 dd=(trial.astype(float)-base.astype(float))*1000.
 es=np.asarray(EG@dd[free].ravel()).ravel()-rhs
 step=np.linalg.norm(dd,axis=1)
 dist=np.linalg.norm(trial.astype(float)-source.astype(float),axis=1)*1000.
 return {'delta_mm':dd,'edge_slack_mm':es,'edge_bad_rows':np.flatnonzero(es < -1e-7),'max_step_mm':float(step.max(initial=0)),'max_source_mm':float(dist.max(initial=0)),'trust_bad_nodes':np.flatnonzero(step>trust_mm+1e-6),'source_bad_nodes':np.flatnonzero(dist>12.000001)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,required=True);ap.add_argument('--target',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--iterations',type=int,default=96);ap.add_argument('--max-edge-gradient',type=float,default=.5);ap.add_argument('--trust-mm',type=float,default=4.);a=ap.parse_args()
 max_edge_gradient,trust_mm=validate_config(a.max_edge_gradient,a.trust_mm)
 if a.out.exists():raise FileExistsError(a.out)
 if not 1<=a.iterations<=128:raise ValueError('bounded iterations required')
 if sha(V567)!=V567_SHA:raise ValueError('frozen 567 source changed')
 if sha(OWNER)!=OWNER_SHA or sha(PRED)!=PRED_SHA:raise ValueError('owner changed')
 st=json.loads(a.state.read_text());bp=Path(st['candidate'])
 if sha(bp)!=st['candidate_sha256']:raise ValueError('baseline candidate changed')
 b=load(bp);t=load(a.target);x=b['points_world_m'];target=t['points_world_m'];f=b['source_O_faces']
 if x.dtype!=np.float32 or target.dtype!=np.float32 or x.shape!=target.shape or not np.isfinite(target).all():raise ValueError('Float32 domain mismatch')
 for k in ['source_prolongation_indptr','source_prolongation_node_rows','source_prolongation_weights']:
  if not same(b[k],t[k]):raise ValueError('material node identity mismatch: '+k)
 anchors=b['typed_coarse_anchor_output_rows_018']
 if not same(x[anchors],target[anchors]):raise ValueError('target moves anchors')
 mutable=np.any(x!=target,axis=1);free=np.flatnonzero(mutable)
 if not len(free):raise ValueError('no saved direction remaining')
 owner=mod('local567owner',OWNER)
 base_mm=x.astype(float)*1000.;ref=b['area_reference_037_points_world_m_f64']
 EG,rhs=edge_constraints(owner,base_mm,f,free,max_edge_gradient)
 R0=ROOT/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
 if sha(R0)!='70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31':raise ValueError('reference changed')
 with np.load(bp,allow_pickle=False)as z:source,_=owner.compose_source_reference(z,R0)
 if not owner.full_f32_area_gate(x,base_mm,ref*1000.,f,.1)['pass']:raise ValueError('baseline area invalid')
 if np.linalg.norm(x.astype(float)-source,axis=1).max()> .012000001:raise ValueError('baseline source cap invalid')
 sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
 pred=mod('numilab_human._local567',PRED)
 records,_=pred.float32_surface_records(x,f,source_seam_groups=())
 baseline,self_report=pred._prepare_complete_self_pair_baseline(records)
 if baseline.triangle_pairs:raise ValueError('baseline self intersections')
 a.out.mkdir(parents=True)
 pins={str(p):sha(p)for p in [Path(__file__),V567,OWNER,PRED,R0,a.state,bp,a.target]}
 alpha=np.ones(len(x));delta=target.astype(float)-x.astype(float);history=[];result=None
 start=time.monotonic()
 for it in range(a.iterations):
  trial=(x.astype(float)+alpha[:,None]*delta).astype(np.float32)
  bad,ori,mag=area_witnesses(trial,x,ref,f)
  area=owner.full_f32_area_gate(trial,base_mm,ref*1000.,f,.1)
  if len(bad)!=len(np.union1d(np.flatnonzero(ori<.1),np.flatnonzero(mag<.1)))or area['pass']!=(len(bad)==0):raise AssertionError('area parity failed')
  metrics=trial_metrics(trial,x,source,free,EG,rhs,trust_mm)
  dd=metrics['delta_mm'];es=metrics['edge_slack_mm'];er=metrics['edge_bad_rows']
  witnesses=[f[bad].ravel()]
  if len(er):witnesses.append(free[np.unique(EG[er].indices//3)])
  witnesses.extend([metrics['trust_bad_nodes'],metrics['source_bad_nodes']])
  rec={'iteration':it,'area_bad_faces':bad.tolist(),'edge_bad_rows':len(er),'max_step_mm':metrics['max_step_mm'],'max_source_mm':metrics['max_source_mm'],'minimum_edge_slack_mm':float(es.min(initial=1e30)),'max_edge_gradient':max_edge_gradient,'trust_mm':trust_mm}
  nodes=np.unique(np.concatenate(witnesses))
  if not len(nodes):
   records,_=pred.float32_surface_records(trial,f,source_seam_groups=())
   _,ss=pred._update_complete_self_pair_baseline(baseline,records)
   pairs=np.asarray(ss['triangle_pairs'],dtype=np.int64).reshape(-1,2)
   rec['exact_self_pairs']=int(ss['count']);rec['self_pairs_first64']=pairs[:64].tolist()
   if len(pairs):nodes=np.unique(f[pairs].ravel())
   else:result=trial;history.append(rec);break
  alpha,affected=damp(alpha,nodes,mutable);rec['damped_node_count']=len(affected);history.append(rec)
 if result is None:
  (a.out/'report.json').write_text(json.dumps({'status':'BOUNDED_LOCAL_DAMPING_NO_VALID_ENDPOINT','iterations':history,'source_pins':pins,'construction_gates':{'max_edge_gradient':max_edge_gradient,'trust_mm':trust_mm,'source_cap_mm':12.0,'area_floor':0.1,'strict_endpoint_self_intersections':True,'scope':'additional construction gates; do not alter final acceptance'},'native_admission':False,'accepted_physical_steps':0},indent=2,allow_nan=False)+'\n')
  raise RuntimeError('bounded local damping found no globally valid endpoint; witnesses retained')
 if same(result,x):raise ValueError('no representable retained correction')
 if not same(result[anchors],x[anchors]):raise AssertionError('anchors changed')
 out=dict(b);out['points_world_m']=result
 if any(not same(b[k],out[k])for k in b if k!='points_world_m'):raise AssertionError('nonpoint payload changed')
 cp=a.out/'candidate.npz';np.savez_compressed(cp,**out)
 pins={str(p):sha(p)for p in [Path(__file__),V567,OWNER,PRED,R0,a.state,bp,a.target]}
 report={'status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED','scope':'witness-local coordinate-target backtracking with whole-mesh actual Float32 gates and exact endpoint self checks; not LP feasibility or native acceptance','base_candidate':str(bp),'base_sha256':sha(bp),'candidate_path':str(cp),'candidate_sha256':sha(cp),'target':str(a.target),'target_sha256':sha(a.target),'free_nodes':free.tolist(),'source_pins':pins,'construction_gates':{'max_edge_gradient':max_edge_gradient,'trust_mm':trust_mm,'source_cap_mm':12.0,'area_floor':0.1,'strict_endpoint_self_intersections':True,'scope':'additional construction gates; final 545/130/159 acceptance remains unchanged'},'iterations':history,'alpha_counts':{str(float(v)):int(np.sum(alpha[free]==v))for v in np.unique(alpha[free])},'full_final_gate':area,'exact_self_pairs':0,'full_545_required':True,'native_admission':False,'accepted_physical_steps':0,'seconds':time.monotonic()-start}
 (a.out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'candidate':str(cp),'iterations':len(history),'alpha_counts':report['alpha_counts'],'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()

