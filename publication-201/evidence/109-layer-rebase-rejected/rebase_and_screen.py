#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
OUT=S/'per-bank-workers-20261027/b109-114/attempt-022-a69-rebase-checks-001'
BASE=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz'
A69=ROOT/'D/publication-200-stage-001/evidence/retained/candidate.npz'
INC=S/'per-bank-workers-20261027/b109-114/attempt-021-witness-union-backtrack-001/alpha-0p125/candidate.npz'
INC_REPORT=S/'per-bank-workers-20261027/b109-114/attempt-021-witness-union-backtrack-001/alpha-0p125/backtrack-report.json'
FIT_REPORT=S/'per-bank-workers-20261027/b109-114/attempt-020-witness-union-resolve-001/report.json'
LAYER=ROOT/'D/publication-200-stage-001/evidence/layer/guarded-65-quarter/skin-fat-layers-541.npz'
LAYER_REPORT=ROOT/'D/publication-200-stage-001/evidence/layer/guarded-65-quarter/report.json'
A69_SELF=ROOT/'D/publication-200-stage-001/evidence/retained/audit-130/surface-self-pairs.npy'
A69_QUERY=ROOT/'D/publication-200-stage-001/evidence/retained/audit-130/query/pairs.csv'
A69_159=ROOT/'D/publication-200-stage-001/evidence/retained/audit-159/report.json'
BT_SCRIPT=S/'per-bank-workers-20261027/b109-114/attempt-016-self-barrier-union-001/backtrack.py'
AUD159=ROOT/'audit_skin_bone_surfaces_159.py'
PYTHON=Path('/Users/numi/numi-human-anatomy-finish-1378/.venv/bin/python')
EXPECTED={'base':'7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af','a69':'a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db','increment':'6a38d42d2161cde3d5972291c4f2cf19c163c395f1a022b6ee330e4dd3a0aa64','layer':'78916a0c05eb130e5efb02174342295c2d8d4444eae963c7b36fc43a2a1b17fc'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def mod(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise RuntimeError('cannot import '+str(path))
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def vol(a,b,c,d):return np.einsum('ij,ij->i',np.cross(b-a,c-a),d-a)/6.0
def layer_failures(z,outerkey,innerkey,fieldkey):
 outer=np.asarray(z[outerkey],dtype=np.float64);inner=np.asarray(z[innerkey],dtype=np.float64);src=np.asarray(z['outer_faces'],dtype=np.int64)
 f=np.sort(src,axis=1);a,b,c=f.T
 inv=((src[:,0]>src[:,1]).astype(np.int8)+(src[:,0]>src[:,2]).astype(np.int8)+(src[:,1]>src[:,2]).astype(np.int8));parity=inv&1
 v0=vol(outer[a],outer[b],outer[c],inner[c]);v1=vol(outer[a],outer[b],inner[b],inner[c]);v2=vol(outer[a],inner[a],inner[b],inner[c])
 field=np.asarray(z[fieldkey],dtype=np.float64);tol=np.maximum(1e-8,np.maximum.reduce([field[a],field[b],field[c]]))**3*1e-12;expected=np.where(parity,1.0,-1.0)
 authored=np.isfinite(v0)&np.isfinite(v1)&np.isfinite(v2)&(np.abs(v0)>tol)&(np.abs(v1)>tol)&(np.abs(v2)>tol)&(np.sign(v0)==expected)&(np.sign(v1)==-expected)&(np.sign(v2)==expected)
 ids=np.stack([np.stack([a,b,c,len(outer)+c],axis=1),np.stack([a,b,len(outer)+b,len(outer)+c],axis=1),np.stack([a,len(outer)+a,len(outer)+b,len(outer)+c],axis=1)],axis=1).reshape(-1,4)
 xyz64=np.concatenate((outer,inner));q=xyz64[ids];v64=vol(q[:,0],q[:,1],q[:,2],q[:,3]);reverse=v64<0
 ids[reverse,0],ids[reverse,1]=ids[reverse,1].copy(),ids[reverse,0].copy()
 xyz32=xyz64.astype(np.float32).astype(np.float64);p=xyz32[ids];v=vol(p[:,0],p[:,1],p[:,2],p[:,3]);long=np.zeros(len(p))
 for i in range(4):
  for j in range(i+1,4):long=np.maximum(long,np.linalg.norm(p[:,i]-p[:,j],axis=1))
 nd=6*v/np.maximum(long,1e-300)**3
 return {'authored_orientation':set(map(int,np.flatnonzero(~authored))),'f32_nonpositive_tets':set(map(int,np.flatnonzero(v<=0))),'f32_conditioned_tets':set(map(int,np.flatnonzero(nd<1e-5)))}
def main():
 t=time.monotonic()
 for key,p in [('base',BASE),('a69',A69),('increment',INC),('layer',LAYER)]:
  got=sha(p)
  if got!=EXPECTED[key]:raise RuntimeError(f'{key} hash mismatch: {got}')
 base=load(BASE);a69=load(A69);inc=load(INC)
 if not(set(base)==set(a69)==set(inc)):raise RuntimeError('typed key set mismatch')
 F=np.asarray(base['source_O_faces'],dtype=np.int64)
 if not(np.array_equal(F,a69['source_O_faces']) and np.array_equal(F,inc['source_O_faces'])):raise RuntimeError('face topology differs')
 for k in base:
  if k=='points_world_m':continue
  if any(x[k].dtype!=base[k].dtype or x[k].shape!=base[k].shape or x[k].tobytes()!=base[k].tobytes() for x in (a69,inc)):raise RuntimeError('typed lineage changed: '+k)
 X=np.asarray(base['points_world_m'],dtype=np.float32);A=np.asarray(a69['points_world_m'],dtype=np.float32);Q=np.asarray(inc['points_world_m'],dtype=np.float32)
 sa=np.flatnonzero(np.any(A.view(np.uint32)!=X.view(np.uint32),axis=1));si=np.flatnonzero(np.any(Q.view(np.uint32)!=X.view(np.uint32),axis=1));over=np.intersect1d(sa,si)
 if len(over):raise RuntimeError('increment and a69 support overlap '+str(over[:30].tolist()))
 delta=Q.astype(np.float64)-X.astype(np.float64);Y=A.copy();Y[si]=(A[si].astype(np.float64)+delta[si]).astype(np.float32)
 cand={k:v.copy() for k,v in a69.items()};cand['points_world_m']=Y
 CP=OUT/'candidate-a69-rebased.npz'
 with CP.open('xb') as f:np.savez_compressed(f,**cand)
 changed=np.flatnonzero(np.any(Y.view(np.uint32)!=A.view(np.uint32),axis=1));actual=(Y.astype(np.float64)-A.astype(np.float64))*1000
 report={'schema':'numi.human.b109-114.a69-rebase-check.v1','status':'REBASING','pins':{'base653_sha256':sha(BASE),'a69_sha256':sha(A69),'increment_sha256':sha(INC),'increment_backtrack_report_sha256':sha(INC_REPORT),'increment_fit_report_sha256':sha(FIT_REPORT),'layer_a69_sha256':sha(LAYER),'layer_report_a69_sha256':sha(LAYER_REPORT)},'support':{'a69_changed_nodes_vs_653':len(sa),'increment_changed_nodes_vs_653':len(si),'support_intersection_count':len(over),'increment_moved_nodes_after_f32_rebase':len(changed),'max_rebased_displacement_mm':float(np.linalg.norm(actual,axis=1).max()),'sum_rebased_displacement_mm':float(np.linalg.norm(actual,axis=1).sum()),'displacement_rule':'Add the Float64 difference between each emitted Float32 increment point and its 653 Float32 baseline point to the a69 Float32 point, then round once to Float32.'},'candidate':{'path':str(CP),'sha256':sha(CP)},'geometry_admitted':False,'physical_steps':0}
 # Shape, ancestry, anchor and trust checks against the actual a69 parent.
 bt=mod('attempt016_shape_helpers',BT_SCRIPT);sh=bt.shape(A,Y,F,a69,cand)
 shcompact={k:v for k,v in sh.items() if k not in ('changed_node_rows','affected_face_rows')};report['shape_against_a69']=shcompact
 (OUT/'candidate-shape.json').write_text(json.dumps(shcompact,indent=2,sort_keys=True)+'\n')
 if not sh['pass']:
  report['status']='STOP_SHAPE_FAILURE';report['elapsed_s']=time.monotonic()-t
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'shape':shcompact},indent=2));return
 # Reanchor the pinned a69 offset prisms with this same outer Float32 displacement.
 layer0=load(LAYER);la=np.asarray(layer0['outer_points_f64'],dtype=np.float64)
 if not np.array_equal(la,A.astype(np.float64)):raise RuntimeError('layer source outer does not bitwise match a69')
 if not np.array_equal(np.sort(layer0['outer_faces'],axis=1),np.sort(F,axis=1)):raise RuntimeError('a69 layer triangle identity mismatch')
 dy=Y.astype(np.float32).astype(np.float64)-A.astype(np.float32).astype(np.float64);layer1={k:v.copy() for k,v in layer0.items()}
 for key in ('outer_points_f64','inner_points_f64','deep_points_f64'):layer1[key]=np.asarray(layer0[key],dtype=np.float64)+dy
 layer1['outer_points_f32']=layer1['outer_points_f64'].astype(np.float32);layer1['inner_points_f32']=layer1['inner_points_f64'].astype(np.float32);layer1['deep_points_f32']=layer1['deep_points_f64'].astype(np.float32)
 layer1['skin_thickness_actual_norm_m']=np.linalg.norm(layer1['inner_points_f64']-layer1['outer_points_f64'],axis=1);layer1['fat_thickness_actual_norm_m']=np.linalg.norm(layer1['deep_points_f64']-layer1['inner_points_f64'],axis=1)
 LP=OUT/'skin-fat-layers-a69-rebased.npz'
 with LP.open('xb') as f:np.savez_compressed(f,**layer1)
 layer_delta={}
 for label,ok,ik,fk in [('skin','outer_points_f64','inner_points_f64','skin_thickness_transferred_m'),('fat','inner_points_f64','deep_points_f64','fat_thickness_transferred_m')]:
  bsets=layer_failures(layer0,ok,ik,fk);csets=layer_failures(layer1,ok,ik,fk);layer_delta[label]={}
  for name in bsets:layer_delta[label][name]={'baseline_count':len(bsets[name]),'candidate_count':len(csets[name]),'added':sorted(csets[name]-bsets[name]),'removed':sorted(bsets[name]-csets[name])}
 report['layer_check']={'baseline_path':str(LAYER),'baseline_sha256':sha(LAYER),'candidate_path':str(LP),'candidate_sha256':sha(LP),'source_layer_outer_matches_a69':True,'transfer':'same inherited offsets reanchored by rebased Float32 outer displacement','per_layer_failure_identity_delta':layer_delta,'added_failure_count':sum(len(v['added']) for row in layer_delta.values() for v in row.values())}
 (OUT/'layer-failure-comparison.json').write_text(json.dumps(report['layer_check'],indent=2,sort_keys=True)+'\n')
 if report['layer_check']['added_failure_count']:
  report['status']='STOP_ADDED_LAYER_FAILURE';report['elapsed_s']=time.monotonic()-t
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'layer':report['layer_check']},indent=2));return
 # Complete exact full-surface self screen, compared with a69's exact pair file.
 a69self=np.asarray(np.load(A69_SELF,allow_pickle=False),dtype=np.int64).reshape(-1,2)
 auditmod=mod('audit159_rebase_self',AUD159);exact=auditmod.exact_module();records,_=exact.float32_surface_records(Y,F,source_seam_groups=())
 sr=exact._audit_self_pairs_touching_faces(records,set(range(len(F))));selfpairs=np.asarray(sr.get('triangle_pairs',()),dtype=np.int64).reshape(-1,2)
 np.save(OUT/'full-surface-self-pairs.npy',selfpairs);keyset=lambda q:{(int(x),int(y)) for x,y in q}
 self_added=keyset(selfpairs)-keyset(a69self);self_removed=keyset(a69self)-keyset(selfpairs)
 report['full_surface_self']={'complete':bool(sr['complete_surface_audit']),'face_count':int(sr['complete_surface_face_count']),'baseline_a69_pair_count':len(a69self),'candidate_pair_count':len(selfpairs),'added_pairs':[list(x) for x in sorted(self_added)],'removed_pairs':[list(x) for x in sorted(self_removed)],'pair_file_sha256':sha(OUT/'full-surface-self-pairs.npy'),'predicate_sha256':sha(auditmod.EXACT)}
 if not sr['complete_surface_audit']:raise RuntimeError('incomplete exact full-surface self audit')
 if self_added:
  report['status']='STOP_NEW_SELF_PAIR';report['elapsed_s']=time.monotonic()-t
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'self':report['full_surface_self']},indent=2));return
 # Exact all-bank crossing audit, then compare full raw and parent-mapped identity sets to a69.
 AUD=OUT/'audit-159';cmd=[str(PYTHON),str(AUD159),str(CP),str(AUD),'--max-seconds','240']
 cp=subprocess.run(cmd,capture_output=True,text=True);(OUT/'audit-159-launch.log').write_text(cp.stdout+'\n---stderr---\n'+cp.stderr)
 if cp.returncode:raise RuntimeError('159 full audit failed: '+cp.stderr[-2000:])
 current=json.loads((AUD/'report.json').read_text());before=json.loads(A69_159.read_text())
 if not current.get('complete') or current.get('bank_count')!=185 or current.get('candidate_sha256')!=sha(CP):
  report['all_bank_audit']={'status':current.get('status'),'complete':current.get('complete'),'bank_count':current.get('bank_count'),'expected_bank_count':current.get('expected_bank_count'),'report_sha256':sha(AUD/'report.json')}
  report['status']='STOP_INCOMPLETE_ALL_BANK_AUDIT';report['elapsed_s']=time.monotonic()-t;(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'all_bank_audit':report['all_bank_audit']},indent=2));return
 def crosses(doc):return {(int(b['body']),int(b['stable_bone_id']),int(f),int(bt)) for b in doc['banks'] for f,bt in ((b.get('result') or {}).get('triangle_pairs') or [])}
 oldraw=crosses(before);newraw=crosses(current);rawadd=newraw-oldraw;rawrem=oldraw-newraw;parents=np.asarray(a69['refinement_face_parent_face_row'],dtype=np.int64)
 oldparent={(body,bone,int(parents[face]),tri) for body,bone,face,tri in oldraw};newparent={(body,bone,int(parents[face]),tri) for body,bone,face,tri in newraw};p_add=newparent-oldparent;p_rem=oldparent-newparent
 oldby={(int(b['body']),int(b['stable_bone_id'])):len((b.get('result') or {}).get('triangle_pairs') or []) for b in before['banks']}
 perbank=[{'body':int(b['body']),'stable_bone_id':int(b['stable_bone_id']),'baseline_pairs':oldby[(int(b['body']),int(b['stable_bone_id']))],'candidate_pairs':len((b.get('result') or {}).get('triangle_pairs') or [])} for b in current['banks']]
 report['all_bank_audit']={'status':current['status'],'complete':True,'bank_count':current['bank_count'],'baseline_report_sha256':sha(A69_159),'candidate_report_path':str(AUD/'report.json'),'candidate_report_sha256':sha(AUD/'report.json'),'baseline_raw_pair_count':len(oldraw),'candidate_raw_pair_count':len(newraw),'raw_pair_add_count':len(rawadd),'raw_pair_remove_count':len(rawrem),'raw_pair_added':[list(x) for x in sorted(rawadd)],'raw_pair_removed':[list(x) for x in sorted(rawrem)],'baseline_parent_pair_count':len(oldparent),'candidate_parent_pair_count':len(newparent),'parent_pair_add_count':len(p_add),'parent_pair_remove_count':len(p_rem),'parent_pair_added':[list(x) for x in sorted(p_add)],'parent_pair_removed':[list(x) for x in sorted(p_rem)],'per_bank_counts':perbank}
 # Sample the 24 109/114 target nodes against all 185 banks and compare to a69.
 fitdoc=json.loads(FIT_REPORT.read_text());targets=np.asarray(fitdoc['target_inside_nodes'],dtype=np.int64);helper=mod('attempt016_rebase_query',BT_SCRIPT)
 querydir=OUT/'sampled-depth-query';rows=helper.target_query(Y,targets,querydir);targetset=set(map(int,targets));base_at={}
 for r in csv.DictReader(A69_QUERY.open()):
  q=int(r['query_row'])
  if q in targetset:base_at.setdefault(q,[]).append(r)
 newinside=set();oldinside=set();after_target={};before_target={}
 for q,rr in base_at.items():
  for r in rr:
   if r['classification']=='inside':oldinside.add((q,int(r['bank_body']),int(r['bank_stable_id'])))
 for r in rows:
  q=int(targets[int(r['query_row'])]);bb,bi=int(r['bank_body']),int(r['bank_stable_id'])
  if r['classification']=='inside':newinside.add((q,bb,bi))
  if r['classification'] not in ('inside','outside') or int(r['ambiguous']):raise RuntimeError('sample query ambiguous/incomplete')
  if (bb,bi)==(109,114):after_target[q]=r
 for q in targets:
  q=int(q);rr=[r for r in base_at.get(q,[]) if (int(r['bank_body']),int(r['bank_stable_id']))==(109,114)]
  if len(rr)!=1 or q not in after_target:raise RuntimeError('missing a69/candidate target depth row '+str(q))
  before_target[q]=rr[0]
 depth=[]
 for q in sorted(after_target):
  bdist=float(before_target[q]['signed_face_distance_m']);adist=float(after_target[q]['signed_face_distance_m'])
  depth.append({'node_row':q,'baseline_a69_classification':before_target[q]['classification'],'candidate_classification':after_target[q]['classification'],'baseline_a69_signed_face_distance_mm':bdist*1000,'candidate_signed_face_distance_mm':adist*1000,'progress_mm':(adist-bdist)*1000})
 progresses=[r['progress_mm'] for r in depth]
 report['sampled_depth']={'target':[109,114],'queried_target_nodes':len(targets),'banks_per_node':185,'new_inside_identities_vs_a69':[list(x) for x in sorted(newinside-oldinside)],'removed_inside_identities_vs_a69':[list(x) for x in sorted(oldinside-newinside)],'signed_depth_rows':depth,'minimum_progress_mm':min(progresses),'sum_progress_mm':sum(progresses),'no_new_inside_identities':not bool(newinside-oldinside),'no_target_depth_regression':min(progresses)>=-1e-6,'positive_total_target_progress':sum(progresses)>0,'query_files':{p.name:sha(p) for p in querydir.iterdir() if p.is_file()}}
 report['status']='REBASED_AND_CHECKED';report['geometry_admitted']=False;report['physical_steps']=0;report['layer_gate']='DIAGNOSTIC_NONREGRESSION_ONLY';report['elapsed_s']=time.monotonic()-t
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'candidate_sha256':sha(CP),'shape':report['shape_against_a69'],'layer_added':report['layer_check']['added_failure_count'],'self':report['full_surface_self'],'all_bank':{k:report['all_bank_audit'][k] for k in ('bank_count','baseline_raw_pair_count','candidate_raw_pair_count','raw_pair_add_count','raw_pair_remove_count','baseline_parent_pair_count','candidate_parent_pair_count','parent_pair_add_count','parent_pair_remove_count')},'depth':{k:report['sampled_depth'][k] for k in ('minimum_progress_mm','sum_progress_mm','no_new_inside_identities','no_target_depth_regression','positive_total_target_progress')},'report':str(OUT/'report.json')},indent=2),flush=True)
if __name__=='__main__':main()
