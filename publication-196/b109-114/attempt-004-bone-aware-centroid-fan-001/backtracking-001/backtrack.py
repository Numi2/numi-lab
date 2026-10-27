#!/usr/bin/env python3
"""Bounded F32 line-search from a rejected SLSQP direction; exact screens under geometry lock."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib,importlib.util,json,sys,time,traceback
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=S/'per-bank-workers-20261027/b109-114/attempt-004-bone-aware-centroid-fan-001';BD=D/'backtracking-001'
C=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz';AUDIT159=C.parent/'audit-159/report.json'
PRE=D/'selective-pre-fit-patch.npz';TRIAL=D/'private-bone-aware-selective-patch.npz';FIT_REPORT=D/'selective-fit-report.json'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT/'audit_skin_bone_surfaces_159.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
BASE_SHA='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
TRIAL_SHA='01ac76bf8bcc500a01717bde8fec160d9de9c24799f9ad21506ba78837cbd6f6'
FIT_REPORT_SHA='1ec6596339e1a4cbfd649f1a6564a7e4bf8e4ae0f2c1e7758096437b8146f6a1'
SCALE_TRUST_MM=3.0;ANGLE_FLOOR=0.5;ASPECT_CEIL=50.0;AREA_FLOOR=0.10;ANGLE_TOL=1e-5
ALPHAS=[1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625]
REPORT=BD/'backtracking-report.json';ACCEPTED=BD/'accepted-backtracked-patch.npz'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def frac(x):return Fraction.from_float(float(x))
def exact_records_mesh(points,faces,face_ids,vertex_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)];out=[]
 for fid,face in zip(face_ids,faces):
  ids=tuple(map(int,face));tri=tuple(pts[i] for i in ids);n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise ValueError(f'exact degenerate face {fid}')
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  out.append((tri,lo,hi,int(fid),tuple(int(vertex_ids[i]) for i in ids)))
 return out
def pairset(result):return {tuple(sorted(map(int,x))) for x in result['triangle_pairs']}
def main():
 started=time.monotonic();rep={'schema':'numi.human.b109_114.bounded_backtracking.v1','status':'STARTED','candidate_modified':False,'admitted':False,'physical_steps':0,'alpha_schedule':ALPHAS,'trust_cap_mm':SCALE_TRUST_MM}
 try:
  if sha(C)!=BASE_SHA or sha(TRIAL)!=TRIAL_SHA or sha(FIT_REPORT)!=FIT_REPORT_SHA:raise RuntimeError('pinned baseline/trial/report changed')
  fitrep=json.loads(FIT_REPORT.read_text());
  if fitrep.get('status')!='PRIVATE_REJECTED_SLSQP' or fitrep.get('optimizer',{}).get('success') is not False:raise RuntimeError('backtracking input is not the pinned rejected optimizer result')
  fitmod=loadmod('selective_fit_helpers',D/'selective_fit.py')
  helper=loadmod('backtrack_bone_helper',HELPER);audit_tool=loadmod('backtrack_bone_audit_owner',AUDIT_TOOL);exact=audit_tool.exact_module()
  with np.load(C,allow_pickle=False) as z:P=z['points_world_m'].astype(np.float32);F=z['source_O_faces'].astype(np.int64)
  B=np.load(PRE,allow_pickle=False);T0=np.load(TRIAL,allow_pickle=False)
  base=B['points_world_m_f32'].astype(np.float32);trial=T0['points_world_m_f32'].astype(np.float32);faces=B['patch_faces'].astype(np.int64);parents=B['face_parent_candidate_rows'].astype(np.int64);srcrows=B['node_source_candidate_rows'].astype(np.int64)
  if base.shape!=trial.shape or not np.array_equal(faces,T0['patch_faces']) or not np.array_equal(parents,T0['face_parent_candidate_rows']) or not np.array_equal(srcrows,T0['node_source_candidate_rows']):raise RuntimeError('topology/ancestry differs between pre-fit and failed trial')
  selected=set(map(int,B['selected_crossing_parent_face_rows']));patch_parent=set(map(int,B['original_patch_face_rows']))
  fixed_source=set(map(int,B['fixed_boundary_source_node_rows']))|set(map(int,B['frozen_unsafe_parent_vertex_rows']))
  source_to_local={int(v):i for i,v in enumerate(srcrows) if v>=0};fixed_nodes={source_to_local[x] for x in fixed_source}
  if not np.array_equal(base[list(fixed_nodes)].view(np.uint32),trial[list(fixed_nodes)].view(np.uint32)):raise RuntimeError('rejected trial moved a frozen node')
  # Load target closed bone surface and exact point-location structures once.
  T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E);body,bone_id=109,114
  mask=(ids[:,0]==body)&(ids[:,1]==bone_id);bone_raw=np.asarray(T[mask],np.float64);bone_stable=np.asarray(ids[mask,3],np.int64)
  bone_records=audit_tool.records(bone_raw,bone_stable,exact);prepared=exact.prepare_point_location(bone_records)
  base_local=(base.astype(np.float64)-pos[body])@M[body].T-off[body]
  skin_local=(P.astype(np.float64)-pos[body])@M[body].T-off[body]
  vertex_ids=[int(v) if v>=0 else int(len(P)+i) for i,v in enumerate(srcrows)]
  def exact_bone(points):
   loc=(points.astype(np.float64)-pos[body])@M[body].T-off[body]
   records=audit_tool.records(loc[faces],np.arange(len(faces),dtype=np.int64),exact)
   out=exact._audit_pair(records,bone_records,same_surface=False)
   raw=pairset(out);pp={(int(parents[a]),int(b)) for a,b in raw}
   levels=Counter(exact.point_location_prepared(tuple(frac(x) for x in p),prepared)['location'] for p in loc)
   inside={i for i,p in enumerate(loc) if exact.point_location_prepared(tuple(frac(x) for x in p),prepared)['location'] in ('inside','boundary')}
   return loc,records,out,raw,pp,levels,inside
  def exact_self(points,loc,records):
   internal=exact._audit_pair(records,records,same_surface=True);iset=pairset(internal)
   global_world=P.copy()
   for i,sr in enumerate(srcrows):
    if sr>=0:global_world[int(sr)]=points[i]
   global_local=(global_world.astype(np.float64)-pos[body])@M[body].T-off[body]
   comp=np.asarray([i for i in range(len(F)) if i not in patch_parent],np.int64)
   tri=loc[faces];lo=tri.min(axis=(0,1));hi=tri.max(axis=(0,1));ctri=global_local[F[comp]];clo=ctri.min(axis=1);chi=ctri.max(axis=1)
   active=np.flatnonzero(np.all(chi>=lo,axis=1)&np.all(clo<=hi,axis=1));compids=comp[active]
   comp_records=exact_records_mesh(global_local,F[compids],len(faces)+compids,np.arange(len(P),dtype=np.int64),exact)
   cross=exact._audit_pair(records,comp_records,same_surface=True);cset=pairset(cross)
   return {'internal':internal,'internal_set':iset,'complement':cross,'complement_set':cset,'active_count':len(active)}
  base_loc,base_records,base_bone,base_raw,base_parent_pairs,base_loc_counts,base_inside=exact_bone(base)
  base_self=exact_self(base,base_loc,base_records)
  report_trials=[];accepted=False;accepted_alpha=None
  # Rejected optimizer direction, tested at decreasing F32 fractions. Shape gates run first.
  affected=np.flatnonzero(np.any(np.isin(F,np.asarray(sorted(set(map(int,srcrows[srcrows>=0]))-fixed_source),np.int64)),axis=1))
  base_ang,base_asp,base_area=fitmod.tri_metrics(P[F[affected]].astype(np.float64))
  for alpha in ALPHAS:
   world=(base.astype(np.float64)+alpha*(trial.astype(np.float64)-base.astype(np.float64))).astype(np.float32)
   world[list(fixed_nodes)]=base[list(fixed_nodes)]
   dis=np.linalg.norm(world.astype(np.float64)-base.astype(np.float64),axis=1)*1000.0
   trust_ok=bool(dis.max()<=SCALE_TRUST_MM+1e-5)
   fanmask=np.isin(parents,np.asarray(sorted(selected),np.int64))
   fanq=fitmod.metrics(world,faces[fanmask],parents[fanmask],skin_local,F)
   fanpass=bool(fanq['orientation_positive'] and fanq['min_parent_area_ratio']>=AREA_FLOOR and fanq['min_angle_degrees']>=ANGLE_FLOOR and fanq['max_aspect_measure']<=ASPECT_CEIL)
   global_world=P.copy()
   for i,sr in enumerate(srcrows):
    if sr>=0:global_world[int(sr)]=world[i]
   now_ang,now_asp,now_area=fitmod.tri_metrics(global_world[F[affected]].astype(np.float64))
   area_ratio=now_area/np.maximum(base_area,1e-300)
   normal0=np.cross(P[F[affected,1]]-P[F[affected,0]],P[F[affected,2]]-P[F[affected,0]])
   normal1=np.cross(global_world[F[affected,1]]-global_world[F[affected,0]],global_world[F[affected,2]]-global_world[F[affected,0]])
   orient=np.einsum('ij,ij->i',normal0,normal1)
   angle_tol=ANGLE_TOL;aspect_tol=np.maximum(1e-6,base_asp*1e-6)
   bad=(now_ang+angle_tol<base_ang)|(now_asp>base_asp+aspect_tol)|(area_ratio<AREA_FLOOR)|(orient<=0)
   no_worse=not bool(np.any(bad))
   item={'alpha':alpha,'max_displacement_mm':float(dis.max()),'trust_pass':trust_ok,'fan_quality':fanq,'fan_shape_pass':fanpass,'affected_source_faces':len(affected),'nonworsening_pass':no_worse,'worsened_source_face_rows':affected[bad][:64].astype(int).tolist(),'min_affected_angle_delta_deg':float((now_ang-base_ang).min()),'max_affected_aspect_ratio':float((now_asp/np.maximum(base_asp,1e-300)).max()),'minimum_affected_area_ratio':float(area_ratio.min())}
   if not (trust_ok and fanpass and no_worse):item['status']='SHAPE_OR_TRUST_REJECT';report_trials.append(item);continue
   # Exact screens only after Float32 shape passes.
   loc,records,bone_out,raw,pp,counts,inside=exact_bone(world)
   new_inside=sorted(inside-base_inside);new_parent=sorted(pp-base_parent_pairs)
   improved=sorted(base_parent_pairs-pp)
   self_out=exact_self(world,loc,records)
   new_internal=sorted(self_out['internal_set']-base_self['internal_set']);new_complement=sorted(self_out['complement_set']-base_self['complement_set'])
   no_new_self=not new_internal and not new_complement
   strict_improvement=bool(improved) and not new_parent
   no_new_inside=not new_inside
   exact_pass=no_new_inside and strict_improvement and no_new_self
   item.update({'status':'EXACT_SCREEN_PASS' if exact_pass else 'EXACT_SCREEN_REJECT','raw_child_bone_pair_count':int(bone_out['count']),'parent_bone_pair_count':len(pp),'baseline_raw_child_bone_pair_count':int(base_bone['count']),'baseline_parent_bone_pair_count':len(base_parent_pairs),'improved_parent_bone_pairs':[list(x) for x in improved],'new_parent_bone_pairs':[list(x) for x in new_parent],'patch_point_location_counts':dict(counts),'baseline_inside_or_boundary_node_count':len(base_inside),'inside_or_boundary_node_count':len(inside),'new_inside_or_boundary_patch_nodes':new_inside,'patch_internal_self_pair_count':int(self_out['internal']['count']),'patch_complement_self_pair_count':int(self_out['complement']['count']),'new_internal_self_pairs':new_internal[:64],'new_complement_self_pairs':new_complement[:64],'no_new_inside':no_new_inside,'no_new_self_pairs':no_new_self,'strict_parent_pair_improvement':strict_improvement})
   report_trials.append(item)
   if exact_pass:
    save={k:B[k] for k in B.files};save['points_world_m_f32']=world;save['backtracking_alpha_f64']=np.asarray([alpha],np.float64);save['source_candidate_sha256']=np.asarray([BASE_SHA]);save['rejected_fit_source_sha256']=np.asarray([TRIAL_SHA])
    np.savez_compressed(ACCEPTED,**save);accepted=True;accepted_alpha=alpha;item['accepted_patch_path']=str(ACCEPTED);item['accepted_patch_sha256']=sha(ACCEPTED);break
  rep.update({'baseline_exact':{'raw_child_bone_pair_count':int(base_bone['count']),'parent_bone_pair_count':len(base_parent_pairs),'inside_or_boundary_node_count':len(base_inside),'patch_internal_self_pair_count':int(base_self['internal']['count']),'patch_complement_self_pair_count':int(base_self['complement']['count'])},'trials':report_trials,'accepted':accepted,'accepted_alpha':accepted_alpha,'accepted_patch_path':str(ACCEPTED) if accepted else None,'accepted_patch_sha256':sha(ACCEPTED) if accepted else None,'all_exact_trials_have_no_new_inside_requirement':True,'final_status':'PRIVATE_BACKTRACKED_PARTIAL_PATCH' if accepted else 'PRIVATE_NO_BACKTRACKED_ALPHA_PASSED'})
 except Exception as e:
  rep.update({'final_status':'PRIVATE_BACKTRACKING_ABORTED','failure':repr(e),'traceback':traceback.format_exc()})
 rep['elapsed_seconds']=time.monotonic()-started;rep['script_path']=str(Path(__file__));rep['script_sha256']=sha(Path(__file__))
 REPORT.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in rep.items() if k!='traceback'},indent=2,allow_nan=False))
if __name__=='__main__':main()
