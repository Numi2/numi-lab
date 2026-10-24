#!/usr/bin/env python3
import hashlib, importlib.util, json, pathlib, sys, time
from collections import Counter,defaultdict
from fractions import Fraction
import numpy as np
E=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'source-route-frame-owner-394verifyunused'; OUT=D/'fixed-boundary-remesh-008-star-fan-38385-retry002'
BASE=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
EMIT=E/'source-route-frame-owner-387/current_pose_preflight.py'; IDENT=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
POSE=pathlib.Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
FK=('right-ring-pip-57.json','right-little-pip-61.json','left-ring-pip-95.json','left-little-pip-99.json')
BUILDER=D/'fixed-boundary-source-fit-001/build_source_fit_007d_robust134.py'
EXACT=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
PINS={'baseline':'8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042','builder':'d470d630fdff725251666a483103dca2a11b21b7ba7ed11491249ba14b040a55','emitter':'f34897a3cbad65313ee59b164a33cb528dfad0fe94e2f683fa1e49a1baecb201','identity003':'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd'}
FKFILES=FK; SID=134; SEEDS=(63329,63331); HUB=38385; SELECTED=(131,132,133,134,137,138,141,142)
CYCLE=(18132,19024,33328,35427,38398,39062,38387,38501)
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def pose_map(d):
 return {int(x['core_body']):{'position_world_m':x['position'],'orientation_world_xyzw':x['quaternion_xyzw']} for x in d['source_poses']}
def patch_rows(F,seeds,n=3):
 ed=defaultdict(list)
 for i,t in enumerate(F):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])): ed[tuple(sorted((int(a),int(b))))].append(i)
 s=set(seeds)
 for _ in range(n):
  add=set()
  for rs in ed.values():
   if any(x in s for x in rs): add.update(rs)
  s.update(add)
 rows=np.array(sorted(s),np.int64);cnt=Counter();di={}
 for t in F[rows]:
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
   k=tuple(sorted((int(a),int(b))));cnt[k]+=1;di[k]=(int(a),int(b))
 be={e for e,c in cnt.items() if c==1};return rows,be,{e:di[e] for e in be}
def cycle_from(d):
 out={a:b for a,b in d.values()};start=min(out);q=[start];cur=start
 while True:
  cur=out.get(cur)
  if cur is None: raise RuntimeError('open patch boundary')
  if cur==start: break
  if cur in q: raise RuntimeError('repeated boundary vertex')
  q.append(cur)
 if len(q)!=len(d): raise RuntimeError('boundary is not one cycle')
 return tuple(q)
def topo(F):
 ed=defaultdict(list)
 for t in np.asarray(F,np.int64):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
   a=int(a);b=int(b);ed[tuple(sorted((a,b)))].append((a,b))
 bd={k:v[0] for k,v in ed.items() if len(v)==1}
 return {'boundary':bd,'boundary_count':len(bd),'overfull':sum(len(v)>2 for v in ed.values()),'bad_orientation':sum(len(v)==2 and v[0]!=v[1][::-1] for v in ed.values())}
def qmetrics(V,F):
 v=np.asarray(V,np.float32);t=v[np.asarray(F,np.int64)].astype(np.float64)
 a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];c=t[:,2]-t[:,1];area=np.linalg.norm(np.cross(a,b),axis=1)*.5
 alt=2*area/np.maximum.reduce([np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)])*1000
 ulp=np.abs(np.spacing(v))*1000;ratio=alt/np.maximum(ulp[np.asarray(F,np.int64)].max(axis=(1,2)),1e-300)
 return {'min_altitude_mm':float(alt.min()),'min_area_mm2':float(area.min()*1e6),'zero_area_faces':int(np.count_nonzero(area==0)),'min_altitude_to_max_vertex_ulp':float(ratio.min())}
def flux(V,F):
 t=np.asarray(V,np.float64)[np.asarray(F,np.int64)];av=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])*.5
 return {'area_vector_m2':av.sum(0).tolist(),'signed_volume_m3':float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6),'area_m2':float(np.linalg.norm(av,axis=1).sum())}
def bbox_rows(V,F,lo,hi):
 tri=np.asarray(V,np.float32)[np.asarray(F,np.int64)];a=tri.min(1);b=tri.max(1)
 return np.flatnonzero(np.all(b>=lo,1)&np.all(a<=hi,1)).astype(np.int64)
def recs(ci,V,F,rows):
 rows=np.asarray(sorted(set(map(int,rows))),np.int64)
 if len(rows)==0:return []
 r,_=ci.float32_surface_records(np.asarray(V,np.float32),np.asarray(F,np.int64)[rows])
 return [(x[0],x[1],x[2],int(row),x[4]) for x,row in zip(r,rows)]
def pairs(ci,a,b,same):
 o=set()
 for x,y in (((a,b),(b,a)) if same else ((a,b),)):
  if x and y:
   for p in ci._audit_pair(x,y,same_surface=same).get('triangle_pairs',[]):
    i,j=map(int,p);o.add(tuple(sorted((i,j))) if same else (i,j))
 return o
def signedvol(V,F):
 t=np.asarray(V,np.float64)[np.asarray(F,np.int64)]
 return float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
def main():
 start=time.time()
 for p in (BASE,EMIT,IDENT,BUILDER,EXACT,*[POSE/f for f in FKFILES]):
  if not p.is_file():raise RuntimeError(f'missing {p}')
 hashes={'baseline':sha(BASE),'builder':sha(BUILDER),'emitter':sha(EMIT),'identity003':sha(IDENT),'exact':sha(EXACT),**{f:sha(POSE/f) for f in FKFILES}}
 for k,v in PINS.items():
  if hashes[k]!=v:raise RuntimeError(f'{k} hash mismatch {hashes[k]} != {v}')
 if OUT.exists():raise RuntimeError(f'output exists, refusing overwrite: {OUT}')
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'exact_star_fan_gate.py').write_bytes(pathlib.Path(__file__).read_bytes())
 src=EMIT.read_text();marker='base=emit_selected(BASE)'
 if src.count(marker)!=1:raise RuntimeError('emitter marker mismatch')
 ns={};exec(compile(src.split(marker)[0],str(EMIT),'exec'),ns);ns['WANTED']=set(SELECTED)
 ident=json.loads(IDENT.read_text());specs=[('current191',{int(x['body_index']):x['current'] for x in ident['all_bodies']}),('rest003',{int(x['body_index']):x['rest'] for x in ident['all_bodies']})]
 for f in FKFILES:specs.append((f[:-5],pose_map(json.loads((POSE/f).read_text()))))
 meshes={}
 for name,poses in specs:ns['poses']=poses;meshes[name]=ns['emit_selected'](BASE)
 V0,F0,_=meshes['current191'][SID];rows,bedges,bdir=patch_rows(F0,SEEDS,3);cycle=cycle_from(bdir);verts=sorted(set(map(int,F0[rows].ravel())))
 if (len(rows),len(verts),len(bedges),cycle)!=(18,14,8,CYCLE):raise RuntimeError(f'disk mismatch: {len(rows)} {len(verts)} {len(bedges)} {cycle}')
 fan=np.array([[cycle[i],cycle[(i+1)%8],HUB] for i in range(8)],np.int64)
 keep=np.array([i for i in range(len(F0)) if i not in set(map(int,rows))],np.int64)
 Fc=np.vstack([F0[keep],fan]);fans=np.arange(len(keep),len(Fc),dtype=np.int64)
 tosrc=np.r_[keep,np.full(len(fans),-1,np.int64)]
 tb=topo(F0);tc=topo(Fc)
 same_boundary=tb['boundary']==tc['boundary']
 if not same_boundary or tc['overfull'] or tc['bad_orientation']:raise RuntimeError('full-bank topology/boundary gate failed')
 # Load the published exact pair owner using unique module identity.
 sys.path.insert(0,str(EXACT.parent.parent));from numilab_human import cardiac_cavity_intersections as ci
 # Full-bank closed/oriented validation and signed-winding preparation at current pose.
 fullbase,_=ci.float32_surface_records(V0,F0)
 fullcand,_=ci.float32_surface_records(V0,Fc)
 winding={}
 for key,rr in [('baseline',fullbase),('candidate',fullcand)]:
  try:
   nc=ci._validate_closed_oriented_record_topology(rr)
   prep=ci.prepare_signed_winding(rr)
   q=tuple(Fraction.from_float(float(z)) for z in np.asarray(V0,np.float64).mean(0))
   winding[key]={'closed_oriented_manifold':True,'component_count':int(nc),'signed_winding_prepared':True,'vertex_mean_query':ci.signed_winding_number(q,prep)}
  except Exception as e:winding[key]={'closed_oriented_manifold':False,'error':f'{type(e).__name__}: {e}'}
 pr={};patchset=set(map(int,rows));fanset=set(map(int,fans))
 for name,mesh in meshes.items():
  V,F,_=mesh[SID];V=np.asarray(V,np.float32);F=np.asarray(F,np.int64);CF=np.vstack([F[keep],fan])
  oldq=qmetrics(V,F[rows]);newq=qmetrics(V,fan)
  oldflux=flux(V,F[rows]);newflux=flux(V,fan)
  oldt=V[F[rows]].astype(np.float64);newt=V[fan].astype(np.float64)
  oldn=np.cross(oldt[:,1]-oldt[:,0],oldt[:,2]-oldt[:,0]).sum(0)
  dots=np.einsum('ij,j->i',np.cross(newt[:,1]-newt[:,0],newt[:,2]-newt[:,0]),oldn)
  if newq['zero_area_faces'] or np.any(dots<=0):raise RuntimeError(f'{name}: fan orientation/area failed')
  box=V[np.asarray(verts,np.int64)];lo=box.min(0);hi=box.max(0)
  bt=bbox_rows(V,F,lo,hi);ct=bbox_rows(V,CF,lo,hi)
  bpair=pairs(ci,recs(ci,V,F,rows),recs(ci,V,F,bt),True)
  cpair=pairs(ci,recs(ci,V,CF,fans),recs(ci,V,CF,ct),True)
  bp=set();bi=[]
  for a,b in bpair:
   ia=a in patchset;ib=b in patchset
   if ia and ib:bi.append((a,b))
   elif ia:bp.add(b)
   elif ib:bp.add(a)
  cp=set();ci_internal=[]
  for a,b in cpair:
   ia=a in fanset;ib=b in fanset
   if ia and ib:ci_internal.append((a,b))
   elif ia:
    if b<0 or b>=len(tosrc) or tosrc[b]<0:raise RuntimeError('candidate face row ancestry invalid')
    cp.add(int(tosrc[b]))
   elif ib:
    if a<0 or a>=len(tosrc) or tosrc[a]<0:raise RuntimeError('candidate face row ancestry invalid')
    cp.add(int(tosrc[a]))
  cross={};newpartners=0
  for tid in SELECTED:
   if tid==SID:continue
   TV,TF,_=mesh[tid];TV=np.asarray(TV,np.float32);TF=np.asarray(TF,np.int64);tr=bbox_rows(TV,TF,lo,hi)
   bpairx=pairs(ci,recs(ci,V,F,rows),recs(ci,TV,TF,tr),False)
   cpairx=pairs(ci,recs(ci,V,CF,fans),recs(ci,TV,TF,tr),False)
   bset={b for a,b in bpairx};cset={b for a,b in cpairx};delta=sorted(cset-bset);newpartners+=len(delta)
   cross[str(tid)]={'baseline_raw_exact_pairs':len(bpairx),'candidate_raw_exact_pairs':len(cpairx),'baseline_partner_face_rows':sorted(bset),'candidate_partner_face_rows':sorted(cset),'new_partner_face_rows':delta,'resolved_partner_face_rows':sorted(bset-cset),'target_aabb_face_count':int(len(tr))}
  gdelta=signedvol(V,CF)-signedvol(V,F)
  area_delta=np.asarray(newflux['area_vector_m2'])-np.asarray(oldflux['area_vector_m2'])
  pr[name]={'self':{'baseline_raw_exact_pairs':len(bpair),'candidate_raw_exact_pairs':len(cpair),'baseline_partner_face_rows':sorted(bp),'candidate_partner_face_rows':sorted(cp),'new_partner_face_rows':sorted(cp-bp),'resolved_partner_face_rows':sorted(bp-cp),'baseline_internal_patch_pairs':[list(x) for x in sorted(bi)],'candidate_internal_fan_pairs':[list(x) for x in sorted(ci_internal)],'baseline_target_aabb_face_count':int(len(bt)),'candidate_target_aabb_face_count':int(len(ct))},
   'other_selected_banks':cross,'new_partner_count':int(len(cp-bp)+newpartners),'baseline_quality':oldq,'candidate_quality':newq,'min_fan_dot_vs_patch_normal':float(dots.min()),
   'patch_area_vector_delta_m2':area_delta.tolist(),'patch_area_vector_delta_norm_m2':float(np.linalg.norm(area_delta)),
   'patch_signed_volume_delta_m3':float(newflux['signed_volume_m3']-oldflux['signed_volume_m3']),'whole_bank_signed_volume_delta_m3':float(gdelta),
   'baseline_patch_flux':oldflux,'candidate_patch_flux':newflux}
  pr[name]['status']='NO_NEW_PARTNER_FACES' if pr[name]['new_partner_count']==0 and not ci_internal else 'NEW_EXACT_LOCAL_HITS'
 all_no_new=all(x['new_partner_count']==0 and not x['self']['candidate_internal_fan_pairs'] for x in pr.values())
 qok=all(x['candidate_quality']['zero_area_faces']==0 and x['candidate_quality']['min_altitude_to_max_vertex_ulp']>=x['baseline_quality']['min_altitude_to_max_vertex_ulp']*(1-1e-6) for x in pr.values())
 windingok=all(winding.get(k,{}).get('closed_oriented_manifold') for k in ('baseline','candidate'))
 status='PASS_LOCAL_EXACT_NONREGRESSION' if all_no_new and qok and windingok else 'FAIL_OR_INCOMPLETE_LOCAL_GATE'
 # Explicit full-cavity construction ancestry. It does not represent overlap.
 offs=np.arange(0,(len(fan)+1)*len(rows),len(rows),dtype=np.uint32);anc=np.tile(rows.astype(np.uint32),len(fan))
 report={'schema':'numi.human.stable134.fixed-boundary-star-fan-diagnostic.v1','status':status,'admission':'diagnostic_only_no_NHTISS_regeneration_no_material_or_FEM_mapping',
  'input_hashes':hashes,'baseline_path':str(BASE),'builder_path':str(BUILDER),'emitter_path':str(EMIT),'identity003_path':str(IDENT),'exact_predicate_path':str(EXACT),
  'pose_inputs':{n:(hashes['identity003'] if n in ('current191','rest003') else hashes[n+'.json']) for n,_ in specs},
  'exact_record_owner':'collision-free-seam-records-owner-315 float32_surface_records; no source seam groups supplied; source vertex rows remain unaliased',
  'patch':{'stable_id':SID,'seed_face_rows':list(SEEDS),'replaced_source_face_rows':rows.tolist(),'source_vertex_rows':verts,'fixed_boundary_edges':[list(e) for e in sorted(bedges)],'fixed_boundary_directed_edges':[list(bdir[e]) for e in sorted(bdir)],'fixed_boundary_cycle':list(cycle),'hub_source_vertex_row':HUB,'fan_source_vertex_triangles':fan.tolist(),'construction_ancestry_kind':'every new fan face has all 18 old disk rows as construction ancestry; this is not exact geometric overlap or material/FEM ancestry','fan_construction_ancestry_offsets':offs.tolist(),'fan_construction_ancestry_source_face_rows':anc.tolist(),'source_coordinates_or_bindings_changed':False,
  'unused_old_patch_vertex_rows':[v for v in verts if v not in set(map(int,Fc.ravel()))]},
  'full_bank_current191_topology':{'baseline_boundary_edges':tb['boundary_count'],'candidate_boundary_edges':tc['boundary_count'],'oriented_boundary_exactly_equal':same_boundary,'candidate_overfull_edges':tc['overfull'],'candidate_bad_internal_edge_orientation':tc['bad_orientation'],'winding_preflight':winding},
  'pose_results':pr,'all_six_no_new_partner_faces':all_no_new,'quality_nonworsening_all_six':qok,'closed_oriented_winding_ready':windingok,'elapsed_seconds':time.time()-start,
  'limitations':['Exact collision gate covers the changed disk against its same-bank AABB neighborhood and AABB-overlapping selected banks only; no full-anatomy, bone, or tissue clearance qualification.','Partner-face sets are compared at patch scope because a fan triangle has the full 18-face disk as construction ancestry; raw exact triangle-pair counts are retained and no one-to-one overlap ancestry is claimed.','No asset regenerated and no FEM/material map or physical admission claim.']}
 rp=OUT/'exact-gate-report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 if status=='PASS_LOCAL_EXACT_NONREGRESSION':
  np.savez_compressed(OUT/'topology-only-candidate.npz',stable_id=np.array([SID],np.uint32),source_patch_face_rows=rows.astype(np.uint32),hub_source_vertex_row=np.array([HUB],np.uint32),fan_source_vertex_triangles=fan.astype(np.uint32),fixed_boundary_cycle=np.asarray(cycle,np.uint32),construction_ancestry_offsets=offs,construction_ancestry_source_face_rows=anc)
 print(json.dumps({'status':status,'report':str(rp),'report_sha256':sha(rp),'source_sha256':sha(OUT/'exact_star_fan_gate.py'),'all_six_no_new':all_no_new,'quality_nonworsening':qok,'winding_ready':windingok,'elapsed_seconds':report['elapsed_seconds'],'per_pose':{n:{'new_partner_count':v['new_partner_count'],'self_raw':v['self']['candidate_raw_exact_pairs'],'internal_fan_pairs':len(v['self']['candidate_internal_fan_pairs'])} for n,v in pr.items()}},indent=2),flush=True)
if __name__=='__main__':main()

