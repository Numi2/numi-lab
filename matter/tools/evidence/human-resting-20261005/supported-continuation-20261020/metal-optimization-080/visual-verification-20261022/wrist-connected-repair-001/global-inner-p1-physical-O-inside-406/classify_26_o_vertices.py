#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
from collections import Counter
ROOT=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
INIT=ROOT/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/p1-interior-face-support-395/refined-background-unsolved-initializer.npz'
C271=ROOT/'bilateral-source-cut-plane-271/candidate-271-max-cut-distance-bilateral-collar.npz'
POSE=ROOT/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
ASSET=pathlib.Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
REBIND=ROOT/'regional-refinement-compose-255/tools/rebind_human_skin_reference.py'
BODYADM=ROOT/'exact-plane-filter-owner-264/src/numilab_human/bodyparts_adm.py'
INV=ROOT/'registered-surrounding-tissue-native013-191/inventory.json'
BANKS=ROOT/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'
MAN=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json')
EXPECTED={
 str(INIT):'fd11505dbeaec0decf8b71fde1dc3f62c1e224bd9e039646b40dd3f9fcd0ef18',
 str(C271):'92f0079a8266bb1f5bdeb289bffa51c2624341e6e6f2e1e9e0eb0c6d969e3f1d',
 str(POSE):'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd',
 str(ASSET):'d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04',
 str(INV):'b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7',
 str(BANKS):'39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792',
}
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p,h in EXPECTED.items():
 got=sha(p)
 if got!=h: raise RuntimeError(f'input SHA mismatch {p}: {got}')
sys.path.insert(0,str(BODYADM.parent.parent))
spec=importlib.util.spec_from_file_location('rebind_human_skin_reference',REBIND); rebind=importlib.util.module_from_spec(spec); spec.loader.exec_module(rebind)
from numilab_human.bodyparts_adm import prepare_nearest_surface_index

def winding(xyz,faces,q):
 tri=np.asarray(xyz,dtype=np.float64)[faces]
 a=tri[:,0]-q; b=tri[:,1]-q; c=tri[:,2]-q
 la=np.linalg.norm(a,axis=1); lb=np.linalg.norm(b,axis=1); lc=np.linalg.norm(c,axis=1)
 num=np.einsum('ij,ij->i',a,np.cross(b,c))
 den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 return float(np.sum(2*np.arctan2(num,den),dtype=np.float64)/(4*np.pi))
def bary_on_triangle(p,tri):
 a,b,c=np.asarray(tri,dtype=np.float64); v0=b-a; v1=c-a; v2=np.asarray(p,dtype=np.float64)-a
 d00=float(v0@v0); d01=float(v0@v1); d11=float(v1@v1); d20=float(v2@v0); d21=float(v2@v1)
 den=d00*d11-d01*d01
 v=(d11*d20-d01*d21)/den; w=(d00*d21-d01*d20)/den
 return np.asarray([1-v-w,v,w])
t0=time.monotonic()
z=np.load(INIT,allow_pickle=False); c=np.load(C271,allow_pickle=False)
outer_ids=set(map(int,z['physical_outer_node_ids']))
rows=np.asarray([int(i) for i in z['physical_outer_node_ids'] if z['inside_closed_union'][int(i)]],dtype=np.int64)
if len(rows)!=26: raise RuntimeError(f'expected retained 26 witnesses, got {len(rows)}')
outer271=np.asarray(c['outer_world_actual007_m'],dtype=np.float64)
inv=json.loads(INV.read_text()); invbanks=inv['banks']; man=json.loads(MAN.read_text()); labels={int(x['stable_id']):x for x in man['source']['surfaces']}
# Current registered 294 detail asset under exact native013 all-body pose capture.
payload=rebind.decode(ASSET.read_bytes()); pose=json.loads(POSE.read_text()); detail_xyz=rebind.registered_positions_float32(payload,pose,frame='current'); detail_faces=np.asarray(payload['inds'],dtype=np.int64)
detail_idx=prepare_nearest_surface_index(detail_xyz,detail_faces)
# Map flagged current O points to exact row(s) in 271 Float32 coordinate domain, then local source ancestry.
rows_out=[]; all_relevant={}
for r0 in rows:
 r=int(r0); q=np.asarray(z['points_world_m'][r],dtype=np.float64)
 match=np.flatnonzero(np.all(outer271.astype(np.float32)==q.astype(np.float32),axis=1))
 if len(match)!=1: raise RuntimeError(f'O row {r} has {len(match)} exact Float32 candidate271 coordinate matches')
 sr=int(match[0]); inc=np.flatnonzero(np.any(c['surface_faces']==sr,axis=1))
 # Exact nearest registered detailed triangle, plus its barycentric embedding/residual.
 dd,cp,fr=detail_idx.closest(q); cp=np.asarray(cp,dtype=np.float64); fr=int(fr); tri=detail_xyz[detail_faces[fr]].astype(np.float64); bary=bary_on_triangle(cp,tri)
 normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]); normal/=np.linalg.norm(normal)
 w_skin=winding(detail_xyz,detail_faces,q); w_skinproj=winding(detail_xyz,detail_faces,cp)
 # Only closed-soft bank AABBs containing this O point are tested, using 191 cache input identities.
 cand=[]
 for bi,bk in enumerate(invbanks):
  if bk['kind']!='soft_tissue': continue
  if np.all(q>=np.asarray(bk['aabb_min_m'])) and np.all(q<=np.asarray(bk['aabb_max_m'])):
   cand.append((bi,bk))
 for bi,bk in cand: all_relevant[bi]=bk
 rows_out.append({'o_node_014':r,'world_m':q.tolist(),'inside_closed_union_395':True,'candidate271_vertex_row':sr,'candidate271_coordinate_match_f32':True,'candidate271_node_kind':str(c['node_kind'][sr]),'candidate271_source_vertex_id':int(c['source_vertex_ids'][sr]),'candidate271_source_original_d36_vertex_row':sr if sr<2411 and str(c['node_kind'][sr])=='coarse' else None,'candidate271_incident_faces':{'count':int(len(inc)),'kinds':sorted(map(int,np.unique(c['surface_face_kind'][inc]))),'source_sides':sorted(map(int,np.unique(c['surface_face_source_side'][inc]))),'original_d36_face_rows':sorted(map(int,np.unique(c['surface_face_original_d36_row'][inc]))),'parent_surface_ids':sorted(map(int,np.unique(c['surface_face_parent_surface_ids'][inc]))),'detailed_source_ids':sorted(map(int,np.unique(c['surface_face_detailed_source_ids'][inc])))},'nearest_registered_294_skin':{'distance_m':float(dd),'closest_world_m':cp.tolist(),'signed_normal_offset_m':float((q-cp)@normal),'normal_world':normal.tolist(),'nhtiss_face_row':fr,'nhtiss_vertex_rows':detail_faces[fr].tolist(),'barycentric_on_triangle':bary.tolist(),'source_embedding_residual_world_m':(q-cp).tolist(),'generalized_winding_at_o':w_skin,'generalized_winding_at_projection':w_skinproj,'interpretation':'winding is diagnostic only; per-vertex map is nearest-triangle barycentric projection, not direct source ancestry'},'soft_aabb_candidates':[],'soft_groups_containing_o':[],'soft_groups_containing_skin_projection':[]})
# Cache each relevant local bank's exact surface topology, nearest index, and winding values.
B=np.load(BANKS,allow_pickle=False)
for bi,bk in all_relevant.items():
 key=f"bank_{bi:04d}"; vv=np.asarray(B[key+'_vertices'],dtype=np.float32); ff=np.asarray(B[key+'_faces'],dtype=np.int64)
 edges=np.sort(np.vstack((ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]])),axis=1)
 _,counts=np.unique(edges,axis=0,return_counts=True); closed=bool(len(counts)>0 and np.all(counts==2))
 bidx=prepare_nearest_surface_index(vv,ff)
 surf=labels.get(int(bk['stable_id']),{})
 bk['_vv']=vv; bk['_ff']=ff; bk['_closed']=closed; bk['_idx']=bidx; bk['_label']=surf.get('label'); bk['_member_id']=surf.get('member_id'); bk['_layer_name']=surf.get('layer')
for rec in rows_out:
 q=np.asarray(rec['world_m'],dtype=np.float64); cp=np.asarray(rec['nearest_registered_294_skin']['closest_world_m'],dtype=np.float64)
 sr=rec['candidate271_vertex_row']
 qresults=[]
 for bi,bk in all_relevant.items():
  # AABB candidates were selected by q; query both original coarse point and source skin projection against that same muscle.
  vv=bk['_vv']; ff=bk['_ff']; idx=bk['_idx']; closed=bk['_closed']
  d0,c0,f0=idx.closest(q); d1,c1,f1=idx.closest(cp); f0=int(f0);f1=int(f1)
  w0=winding(vv,ff,q) if closed else None; w1=winding(vv,ff,cp) if closed else None
  in0=(abs(w0)>0.5) if closed else None; in1=(abs(w1)>0.5) if closed else None
  nearest0=np.asarray(c0,dtype=np.float64); nearest1=np.asarray(c1,dtype=np.float64)
  tri0=vv[ff[f0]].astype(np.float64); tri1=vv[ff[f1]].astype(np.float64)
  nb0=np.cross(tri0[1]-tri0[0],tri0[2]-tri0[0]); nb0/=np.linalg.norm(nb0)
  nb1=np.cross(tri1[1]-tri1[0],tri1[2]-tri1[0]); nb1/=np.linalg.norm(nb1)
  mrec={'bank_index':int(bi),'bank_key':bk['key'],'stable_id':int(bk['stable_id']),'layer':int(bk['layer']),'label':bk['_label'],'member_id':bk['_member_id'],'body_indices':bk['body_indices'],'closed_by_edge_incidence2':closed,'triangle_count':int(len(ff)),'o_winding':w0,'o_inside':in0,'o_unsigned_distance_m':float(d0),'o_signed_depth_m':(-float(d0) if in0 else float(d0)) if in0 is not None else None,'o_nearest_face_row':f0,'o_nearest_face_barycentric':bary_on_triangle(nearest0,vv[ff[f0]]).tolist(),'skin_projection_winding':w1,'skin_projection_inside':in1,'skin_projection_unsigned_distance_m':float(d1),'skin_projection_signed_depth_m':(-float(d1) if in1 else float(d1)) if in1 is not None else None,'skin_projection_nearest_face_row':f1,'skin_projection_nearest_face_barycentric':bary_on_triangle(nearest1,vv[ff[f1]]).tolist()}
  rec['soft_aabb_candidates'].append(mrec)
  if in0: rec['soft_groups_containing_o'].append({'stable_id':mrec['stable_id'],'label':mrec['label'],'member_id':mrec['member_id'],'penetration_depth_m':float(d0),'bank_index':bi})
  if in1: rec['soft_groups_containing_skin_projection'].append({'stable_id':mrec['stable_id'],'label':mrec['label'],'member_id':mrec['member_id'],'penetration_depth_m':float(d1),'bank_index':bi})
 if rec['soft_groups_containing_o'] and not rec['soft_groups_containing_skin_projection']:
  rec['local_repair_assessment']='matched detailed-skin projection lies outside all tested containing soft groups; source-anchored coupled O patch is a plausible correction candidate, subject to patch-wide topology/collision/mass gates'
 elif rec['soft_groups_containing_skin_projection']:
  rec['local_repair_assessment']='Nearest projection to the current detailed skin remains inside an implicated soft group; snapping or refitting coarse O to the unchanged detailed skin is insufficient. This overlap alone cannot determine whether the detailed skin is undersized or the soft-tissue geometry/registration is displaced. Resolve with attachment/reference-landmark evidence, then update detailed skin and physical O together with explicit source-face barycentric ancestry.'
 else:
  rec['local_repair_assessment']='395 inside-union flag not reproduced by the tested AABB-local Float64 generalized-winding diagnostic on edge-incidence-2 banks; do not infer a specific muscle; retained union classifier needs audit'
report={'schema':'numi.human.physical-O-26-source-and-softgroup-classification.406.v1','status':'bounded diagnostic only; no geometry modified or admitted','scope':'the 26 O vertices flagged inside_closed_union by unsolved 395 initializer; current registered 294 detailed skin; only soft banks whose AABBs contain each O point were tested; no 689-bank rescan','input_sha256':{p:sha(p) for p in EXPECTED},'additional_pins':{str(MAN):sha(MAN),str(REBIND):sha(REBIND),str(BODYADM):sha(BODYADM)},'source_poses_sha256':sha(POSE),'source_294':{'path':str(ASSET),'vertex_count':int(payload['nv']),'face_count':int(len(detail_faces)),'registered_current_xyz_sha256':hashlib.sha256(detail_xyz.tobytes()).hexdigest()},'coarse_candidate_271':{'path':str(C271),'sha256':sha(C271),'points':int(len(outer271)),'detail_source_direct_vertex_count':int(np.count_nonzero(c['source_vertex_ids']>=0))},'retained_395':{'path':str(INIT),'sha256':sha(INIT),'flagged_physical_o_node_count':int(len(rows)),'physical_o_count':int(len(outer_ids))},'tested_local_soft_banks':[{'bank_index':int(i),'key':b['key'],'stable_id':int(b['stable_id']),'label':labels.get(int(b['stable_id']),{}).get('label'),'member_id':labels.get(int(b['stable_id']),{}).get('member_id'),'aabb_candidate_for_o_count':int(sum(np.all(z['points_world_m'][r]>=np.asarray(b['aabb_min_m'])) and np.all(z['points_world_m'][r]<=np.asarray(b['aabb_max_m'])) for r in rows)),'closed_by_edge_incidence2':bool(b['_closed']),'triangles':int(len(b['_ff']))} for i,b in sorted(all_relevant.items())],'nodes':rows_out,'summary':{'groups_float64_winding_diagnostic_inside_o':{f"{sid}|{label}":count for (sid,label),count in Counter((int(g['stable_id']),str(g.get('label'))) for r in rows_out for g in r['soft_groups_containing_o']).items()},'nodes_with_any_tested_winding_group_inside_o':sum(bool(r['soft_groups_containing_o']) for r in rows_out),'nodes_where_skin_projection_remains_in_candidate_group':sum(bool(r['soft_groups_containing_skin_projection']) for r in rows_out),'all_26_coarse_only':all(r['candidate271_node_kind']=='coarse' and r['candidate271_incident_faces']['detailed_source_ids']==[-1] for r in rows_out),'method_limit':'nearest detailed skin projection and AABB-local winding are diagnostics; no patch-wide exact intersection scan or clinical/anatomical qualification'},'elapsed_s':time.monotonic()-t0}
out=ROOT/'global-inner-p1-physical-O-inside-406'/'classification-report.json'; out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(out),'report_sha256':sha(out),'elapsed_s':report['elapsed_s'],'nodes':len(rows_out),'relevant_banks':len(all_relevant),'groups_inside_o':report['summary']['groups_float64_winding_diagnostic_inside_o'],'nodes_inside_o':report['summary']['nodes_with_any_tested_winding_group_inside_o'],'skin_projection_still_in_group':report['summary']['nodes_where_skin_projection_remains_in_candidate_group'],'all_26_coarse_only':report['summary']['all_26_coarse_only']},indent=2))
