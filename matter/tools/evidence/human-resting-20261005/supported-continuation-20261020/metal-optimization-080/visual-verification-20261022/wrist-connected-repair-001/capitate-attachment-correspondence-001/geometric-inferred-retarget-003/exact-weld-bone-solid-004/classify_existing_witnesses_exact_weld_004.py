#!/usr/bin/env python3
"""Reclassify existing proposal points after exact registered-Float32 seam weld.
No candidate search, payload edits, or geometry changes are performed.
"""
import csv,hashlib,importlib.util,json,struct,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=ROOT/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
OUT=E/'capitate-attachment-correspondence-001/geometric-inferred-retarget-003/exact-weld-bone-solid-004'
PROPOSAL=E/'capitate-attachment-correspondence-001/geometric-inferred-retarget-003/geometric-retarget-ladder.json'
GEN=E/'capitate-attachment-correspondence-001/geometric-inferred-retarget-003/generate_geometric_retargets_003.py'
REG=E/'native-pose-checks/connected-hand-registration069i/registered-skin.csv'
BM=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
BP=BM.parent/'bodyparts3d-myosim-major-bones.nhbones'
FRAME=E/'capitate-attachment-correspondence-001/coordinate-frame-review-001/coordinate-frame-review.json'
PRIOR_ID=E/'capitate-attachment-correspondence-001/source-slot-provenance-002/source-slot-provenance-audit.json'
PRIOR_INF=E/'capitate-attachment-correspondence-001/source-slot-provenance-002/inferred-capitate-retargeting-proposals.json'
REPORT=OUT/'exact-weld-bone-witnesses.json'
MAP=OUT/'exact-weld-maps.npz'
BOUND_M=20e-6

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def edge_topology(F):
 directed=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0)
 und=np.sort(directed,axis=1);unique,inv,cnt=np.unique(und,axis=0,return_inverse=True,return_counts=True)
 dirs=defaultdict(list)
 for e,i in zip(directed,inv):dirs[int(i)].append(1 if int(e[0])<int(e[1]) else -1)
 wrong=sum(1 for i,c in enumerate(cnt) if c==2 and sum(dirs[i])!=0)
 return unique,cnt,inv,{'unique_edges':int(len(unique)),'boundary_edges':int(np.sum(cnt==1)),'nonmanifold_edges':int(np.sum(cnt>2)),'inconsistently_oriented_shared_edges':int(wrong)}
def face_components(F,edge_inv):
 n=len(F);parent=list(range(n));rank=[0]*n
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a=find(a);b=find(b)
  if a==b:return
  if rank[a]<rank[b]:a,b=b,a
  parent[b]=a
  if rank[a]==rank[b]:rank[a]+=1
 edgefaces=defaultdict(list)
 for fi in range(n):
  for e in edge_inv[fi*3:fi*3+3]:edgefaces[int(e)].append(fi)
 for fs in edgefaces.values():
  for q in fs[1:]:union(fs[0],q)
 roots=[find(i) for i in range(n)];remap={r:i for i,r in enumerate(sorted(set(roots)))}
 labels=np.asarray([remap[x] for x in roots],dtype=np.int32)
 sizes=Counter(labels.tolist())
 return labels,{str(k):int(v) for k,v in sorted(sizes.items())}
def boundary_components(edges,verts):
 adj=defaultdict(list)
 for a,b in edges:adj[int(a)].append(int(b));adj[int(b)].append(int(a))
 unseen=set(adj);out=[]
 while unseen:
  stack=[min(unseen)];vs=set()
  while stack:
   v=stack.pop()
   if v in vs:continue
   vs.add(v);unseen.discard(v);stack.extend(n for n in adj[v] if n not in vs)
  hist=Counter(len(adj[v]) for v in vs)
  out.append({'vertex_count':len(vs),'edge_count':sum(len(adj[v]) for v in vs)//2,'degree_histogram':{str(k):int(v) for k,v in sorted(hist.items())},'closed_loop':bool(vs) and all(len(adj[v])==2 for v in vs),'sample_registered_vertices_f32':verts[sorted(vs)[:8]].astype(float).tolist()})
 return out
def winding(p,tri):
 a=tri[:,0]-p;b=tri[:,1]-p;c=tri[:,2]-p
 la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
 det=np.einsum('ij,ij->i',a,np.cross(b,c))
 den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 return float(np.sum(2*np.arctan2(det,den))/(4*np.pi))
def load_frames(geom):return geom.frames_from_csv(list(csv.DictReader(REG.open())))
def make_bones(frames,wanted):
 manifest=json.loads(BM.read_text());raw=BP.read_bytes();head=struct.calcsize('<8s5I32s')
 magic,abi,bc,vc,ic,fp,source=struct.unpack_from('<8s5I32s',raw)
 if abi!=3 or source.hex()!=manifest['source']['myosim_source_archive_sha256']:raise RuntimeError('bone payload ABI/source mismatch')
 rs=struct.calcsize('<6I8fI');records=[struct.unpack_from('<6I8fI',raw,head+i*rs) for i in range(bc)]
 off=head+bc*rs;allV=np.asarray([struct.unpack_from('<6f',raw,off+i*24)[:3] for i in range(vc)],dtype=np.float32);off+=vc*24
 allI=np.frombuffer(raw,dtype='<u4',count=ic,offset=off).reshape(-1,3)
 out={};maps={}
 for rec,anchor in zip(records,manifest['source']['anchors']):
  body=int(anchor['core_body_index']);name=str(anchor['myosim_body']).lower()
  if name not in wanted or body not in frames:continue
  bi0,fv,nv,fi,ni,st,tx,ty,tz,qx,qy,qz,qw,scale,sr=rec
  q=np.asarray((qx,qy,qz,qw),float);q/=np.linalg.norm(q);x,y,z,w=q
  R=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
  local=(allV[fv:fv+nv].astype(float)*float(scale))@R.T+np.asarray((tx,ty,tz),float)
  M,t=frames[body][:2];V=((local-t)@M).astype(np.float32)
  F=allI[fi//3:(fi+ni)//3].astype(np.int64)-int(fv)
  if not np.isfinite(V).all() or np.any(F<0) or np.any(F>=nv):raise RuntimeError(f'invalid mesh {name}')
  used=np.unique(F.ravel());raw_to_weld=np.full(nv,-1,dtype=np.int64);key_to_id={};groups=[];weldV=[]
  for rid in used:
   row=V[int(rid)].copy();row[row==0]=np.float32(0.0);key=row.astype('<f4',copy=False).tobytes()
   if key not in key_to_id:key_to_id[key]=len(groups);groups.append([]);weldV.append(row)
   wid=key_to_id[key];raw_to_weld[int(rid)]=wid;groups[wid].append(int(rid))
  Fw=raw_to_weld[F]
  u0,c0,inv0,top0=edge_topology(F);bd0=u0[c0==1]
  u1,c1,inv1,top1=edge_topology(Fw);bd1=u1[c1==1]
  comps,sizes=face_components(Fw,inv1)
  tri=V[F].astype(float);areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
  deg_idx=np.flatnonzero((Fw[:,0]==Fw[:,1])|(Fw[:,1]==Fw[:,2])|(Fw[:,2]==Fw[:,0]))
  zero_area=np.flatnonzero(areas<=1e-15)
  closed=top1['boundary_edges']==0 and top1['nonmanifold_edges']==0 and top1['inconsistently_oriented_shared_edges']==0 and len(deg_idx)==0 and len(zero_area)==0
  slug=f'b{body}_{name}';member_ind=[0];member_ids=[]
  for group in groups:member_ids.extend(group);member_ind.append(len(member_ids))
  maps[slug+'_raw_used_vertex_ids']=used.astype(np.int64)
  maps[slug+'_raw_to_weld_id']=raw_to_weld[used].astype(np.int64)
  maps[slug+'_welded_registered_vertices_f32']=np.asarray(weldV,dtype='<f4')
  maps[slug+'_weld_member_indptr']=np.asarray(member_ind,dtype=np.int64)
  maps[slug+'_weld_member_local_ids']=np.asarray(member_ids,dtype=np.int64)
  maps[slug+'_faces_raw_local']=F.astype(np.int32)
  maps[slug+'_faces_welded']=Fw.astype(np.int32)
  maps[slug+'_welded_face_component']=comps.astype(np.int32)
  out[name]={'body_index':body,'frame_rows':frames[body][2],'raw_used_vertex_count':int(len(used)),'face_count':int(len(F)),'raw_edge_incidence':top0,'raw_boundary_components':boundary_components(bd0,V),'exact_f32_unique_coordinate_count':int(len(weldV)),'exact_f32_duplicate_vertices_welded':int(len(used)-len(weldV)),'weld_rule':'exact registered Float32 xyz equality; signed zero canonicalized; no tolerance/averaging/motion/caps','exact_f32_weld_edge_incidence':top1,'exact_f32_boundary_components':boundary_components(bd1,np.asarray(weldV,dtype=np.float32)),'degenerate_faces_after_weld':int(len(deg_idx)),'zero_area_faces_after_weld':int(len(zero_area)),'connected_face_components':sizes,'closed_consistently_oriented_after_exact_weld':bool(closed),'weld_mapping_arrays_prefix':slug}
 if set(out)!=wanted:raise RuntimeError(f'missing requested bones {sorted(wanted-set(out))}')
 return out,maps,source.hex()
def classify(p,mesh,V,Fw,components):
 T=V[Fw].astype(float);d,b=geom.closest_all(np.asarray(p,float),T);dist=float(np.min(d))
 out={'distance_to_registered_surface_mm':dist*1000,'raw_boundary_edge_count':mesh['raw_edge_incidence']['boundary_edges'],'after_exact_weld_boundary_edge_count':mesh['exact_f32_weld_edge_incidence']['boundary_edges'],'after_exact_weld_nonmanifold_edges':mesh['exact_f32_weld_edge_incidence']['nonmanifold_edges'],'after_exact_weld_orientation_errors':mesh['exact_f32_weld_edge_incidence']['inconsistently_oriented_shared_edges'],'closed_consistently_oriented_after_exact_weld':mesh['closed_consistently_oriented_after_exact_weld']}
 if dist<=BOUND_M:out['solid_classification']='near_surface_20um_numeric_band';return out
 if not mesh['closed_consistently_oriented_after_exact_weld']:out['solid_classification']='indeterminate_open_or_nonmanifold_after_exact_weld';return out
 wind={}
 for c in sorted(set(components.tolist())):wind[str(int(c))]=winding(np.asarray(p,float),T[components==c])
 absw={k:abs(v) for k,v in wind.items()}
 if any(v>=.5 for v in absw.values()):cls='inside_closed_exact_weld_component'
 elif all(v<=.1 for v in absw.values()):cls='outside_closed_exact_weld_components'
 else:cls='indeterminate_solid_angle_band'
 out.update({'solid_classification':cls,'signed_winding_by_component':wind,'absolute_winding_by_component':absw,'winding_rule':'closed consistently oriented exact-weld components only; inside if any |winding|>=0.5, outside if every |winding|<=0.1'})
 return out

def main():
 global geom
 spec=importlib.util.spec_from_file_location('geometric_retargets',GEN);geom=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geom;spec.loader.exec_module(geom)
 frames=load_frames(geom);prior=json.loads(PROPOSAL.read_text());wanted=set()
 for site0 in prior['sites']:
  chosen0=next((r['proposal'] for r in site0['displacement_ladder'] if r.get('proposal') is not None),None)
  if chosen0:
   aw0=chosen0['anatomical_witness'];wanted.update(x['name'] for x in aw0['bone_landmark_witnesses'])
   for key0 in ('capitate_bone_witness','nearest_owner_bone_witness'):
    if aw0.get(key0):wanted.add(aw0[key0]['name'])
  else:
   near0=site0.get('nearest_exact_same_side_region_face_any_normal')
   if near0:wanted.update(near0['owners'])
   wanted.add(site0['donor_name_preserved'])
 bones,maps,payload_source=make_bones(frames,wanted);sites=[]
 for site in prior['sites']:
  proposal=next((r['proposal'] for r in site['displacement_ladder'] if r.get('proposal') is not None),None);diagnostic=proposal is None
  if proposal:
   point=np.asarray(proposal['new_zero_strain_target_world_m'],float);binding=proposal['surface_binding']
  else:
   binding=site.get('nearest_exact_same_side_region_face_any_normal')
   if binding is None:raise RuntimeError(f"no recorded D point for site {site['legacy_site_index']}")
   point=np.asarray(binding['point_world_m'],float)
  side='right' if site['donor_body_preserved']==49 else 'left';suffix='_r' if side=='right' else '_l';rows={}
  for name,mesh in bones.items():
   if not name.endswith(suffix):continue
   prefix=mesh['weld_mapping_arrays_prefix'];V=maps[prefix+'_welded_registered_vertices_f32'];F=maps[prefix+'_faces_welded'];components=maps[prefix+'_welded_face_component']
   rows[name]=classify(point,mesh,V,F,components)
  sites.append({'legacy_site_index':site['legacy_site_index'],'legacy_stable_id':site['legacy_stable_id'],'historical_source_vertex_id_not_current_identity':site['historical_source_vertex_id_not_current_identity'],'donor_body_preserved_as_proposal_parameter':site['donor_body_preserved'],'donor_name_preserved_as_proposal_parameter':site['donor_name_preserved'],'point_role':'nearest exact same-side source-face diagnostic only; no proposal within20mm' if diagnostic else 'existing geometric proposal point; no new search performed','deep_point_world_m':point.tolist(),'binding':{k:binding[k] for k in ('face','nodes','barycentric','source_face_kind','source_face_parent','source_parent_ancestry_ids')},'witnesses_by_same_side_bone':rows})
 np.savez_compressed(MAP,**maps)
 inputs=(PROPOSAL,GEN,REG,BM,BP,FRAME,PRIOR_ID,PRIOR_INF)
 result={'schema':'numi.human.capitate.proposal_bone_exact_f32_weld_classification.v1','status':'existing proposal points reclassified against exact-welded registered bones; no candidates searched and no attachment admitted','scope':'Recorded geometric-proposal points for sites2,3,9,10,17 plus the already recorded nearest D-face diagnostic for site16 only. Original candidate report and strict identity evidence are unchanged.','method':{'registered_coordinates':'same 069i CSV frame transform and registered Float32 emission as proposal generator','weld':'exact equality of registered Float32 xyz, signed zero canonicalized; no epsilon, averaging, motion, caps or topology edits','containment':'signed solid-angle winding per face component only after exact-weld edge incidence is closed, manifold, consistently oriented and nondegenerate','raw_boundary':'reported before welding; raw seam-open edges are not called genuine boundaries when exact welding closes the surface','limits':'Bone point containment/distance only; this does not establish a ligament/fascia path, attachment law, dynamic clearance, or physical/compiler admission.'},'identity_result_nomenclature':{'source_identity_recovered':'0/6','source_identity_evidence':'source-slot-provenance-002/source-slot-provenance-audit.json; accepted_replacements=0 and all six raw IDs were reused output slots','prior_strict_inference':'1/6 proposal under the earlier narrow inferred-attachment criterion; source-slot-provenance-002/inferred-capitate-retargeting-proposals.json','current_geometric_screen':'5/6 under the separate candidate-only screen; site16 remains outside the 20mm search ladder','distinction':'Inferred proposal count is not source-identity recovery.'},'inputs':{str(p):sha(p) for p in inputs},'bone_meshes':bones,'sites':sites,'weld_map_npz':{'path':str(MAP),'sha256':sha(MAP),'mapping_arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in sorted(maps.items())}},'bone_payload_source_archive_sha256':payload_source}
 REPORT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 check=json.loads(REPORT.read_text());assert len(check['sites'])==6 and all(x['closed_consistently_oriented_after_exact_weld'] for x in check['bone_meshes'].values())
 print(json.dumps({'report':str(REPORT),'report_sha256':sha(REPORT),'mapping':str(MAP),'mapping_sha256':sha(MAP),'bone_topology':{k:{'raw_boundary':v['raw_edge_incidence']['boundary_edges'],'welded_boundary':v['exact_f32_weld_edge_incidence']['boundary_edges'],'welded_nonmanifold':v['exact_f32_weld_edge_incidence']['nonmanifold_edges'],'welded_orientation_errors':v['exact_f32_weld_edge_incidence']['inconsistently_oriented_shared_edges'],'closed':v['closed_consistently_oriented_after_exact_weld'],'duplicates_welded':v['exact_f32_duplicate_vertices_welded']} for k,v in bones.items()},'point_classes':[{'site':s['legacy_site_index'],'classifications':{k:v['solid_classification'] for k,v in s['witnesses_by_same_side_bone'].items()}} for s in sites]},indent=2))
if __name__=='__main__':main()
