#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import hashlib,importlib.util,json,sys,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';D=W/'b43-16/attempt-028-edge-collapse-001'
BASE=W/'combined-d071-b109-b61-001/candidate.npz';BASE159=W/'combined-d071-b109-b61-001/audit-159/report.json';BASE130=W/'combined-d071-b109-b61-001/audit-130/summary.json';LAYER=W/'combined-d071-b109-b61-001/skin-fat-layers-541.npz';RANK=D/'target-bank-edge-ranking-v2.json'
PY=Path('/Users/numi/numi-human-prep-venv-20261005/bin/python')
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
LAYER_OWNER=D.parent.parent/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
PIN='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def audit_pairs(points,faces,face_ids,bank,bank_ids,body,pos,M,off,aud,exact):
 local=(points-pos[body])@M[body].T-off[body];skin=local[np.asarray(faces,dtype=np.int64)];B=np.asarray(bank)
 if not len(skin) or not len(B):return set(),{'skin_aabb':0,'bank_aabb':0}
 blo,bhi=B.min((0,1)),B.max((0,1));slo,shi=skin.min(1),skin.max(1)
 active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
 if not len(active):return set(),{'skin_aabb':0,'bank_aabb':0}
 rlo,rhi=slo[active].min(0),shi[active].max(0);bactive=np.flatnonzero(np.all(B.max(1)>=rlo,axis=1)&np.all(B.min(1)<=rhi,axis=1))
 sr=aud.records(skin[active],np.asarray(face_ids,dtype=np.int64)[active],exact);br=aud.records(B[bactive],np.asarray(bank_ids,dtype=np.int64)[bactive],exact)
 res=exact._audit_pair(sr,br,same_surface=False)
 return {(int(a),int(b)) for a,b in (res.get('triangle_pairs') or [])},{'skin_aabb':int(len(active)),'bank_aabb':int(len(bactive))}
def main():
 if RANK.exists():raise FileExistsError(RANK)
 assert sha(BASE)==PIN
 with np.load(BASE,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 topo=json.loads((D/'report.json').read_text());items=[x for x in topo['batch_candidates'] if x['status']=='PASS_CHEAP']
 F=np.asarray(a['source_O_faces'],np.int64);P=np.asarray(a['points_world_m'],np.float32).astype(np.float64);parent=np.asarray(a['refinement_face_parent_face_row'],np.int64)
 base159=json.loads(BASE159.read_text());brec=next(x for x in base159['banks'] if int(x['body'])==43 and int(x['stable_bone_id'])==16)
 base_pairs={(int(f),int(b)) for f,b in ((brec.get('result') or {}).get('triangle_pairs') or [])}
 base_mapped={(int(parent[f]),int(b)) for f,b in base_pairs}
 helper=mod('bank_input_helper_43collapse',HELPER);exactmod=mod('audit159_43collapse',AUD159);exact=exactmod.exact_module()
 T,ids,gids,bodies,M,off,pos,keys,pins,unused=helper.read_bank_inputs(BANKDIR,E);mask=(ids[:,0]==43)&(ids[:,1]==16)
 bank=np.asarray(T[mask]);bank_ids=np.asarray(ids[mask,3])
 # Validate exact baseline local reconstruction for every possibly affected old row.
 all_changed=sorted({int(f) for x in items for f in x['changed_faces']});base_local,base_aabb=audit_pairs(P,F[all_changed],all_changed,bank,bank_ids,43,pos,M,off,exactmod,exact)
 expected={(f,b) for f,b in base_pairs if f in set(all_changed)}
 if base_local!=expected:raise ValueError(f'baseline exact subset mismatch; got={len(base_local)} expected={len(expected)}')
 with np.load(LAYER,allow_pickle=False) as z:l={k:z[k].copy() for k in z.files}
 if not np.array_equal(np.asarray(l['outer_points_f32'],np.float32),a['points_world_m']):raise ValueError('layer bundle outer points not pinned to 806')
 layer_owner=mod('layer_owner_43collapse',LAYER_OWNER)
 O=np.asarray(l['outer_points_f64'],np.float64);I=np.asarray(l['inner_points_f64'],np.float64);Deep=np.asarray(l['deep_points_f64'],np.float64)
 skind=np.asarray(l['skin_thickness_transferred_m'],np.float64);fatd=np.asarray(l['fat_thickness_transferred_m'],np.float64)
 reports=[];all_bank_rank=[]
 for item in items:
  v=int(item['inside_node']);u=int(item['survivor']);drop=set(map(int,item['collapse_faces_removed']));changed=sorted(map(int,item['changed_faces']))
  keepf=np.asarray([i for i in range(len(F)) if i not in drop],np.int64);fo2n=np.full(len(F),-1,np.int64);fo2n[keepf]=np.arange(len(keepf))
  nmap=np.full(len(P),-1,np.int64);n2o=np.asarray([i for i in range(len(P)) if i!=v],np.int64);nmap[n2o]=np.arange(len(n2o))
  Fc=F[keepf].copy()
  for oldfi in changed:Fc[int(fo2n[oldfi])][Fc[int(fo2n[oldfi])]==v]=u
  Fc=nmap[Fc]
  if np.any(Fc<0):raise ValueError('collapse candidate has removed-vertex face')
  new_changed=np.asarray([fo2n[x] for x in changed],np.int64)
  ids_for_exact=np.asarray(changed,np.int64)
  candpairs,ab=audit_pairs(P[n2o],Fc[new_changed],ids_for_exact,bank,bank_ids,43,pos,M,off,exactmod,exact)
  affected=drop|set(changed)
  raw_after=(base_pairs-{x for x in base_pairs if x[0] in affected})|candpairs
  mapped_unaffected={(int(parent[f]),int(b)) for f,b in base_pairs if f not in affected}
  mapped_changed={(int(parent[f]),int(b)) for f,b in candpairs}
  mapped_after=mapped_unaffected|mapped_changed
  new_mapped=mapped_after-base_mapped;removed_mapped=base_mapped-mapped_after
  # Skin/fat prism gate on changed output faces only, using inherited layer offsets and exact [0,2,1] layer winding.
  lf=Fc[new_changed][:,[0,2,1]]
  sf=layer_owner.face_flags(O[n2o],I[n2o],lf,skind[n2o]);ff=layer_owner.face_flags(I[n2o],Deep[n2o],lf,fatd[n2o])
  layer_fail={'skin_authored':[int(changed[i]) for i in np.flatnonzero(sf['authored_bad'])],
   'skin_nonpositive_tets':[[int(changed[i]),int(t)] for i,t in np.argwhere(sf['nonpositive'])],
   'skin_conditioned_tets':[[int(changed[i]),int(t)] for i,t in np.argwhere(sf['conditioned'])],
   'fat_authored':[int(changed[i]) for i in np.flatnonzero(ff['authored_bad'])],
   'fat_nonpositive_tets':[[int(changed[i]),int(t)] for i,t in np.argwhere(ff['nonpositive'])],
   'fat_conditioned_tets':[[int(changed[i]),int(t)] for i,t in np.argwhere(ff['conditioned'])]}
  layer_pass=not any(layer_fail.values())
  rec={'inside_node':v,'survivor':u,'edge_length_mm':float(item['edge_length_mm']),'changed_faces_old_rows':changed,'removed_faces_old_rows':sorted(drop),
   'raw_43_16_before':len(base_pairs),'raw_43_16_after':len(raw_after),'raw_removed_count':len(base_pairs-raw_after),'raw_added_count':len(raw_after-base_pairs),
   'raw_removed_pairs':[list(x) for x in sorted(base_pairs-raw_after)],'raw_added_pairs':[list(x) for x in sorted(raw_after-base_pairs)],
   'mapped_43_16_before':len(base_mapped),'mapped_43_16_after':len(mapped_after),'mapped_removed_count':len(removed_mapped),'mapped_added_count':len(new_mapped),
   'mapped_removed_pairs':[list(x) for x in sorted(removed_mapped)],'mapped_added_pairs':[list(x) for x in sorted(new_mapped)],
   'exact_new_changed_face_pairs':[list(x) for x in sorted(candpairs)],'target_aabb_candidates':ab,'layer_pass':layer_pass,'layer_failures':layer_fail,
   'preflight_shape':item.get('shape_summary'),'topology_boundary_preserved':item.get('fixed_patch_boundary_preserved',False)}
  rec['target_identity_improvement']=rec['mapped_removed_count']>0 and rec['mapped_added_count']==0
  rec['strict_raw_improvement']=rec['raw_removed_count']>rec['raw_added_count']
  reports.append(rec)
 # Prioritize no-added mapped identity and real mapped reduction, then raw reduction and layer validity.
 reports.sort(key=lambda x:(not x['target_identity_improvement'],not x['layer_pass'],-x['mapped_removed_count'],x['mapped_added_count'],-x['raw_removed_count'],x['raw_added_count'],x['edge_length_mm']))
 regression=next(x for x in reports if x['inside_node']==21095 and x['survivor']==18590)
 if regression['raw_43_16_before']!=247 or regression['raw_43_16_after']!=247 or regression['mapped_43_16_before']!=149 or regression['mapped_43_16_after']!=149 or regression['mapped_removed_count']!=0 or regression['mapped_added_count']!=0:
  raise ValueError('dd31 ranker regression failed against root exact full audit: '+str(regression))
 result={'schema':'numi.human.b43-16.local-edge-collapse-target-rank.v1','status':'TARGET_RANK_COMPLETE','baseline_sha256':PIN,'baseline159_sha256':sha(BASE159),'layer541_sha256':sha(LAYER),'audit159_source_sha256':sha(AUD159),'exact_predicate_sha256':sha(Path(exactmod.EXACT)),'target_bank':[43,16],'baseline_raw_pairs':len(base_pairs),'baseline_mapped_parent_pairs':len(base_mapped),'chart_shape_passed_candidates':len(items),'exact_baseline_pair_parity_on_candidate_rows':True,'dd31_21095_to_18590_full_audit_regression':{'raw_before':247,'raw_after':regression['raw_43_16_after'],'mapped_before':149,'mapped_after':regression['mapped_43_16_after'],'passed':True},'ranked_candidates':reports,'summary':{'candidates_with_raw_improvement':sum(x['strict_raw_improvement'] for x in reports),'candidates_with_mapped_identity_reduction':sum(x['target_identity_improvement'] for x in reports),'candidates_with_layer_pass':sum(x['layer_pass'] for x in reports),'candidates_with_mapped_reduction_and_layer_pass':sum(x['target_identity_improvement'] and x['layer_pass'] for x in reports)},'limitations':['Screen ranks one edge collapse at a time; it does not prove combined-candidate interaction.','Exact target test is limited to 43/16 changed triangles; self intersections and other banks are not evaluated here.','Layer checks use inherited 806 outer/inner/deep coordinates and test changed faces only.','Removed inside nodes remain unresolved physical-containment witnesses.'],'geometry_admitted':False,'physical_steps':0}
 RANK.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':result['status'],'candidate_count':len(items),'baseline_raw':len(base_pairs),'baseline_mapped':len(base_mapped),'summary':result['summary'],'top10':[{k:x[k] for k in ('inside_node','survivor','raw_43_16_after','raw_removed_count','raw_added_count','mapped_43_16_after','mapped_removed_count','mapped_added_count','layer_pass','layer_failures')} for x in reports[:10]],'path':str(RANK),'sha256':sha(RANK)},indent=2,sort_keys=True))

if __name__=="__main__": main()
