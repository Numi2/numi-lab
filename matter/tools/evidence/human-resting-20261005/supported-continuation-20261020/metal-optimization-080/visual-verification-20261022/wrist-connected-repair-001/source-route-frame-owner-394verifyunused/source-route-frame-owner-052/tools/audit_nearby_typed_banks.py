#!/usr/bin/env python3
import hashlib, importlib.util, json, pathlib, sys, time
import numpy as np

E=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'source-route-frame-owner-394verifyunused'
OUT=D/'source-route-frame-owner-052'
PAY=OUT/'payload'
BASE=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
BASE_MAN=BASE.with_suffix('.manifest.json')
CAND=PAY/'bodyparts3d-myosim-fullbody-muscle-surfaces-topology-134-v2.nhtissue'
CAND_MAN=PAY/'bodyparts3d-myosim-fullbody-muscle-surfaces-topology-134-v2.manifest.json'
MATERIAL=PAY/'nhtiss-materialization-v2-report.json'
SIDECAR=D/'fixed-boundary-remesh-008-star-fan-38385-retry003/topology-only-candidate.npz'
OLD_GATE=D/'fixed-boundary-remesh-008-star-fan-38385-retry003/exact-gate-report.json'
EMIT=E/'source-route-frame-owner-387/current_pose_preflight.py'
IDENT=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
CACHE=E/'registered-surrounding-tissue-native013-191'
INV=CACHE/'inventory.json'; CACHE_NPZ=CACHE/'registered-tissue-banks.npz'
OWNER=OUT/'payload/external-inputs/rebind_human_tissue_reference.py'
EXACT=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
REPORT=PAY/'nearby-typed-bank-exact-gate-current191.json'

PINS={
 'baseline_nhtiss':'8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042',
 'baseline_manifest':'ffe60f996e92325ce21123da86e259a02a9aaed1a874373a0e71721a2ceba07b',
 'candidate_nhtiss':'2840c0e356c2e2f7118bd37a0359a58efd028a36f2263219a5d9f407fbe2684d',
 'candidate_manifest':'29a4dd9a95ae273c8eb3a091456946f01ff38962f2f598af9d2857664a9ff6cb',
 'materialization_report':'adacbb7644de1e909c239df503ec559940b05420c8704a7cb6fcf9f60be66c47',
 'topology_sidecar':'fbff61585f65a01825d582fafd09147b5420aa84be19532c013f41d09e76945b',
 'old_six_pose_gate':'6e1075fddbb5ac560119e7e0d559059a17c71cddacbd29c95fc8f0c95f2dbb38',
 'emitter':'f34897a3cbad65313ee59b164a33cb528dfad0fe94e2f683fa1e49a1baecb201',
 'identity003':'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd',
 'inventory':'b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7',
 'cache_npz':'39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792',
 'exact_predicate':'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d',
}

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def require_pin(name,p):
 actual=sha(p)
 if actual!=PINS[name]:raise RuntimeError(f'{name} SHA mismatch: {actual} != {PINS[name]}')
 return actual

def bbox_rows(V,F,lo,hi):
 tri=np.asarray(V,np.float32)[np.asarray(F,np.int64)]
 a=tri.min(axis=1);b=tri.max(axis=1)
 return np.flatnonzero(np.all(b>=lo,axis=1)&np.all(a<=hi,axis=1)).astype(np.int64)

def records(ci,V,F,rows):
 rows=np.asarray(sorted(set(map(int,rows))),np.int64)
 if not len(rows):return []
 rs,_=ci.float32_surface_records(np.asarray(V,np.float32),np.asarray(F,np.int64)[rows])
 return [(r[0],r[1],r[2],int(row),r[4]) for r,row in zip(rs,rows)]

def exact_pairs(ci,a,b,same_surface):
 found=set()
 sides=((a,b),(b,a)) if same_surface else ((a,b),)
 for x,y in sides:
  if not x or not y:continue
  result=ci._audit_pair(x,y,same_surface=same_surface)
  for pair in result.get('triangle_pairs',[]):
   i,j=map(int,pair)
   found.add(tuple(sorted((i,j))) if same_surface else (i,j))
 return found

def main():
 start=time.time()
 paths={'baseline_nhtiss':BASE,'baseline_manifest':BASE_MAN,'candidate_nhtiss':CAND,'candidate_manifest':CAND_MAN,'materialization_report':MATERIAL,'topology_sidecar':SIDECAR,'old_six_pose_gate':OLD_GATE,'emitter':EMIT,'identity003':IDENT,'inventory':INV,'cache_npz':CACHE_NPZ,'exact_predicate':EXACT,'canonical_owner':OWNER}
 for name,p in paths.items():
  if not pathlib.Path(p).is_file():raise RuntimeError(f'missing pinned input {name}: {p}')
 hashes={name:require_pin(name,p) for name,p in paths.items() if name in PINS}
 material=json.loads(MATERIAL.read_text())
 if material['output']['sha256']!=hashes['candidate_nhtiss'] or material['physical_admission'] is not False:
  raise RuntimeError('candidate materialization receipt mismatch/admission flag')
 if material['old_mechanical_face_map'].find('INVALIDATED')<0:raise RuntimeError('old mechanical face map not explicitly invalidated')
 side=np.load(SIDECAR,allow_pickle=False)
 if int(side['stable_id'][0])!=134:raise RuntimeError('topology sidecar stable ID mismatch')
 patch=np.asarray(side['source_patch_face_rows'],np.int64)
 fan=np.asarray(side['fan_source_vertex_triangles'],np.int64)
 if (len(patch),len(fan))!=(18,8):raise RuntimeError(f'unexpected patch shape {len(patch)}/{len(fan)}')
 inv=json.loads(INV.read_text())
 identity=json.loads(IDENT.read_text())
 if inv.get('pose_path')!=str(IDENT):raise RuntimeError('registered inventory pose path mismatch')
 if inv.get('pose_sha256')!=hashes['identity003']:raise RuntimeError('registered inventory pose hash mismatch')
 banks=inv['banks']
 if len(banks)!=689:raise RuntimeError(f'inventory size changed: {len(banks)}')
 target_matches=[b for b in banks if b['kind']=='soft_tissue' and int(b['stable_id'])==134]
 if len(target_matches)!=1:raise RuntimeError(f'expected one soft-tissue stable134 bank, got {len(target_matches)}')
 target_bank=target_matches[0]
 if target_bank['key']!='bank_0133':raise RuntimeError(f'unexpected target cache key {target_bank["key"]}')
 # Execute the exact 191 current-pose emitter implementation, before its fixed eight-bank calls.
 emitter_source=EMIT.read_text(); marker='base=emit_selected(BASE)'
 if emitter_source.count(marker)!=1:raise RuntimeError('pinned emitter call boundary changed')
 ns={};exec(compile(emitter_source.split(marker)[0],str(EMIT),'exec'),ns)
 wanted={int(b['stable_id']) for b in banks if b['kind']=='soft_tissue'}
 if len(wanted)!=164:raise RuntimeError(f'expected 164 unique soft stable IDs, got {len(wanted)}')
 ns['WANTED']=wanted
 # Use the exact all-body current records pinned in identity003.
 ns['poses']={int(x['body_index']):x['current'] for x in identity['all_bodies']}
 baseline=ns['emit_selected'](BASE)
 candidate=ns['emit_selected'](CAND)
 if set(baseline)!=wanted or set(candidate)!=wanted:raise RuntimeError('emitted soft-tissue stable IDs differ from inventory')
 cache=np.load(CACHE_NPZ,allow_pickle=False)
 parity=[]
 for bank in banks:
  if bank['kind']!='soft_tissue':continue
  sid=int(bank['stable_id']);key=bank['key']
  bv,bf,_=baseline[sid];cv,cf,_=candidate[sid]
  kv=cache[key+'_vertices'];kf=cache[key+'_faces']
  if not np.array_equal(bv,kv) or not np.array_equal(bf,kf):
   raise RuntimeError(f'source NHTISS baseline is not exact current191 cache for stable={sid} key={key}')
  if not np.array_equal(cv,bv):raise RuntimeError(f'candidate changed emitted current vertices stable={sid}')
  if sid!=134 and not np.array_equal(cf,bf):raise RuntimeError(f'candidate changed topology outside stable134: {sid}')
  parity.append({'stable_id':sid,'key':key,'vertices':len(bv),'faces_baseline':len(bf),'faces_candidate':len(cf),'baseline_cache_vertex_bytes_exact':True,'baseline_cache_face_order_exact':True,'candidate_current_vertex_bytes_equal_baseline':True,'topology_unchanged':bool(sid==134 or np.array_equal(cf,bf))})
 V0,F0,_=baseline[134];V1,F1,_=candidate[134]
 if not np.array_equal(V0,V1):raise RuntimeError('stable134 target current coordinates changed')
 # Verify expected deterministic face-row mapping: eight fan faces inserted at the minimum removed row.
 removed=set(map(int,patch));insert=min(removed);fan_rows=set(range(insert,insert+len(fan)))
 expected=[]
 for row in range(len(F0)):
  if row==insert:expected.extend([None]*len(fan))
  if row not in removed:expected.append(row)
 if len(expected)!=len(F1):raise RuntimeError(f'candidate row-map length mismatch {len(expected)} != {len(F1)}')
 for out_row,src_row in enumerate(expected):
  if src_row is None:
   k=out_row-insert
   if not np.array_equal(F1[out_row],fan[k]):raise RuntimeError(f'fan output row {out_row} mismatch')
  elif not np.array_equal(F1[out_row],F0[src_row]):raise RuntimeError(f'unmodified source face remap mismatch at {src_row}->{out_row}')
 if not np.array_equal(F1[np.array(sorted(fan_rows))],fan):raise RuntimeError('candidate fan rows do not match declared fan')
 # Exact owner and source identity IDs; no seam quotient is used because vertex rows are untouched.
 sys.path.insert(0,str(EXACT.parent.parent))
 sys.path.insert(1,str(E/'exact-plane-filter-owner-264/src'))
 spec=importlib.util.spec_from_file_location('numilab_human.cardiac_cavity_intersections_starfan052',EXACT)
 ci=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ci;spec.loader.exec_module(ci)
 # Base comparison scope is fixed to the union AABB of the removed patch and new fan geometry.
 involved=np.unique(np.concatenate([F0[patch].reshape(-1),fan.reshape(-1)]))
 patch_xyz=V0[involved]
 lo=patch_xyz.min(axis=0);hi=patch_xyz.max(axis=0)
 nearby=[];not_near=[]
 for b in banks:
  blo=np.asarray(b['aabb_min_m'],np.float32);bhi=np.asarray(b['aabb_max_m'],np.float32)
  if np.all(bhi>=lo)&np.all(blo<=hi):nearby.append(b)
  else:not_near.append(b)
 if target_bank not in nearby:raise RuntimeError('target stable134 excluded from its own AABB query')
 # Target muscle self-pairs: compare removed source rows/fan rows against same-bank local triangle AABBs.
 old_target_rows=bbox_rows(V0,F0,lo,hi)
 new_target_rows=bbox_rows(V1,F1,lo,hi)
 old_patch_records=records(ci,V0,F0,patch)
 new_fan_records=records(ci,V1,F1,sorted(fan_rows))
 old_near_records=records(ci,V0,F0,old_target_rows)
 new_near_records=records(ci,V1,F1,new_target_rows)
 old_self=exact_pairs(ci,old_patch_records,old_near_records,True)
 new_self=exact_pairs(ci,new_fan_records,new_near_records,True)
 baseline_external=set();candidate_external=set();internal_new=[]
 for a,b in old_self:
  if a in removed and b in removed:continue
  if a in removed:baseline_external.add(b)
  elif b in removed:baseline_external.add(a)
 for a,b in new_self:
  ia=a in fan_rows;ib=b in fan_rows
  if ia and ib:internal_new.append([a,b])
  elif ia:
   if not (0<=b<len(expected)) or expected[b] is None:raise RuntimeError('candidate self partner has no source row mapping')
   candidate_external.add(int(expected[b]))
  elif ib:
   if not (0<=a<len(expected)) or expected[a] is None:raise RuntimeError('candidate self partner has no source row mapping')
   candidate_external.add(int(expected[a]))
 self_result={'baseline_raw_exact_pairs':len(old_self),'candidate_raw_exact_pairs':len(new_self),'baseline_patch_internal_pairs':sorted([list(x) for x in old_self if x[0] in removed and x[1] in removed]),'candidate_fan_internal_pairs':sorted(internal_new),'baseline_external_partner_source_rows':sorted(baseline_external),'candidate_external_partner_source_rows':sorted(candidate_external),'new_external_partner_source_rows':sorted(candidate_external-baseline_external),'resolved_external_partner_source_rows':sorted(baseline_external-candidate_external),'baseline_target_aabb_triangles':int(len(old_target_rows)),'candidate_target_aabb_triangles':int(len(new_target_rows))}
 # All nearby typed 191 banks, queried at exact current F32 coordinates with stable typed identity retained.
 nearby_results=[];total_base=total_cand=0
 for bank in nearby:
  kind=str(bank['kind']);sid=int(bank['stable_id']);key=str(bank['key'])
  if kind=='soft_tissue' and sid==134:continue
  bv=cache[key+'_vertices'];bf=cache[key+'_faces']
  if len(bv)!=int(bank['vertices']) or len(bf)!=int(bank['triangles']):raise RuntimeError(f'cache/inventory shape mismatch {key}')
  row_ids=bbox_rows(bv,bf,lo,hi)
  if not len(row_ids):
   nearby_results.append({'key':key,'kind':kind,'stable_id':sid,'layer':int(bank['layer']),'bank_aabb_overlaps_patch':True,'aabb_candidate_triangles':0,'baseline_raw_exact_pairs':0,'candidate_raw_exact_pairs':0,'new_partner_face_rows':[],'resolved_partner_face_rows':[]})
   continue
  bank_records=records(ci,bv,bf,row_ids)
  bp=exact_pairs(ci,old_patch_records,bank_records,False)
  cp=exact_pairs(ci,new_fan_records,bank_records,False)
  bp_rows={int(j) for i,j in bp};cp_rows={int(j) for i,j in cp}
  total_base+=len(bp);total_cand+=len(cp)
  nearby_results.append({'key':key,'kind':kind,'stable_id':sid,'layer':int(bank['layer']),'bank_aabb_overlaps_patch':True,'aabb_candidate_triangles':int(len(row_ids)),'baseline_raw_exact_pairs':len(bp),'candidate_raw_exact_pairs':len(cp),'new_partner_face_rows':sorted(cp_rows-bp_rows),'resolved_partner_face_rows':sorted(bp_rows-cp_rows),'baseline_partner_face_rows':sorted(bp_rows),'candidate_partner_face_rows':sorted(cp_rows)})
 new_bank_pairs=sum(len(x['new_partner_face_rows']) for x in nearby_results)
 status='PASS_LOCAL_CURRENT191_NEARBY_EXACT_NONREGRESSION' if not self_result['new_external_partner_source_rows'] and not internal_new and new_bank_pairs==0 else 'FAIL_LOCAL_CURRENT191_NEW_EXACT_PARTNER'
 report={'schema':'numi.human.nhtiss.stable134.topology-134-nearby-typed-bank-current191-exact.v1','status':status,'admission':'diagnostic_only_no_mechanical_or_physical_admission','scope':{'inventory_bank_count':len(banks),'nearby_bank_count_including_target':len(nearby),'non_nearby_bank_count':len(not_near),'soft_tissue_banks':sum(b['kind']=='soft_tissue' for b in banks),'anatomy_banks':sum(b['kind']=='anatomy' for b in banks),'query_pose':'identity003 all_bodies.current; exact 191 emitter expression; Float32 output','target_kind':target_bank['kind'],'target_stable_id':134,'target_cache_key':target_bank['key'],'target_patch_aabb_min_m':lo.tolist(),'target_patch_aabb_max_m':hi.tolist(),'nearby_bank_selection':'inventory bank AABB overlaps union AABB of old patch and fan geometry; triangle-level AABB then exact predicate','self_pair_query':'removed 18 source rows / new 8 fan output rows vs target bank triangles in same union AABB','external_query':'patch rows vs every AABB-overlapping typed bank in all 689 inventory entries; target stable134 self handled separately'},'input_hashes':hashes,'input_paths':{k:str(v) for k,v in paths.items()},'candidate_payload':{'sha256':hashes['candidate_nhtiss'],'surface_count':164,'vertex_count':467552,'index_count':2536812,'stable134_baseline_faces':int(len(F0)),'stable134_candidate_faces':int(len(F1)),'face_count_delta':int(len(F1)-len(F0)),'fan_output_face_rows':sorted(fan_rows),'removed_source_face_rows':sorted(removed),'source_vertices_and_bindings_unchanged':True,'stable134_current_coordinates_bitwise_baseline':True,'all_164_soft_tissue_baselines_match_registered191_cache_exactly':True,'all_164_candidate_current_vertex_arrays_match_baseline':True,'all_163_unaffected_soft_tissue_face_arrays_match_baseline':True,'candidate_to_source_face_map':expected,'old_mechanical_face_map':'invalidated; no old mechanical/material/FEM face map reused'},'soft_tissue_current191_parity':parity,'self_result':self_result,'nearby_typed_bank_results':nearby_results,'exact_counts':{'baseline_nearby_external_pairs':total_base,'candidate_nearby_external_pairs':total_cand,'new_nearby_partner_face_rows_across_banks':int(new_bank_pairs),'internal_candidate_fan_pairs':len(internal_new)},'gate_status':status,'elapsed_seconds':time.time()-start,'limitations':['This is a current-pose regional exact-intersection nonregression check against AABB-nearby entries from the 689-bank registered inventory; no full-scene all-pairs audit.','No rest or named joint-pose checks are repeated here; the prior six-pose local gate is separately pinned above.','No FEM/material remap, force evaluation, native physics, or physical/anatomical admission. Construction ancestry is not geometric overlap ancestry.']}
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':status,'report':str(REPORT),'report_sha256':sha(REPORT),'inventory_banks':len(banks),'nearby_banks':len(nearby),'self_baseline_pairs':len(old_self),'self_candidate_pairs':len(new_self),'nearby_baseline_pairs':total_base,'nearby_candidate_pairs':total_cand,'new_nearby_partner_rows':new_bank_pairs,'elapsed_seconds':report['elapsed_seconds'],'nearby':[{k:x[k] for k in ('key','kind','stable_id','aabb_candidate_triangles','baseline_raw_exact_pairs','candidate_raw_exact_pairs','new_partner_face_rows')} for x in nearby_results]},indent=2),flush=True)

if __name__=='__main__':main()
