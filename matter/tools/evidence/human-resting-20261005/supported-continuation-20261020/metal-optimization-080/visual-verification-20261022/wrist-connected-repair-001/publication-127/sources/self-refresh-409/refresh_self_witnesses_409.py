#!/usr/bin/env python3
"""Refresh historical self-separation identities across four explicit local edge flips.

This emits construction-only pair seeds plus regenerated 026 rows at the frozen 6c115d
candidate. Exact whole-surface and all-185-bank audits remain separate authorities.
"""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
C=D/'D/local-two-parent-retriangulation-405/run-011-replay-405-four-flips/candidate.npz'
BASE=D/'D/postfit-backtrack-404/run-002/backtrack-373/pass-003/candidate.npz'
PRE=D/'barycentric-connected-clearance-170/fine-point-basis-enrichment-405/run-001/candidate.npz'
FLIP=D/'D/local-two-parent-retriangulation-405/run-011-replay-405-four-flips/report.json'
AUD130=D/'D/postfit-backtrack-405-topology-full-audits/run-001/audit-130/summary.json'
AUD159=D/'D/postfit-backtrack-405-topology-full-audits/run-001/audit-159/report.json'
SEEDS=D/'D/self-witness-refresh-406/merged-self-pair-seeds.npy'
SEED_REPORT=D/'D/self-witness-refresh-406/report.json'
OWNER=D/'barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py'
OUT=D/'D/self-witness-refresh-409/run-001'
EXPECTED={
 'candidate':'6c115d5beef83989bed2af6309771b8d34cece7e30eaaf3f99f397ee3a304fda',
 'base':'b41c56ac6ec2a99f7e1f2c6a02db10cf61b6cd04730f7a6bcdea65e224f80e52',
 'pre':'52ddadb6f6fbfd528ffd4fc72672a00aba8d40ad36914d17e92ed3b19d20c94a',
 'flip':'97d3b316003bceb7edf992809490a6fa0f05841326c6f7c4a4e280f61c11547a',
 'audit130':'bb81b34a092a468f456920fb0630afe576861390f960cffa5e1395b7f1eb3402',
 'audit159':'ec081def995f7fd59050e0c5f9e858b13177f3b53f416a81727c4b0b7c9238f0',
 'seeds':'9e9b286a0649c974fb03e20312dfdb5e4845927bf79ce81d70b6aa1edfb48d51',
 'seed_report':'74ab349293e07a1222469d2c8e68f36b7ff3177a118c6de857226559d593f265',
 'owner':'c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(path,key):
    got=sha(path)
    if got!=EXPECTED[key]: raise ValueError(f'{key} SHA mismatch: {got}')
    return got
def load_json(p): return json.loads(Path(p).read_text())
def csr_values(z,ipkey,rowkey,r):
    ip=z[ipkey]; vals=z[rowkey]
    return vals[int(ip[r]):int(ip[r+1])]
def save_npy_x(path,a):
    with Path(path).open('xb') as f: np.save(f,a,allow_pickle=False)
def save_json_x(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
def count_surface_pairs(report):
    """Count exact face pairs, allowing null results only for inactive banks in a complete 185-bank audit."""
    if not isinstance(report,dict) or report.get('complete') is not True:
        raise ValueError('159 report is incomplete')
    banks=report.get('banks')
    if not isinstance(banks,list) or len(banks)!=185 or int(report.get('bank_count',-1))!=185 or int(report.get('expected_bank_count',-1))!=185:
        raise ValueError('159 report does not contain exactly 185 banks')
    keys=set(); total=0
    for b in banks:
        key=(int(b['body']),int(b['stable_bone_id']))
        if key in keys: raise ValueError('duplicate typed 159 bank key')
        keys.add(key)
        result=b.get('result')
        if result is None:
            if int(b.get('skin_aabb_active',-1))!=0 or int(b.get('bank_aabb_active',-1))!=0:
                raise ValueError('null 159 result on a bank with active AABBs')
            continue
        count=int(result.get('count',-1)); pairs=result.get('triangle_pairs')
        if count<0 or not isinstance(pairs,list) or len(pairs)!=count:
            raise ValueError('159 result pair count does not match its explicit rows')
        total+=count
    return total

for key,path in [('candidate',C),('base',BASE),('pre',PRE),('flip',FLIP),('audit130',AUD130),('audit159',AUD159),('seeds',SEEDS),('seed_report',SEED_REPORT),('owner',OWNER)]: require(path,key)
OUT.mkdir(parents=True,exist_ok=False)
base=np.load(BASE,allow_pickle=False); pre=np.load(PRE,allow_pickle=False); cur=np.load(C,allow_pickle=False)
flip=load_json(FLIP); s130=load_json(AUD130); s159=load_json(AUD159); seed_report=load_json(SEED_REPORT)
old=np.load(SEEDS,allow_pickle=False)
if flip.get('candidate_input_sha256')!=EXPECTED['pre'] or flip.get('candidate_output_sha256')!=EXPECTED['candidate']:
    raise ValueError('flip report candidate lineage mismatch')
if s130.get('candidate_sha256')!=EXPECTED['candidate'] or int(s130.get('inside_nodes',-1))<0:
    raise ValueError('130 point/self report candidate mismatch')
if int(s130.get('exact_PLC_surface_pairs',-1))!=0: raise ValueError('frozen candidate does not have zero exact self-pairs')
if s159.get('candidate_sha256')!=EXPECTED['candidate'] or s159.get('complete') is not True or int(s159.get('bank_count',-1))!=185 or int(s159.get('expected_bank_count',-1))!=185:
    raise ValueError('159 report is not complete/matched across 185 banks')
if seed_report.get('candidate_sha256')!=EXPECTED['base'] or int(seed_report.get('merged_pair_count',-1))!=len(old):
    raise ValueError('406 seed report does not match frozen old seed file')
if old.ndim!=2 or old.shape[1]!=2 or len(old)!=17865 or np.any(old<0) or np.any(old>=len(base['source_O_faces'])):
    raise ValueError('unexpected old seed shape/domain')
# Add every failed exact self-pair witness set from 406 passes 1-6.
# Each audit binds its pair array to its candidate; each candidate must retain
# the exact 404 face-row topology before pair identities can be remapped.
EXTRA=[
 {'pass':'001','candidate_sha256':'c81268dbc7ca14924f2b9d908ae750d6c771dc26a307f5623125449d527493ea','pairs_sha256':'ce491925801f5a6ff2e72bacbeddde75b15c65a11d8b28421a07526cffd8ce2d','summary_sha256':'46ffa1fe786959d704f144157e2fbedf8a90ed7acf1bf6753f082c36cf720947'},
 {'pass':'002','candidate_sha256':'077f1389d8d4ea926088d195f978e33ce4e73b0132393fbfe8cfeabbb976482e','pairs_sha256':'37329b06f033cd26819f5a654e892ec425f5acb749969713fe37c3cc72487249','summary_sha256':'8b1d2a0adf2f31ed46c6d72eeaa36cf51f63e1eea760af787bdab9b075101612'},
 {'pass':'003','candidate_sha256':'2c5205ca673394e782ab83453ea53973eaf095bc541db3a32eaac42fe4d99a0b','pairs_sha256':'172666b21fa921b9767b30965a3da78474f84dc6bdc9f0314fda37d22bd03d55','summary_sha256':'fdbe16edbafe25edced688b72de8e5ef193a2716bca5a1c9e73b333845aef949'},
 {'pass':'004','candidate_sha256':'a9510a40f5a222e198e9cdbc9b7d02dd764680da82062798cf6e39fc6a819c8c','pairs_sha256':'dafa9ed925913e7f7b99d362800afd5ca12bb48ff6cfbeb1c8d059413331dcf8','summary_sha256':'6f43d186b6c7c834168f3a0d80cfde1bc9e5d50f79e1ee4dc0cd41d45f6a7f77'},
 {'pass':'005','candidate_sha256':'71b041644e057a02a7c3956111b39cdb3f3aae6e4080446939cadc526b001465','pairs_sha256':'fe015716d108e531ee9d78938d8807ce0139157daea63ee35d11dbd8e52e5447','summary_sha256':'073abb9e9ac80a9053ae2782db499fa3782cf48d7795c6720e97dcb7a7952fce'},
 {'pass':'006','candidate_sha256':'e76fbd37431055e1f5c5df547dd69fa4302d912803602ab098d4908456a93c26','pairs_sha256':'fe015716d108e531ee9d78938d8807ce0139157daea63ee35d11dbd8e52e5447','summary_sha256':'d641b0b268277015392676e2490236200a8bee9ecafc993ad93103553447283d'},
]
extra_arrays=[]; extra_source_receipts=[]
for item in EXTRA:
    p=D/f"D/postfit-backtrack-406/run-001/backtrack-373/pass-{item['pass']}"
    cp=p/'candidate.npz'; pp=p/'audit-130/surface-self-pairs.npy'; sp=p/'audit-130/summary.json'
    for q,key in ((cp,'candidate_sha256'),(pp,'pairs_sha256'),(sp,'summary_sha256')):
        if sha(q)!=item[key]: raise ValueError(f"406 pass {item['pass']} {key} mismatch")
    cand=np.load(cp,allow_pickle=False); pairs=np.load(pp,allow_pickle=False); summary=load_json(sp)
    if summary.get('candidate_sha256')!=item['candidate_sha256'] or int(summary.get('exact_PLC_surface_pairs',-1))!=len(pairs): raise ValueError(f"406 pass {item['pass']} audit identity/count mismatch")
    if not np.array_equal(cand['source_O_faces'],base['source_O_faces']): raise ValueError(f"406 pass {item['pass']} face topology differs from 404 seed domain")
    if pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0) or np.any(pairs>=len(base['source_O_faces'])): raise ValueError(f"406 pass {item['pass']} malformed pair indices")
    extra_arrays.append(pairs.astype(np.int64,copy=False))
    extra_source_receipts.append({'pass':item['pass'],'candidate_path':str(cp),'candidate_sha256':item['candidate_sha256'],'pairs_path':str(pp),'pairs_sha256':item['pairs_sha256'],'pair_count':int(len(pairs)),'summary_path':str(sp),'summary_sha256':item['summary_sha256']})
merged_preflip=np.concatenate([old.astype(np.int64,copy=False),*extra_arrays],axis=0)
merged_preflip=np.unique(np.sort(merged_preflip,axis=1),axis=0)
if merged_preflip.shape[1]!=2 or np.any(merged_preflip[:,0]>=merged_preflip[:,1]): raise ValueError('malformed merged pre-flip seed set')
if not np.array_equal(base['source_O_faces'],pre['source_O_faces']): raise ValueError('pre-flip input does not preserve old face-row topology')
if not np.array_equal(cur['points_world_m'],pre['points_world_m']): raise ValueError('edge flip changed coordinates')
if cur['source_O_faces'].shape!=base['source_O_faces'].shape: raise ValueError('face dimensions changed')
changed=np.flatnonzero(np.any(cur['source_O_faces']!=base['source_O_faces'],axis=1))
reported=np.asarray(flip.get('changed_face_rows',[]),dtype=np.int64)
if not np.array_equal(changed,np.sort(reported)) or len(changed)!=8: raise ValueError('changed-face report does not exactly cover candidate topology delta')
# Verify the full flip operation ledger against old/new triangles and source ancestry.
ops=flip.get('operations',[])
if len(ops)!=4: raise ValueError('expected exactly four explicit edge-flip operations')
groups=[]; old_to_new={}
for op in ops:
    rows=list(map(int,op['face_rows']))
    if len(rows)!=2 or any(r not in set(map(int,changed)) for r in rows): raise ValueError('invalid changed-face operation rows')
    if not np.array_equal(base['source_O_faces'][rows],np.asarray(op['expected_old_faces'],dtype=np.int64)):
        raise ValueError('operation old triangles do not match frozen base')
    if not np.array_equal(cur['source_O_faces'][rows],np.asarray(op['new_faces'],dtype=np.int64)):
        raise ValueError('operation new triangles do not match frozen candidate')
    src_old=set()
    sub_old=set()
    src_new=set()
    sub_new=set()
    for r in rows:
        for z,srcset,subset in ((base,src_old,sub_old),(cur,src_new,sub_new)):
            srcset.update(map(int,csr_values(z,'face_source_ancestry_indptr','face_source_ancestry_face_rows',r)))
            subset.update(map(int,csr_values(z,'face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows',r)))
        old_to_new[r]=rows
    if src_old!=src_new or sub_old!=sub_new: raise ValueError('flip group source/subface ancestry union changed')
    parent_union=set(map(int,op['immediate_parent_union']))
    for r in rows:
        if set(map(int,flip['changed_face_parent_unions'][str(r)]))!=parent_union:
            raise ValueError('flip report parent union differs across a replacement group')
        if not np.array_equal(csr_values(cur,'refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows',r),np.asarray(sorted(parent_union),dtype=np.int64)):
            raise ValueError('replacement face immediate-parent CSR does not match flip report')
    key=str(rows[0])
    if set(map(int,flip['changed_face_source_unions'][key]))!=src_new or set(map(int,flip['changed_face_subface_unions'][key]))!=sub_new:
        raise ValueError('flip report ancestry union does not match NPZ CSR')
    groups.append({'face_rows':rows,'edge':list(map(int,op['edge'])),'source_detail_face_row':int(op['source_detail_face_row']),'source_subface_row':int(op['source_subface_row']),'source_ancestry_union':sorted(src_new),'subface_ancestry_union':sorted(sub_new),'old_triangles':np.asarray(op['expected_old_faces'],dtype=np.int64).tolist(),'new_triangles':np.asarray(op['new_faces'],dtype=np.int64).tolist()})
if set(old_to_new)!=set(map(int,changed)): raise ValueError('not all changed rows have explicit operation mapping')
# Every untouched face row retains exact topology and source ancestry identity.
unchanged_rows=np.setdiff1d(np.arange(len(base['source_O_faces']),dtype=np.int64),changed)
if not np.array_equal(base['source_O_faces'][unchanged_rows],cur['source_O_faces'][unchanged_rows]): raise ValueError('unexpected non-flip face topology changes')
for ipk,rk in [('face_source_ancestry_indptr','face_source_ancestry_face_rows'),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows')]:
    for r in unchanged_rows:
        if not np.array_equal(csr_values(base,ipk,rk,int(r)),csr_values(cur,ipk,rk,int(r))): raise ValueError(f'unchanged row ancestry changed: {r} {rk}')
# Reinterpret changed historical witness rows through each exact 2-face flip group.
canonical=merged_preflip
touched_mask=np.isin(canonical,changed).any(axis=1)
touched=canonical[touched_mask]; unaffected=canonical[~touched_mask]
expanded=[]; touched_ledger=[]; shared_edge_drops=[]
faces=cur['source_O_faces']
for a,b in canonical:
    ma=old_to_new.get(int(a),[int(a)]); mb=old_to_new.get(int(b),[int(b)])
    generated=[]; dropped=[]
    for x in ma:
      for y in mb:
        if x==y: continue
        lo,hi=sorted((int(x),int(y)))
        common=set(map(int,faces[lo]))&set(map(int,faces[hi]))
        if len(common)>=2:
            dropped.append({'faces':[lo,hi],'shared_vertices':sorted(common),'reason':'intended shared-edge adjacency after exact flip'})
            shared_edge_drops.append((lo,hi))
            continue
        generated.append([lo,hi]); expanded.append([lo,hi])
    if int(a) in old_to_new or int(b) in old_to_new:
        touched_ledger.append({'old_seed_pair':[int(a),int(b)],'replayed_new_pair_rows':sorted({tuple(x) for x in generated}),'superseded_adjacency_rows':dropped})
expanded=np.asarray(expanded,dtype=np.int64).reshape((-1,2))
expanded=np.unique(np.sort(expanded,axis=1),axis=0)
if len(expanded)==0 or np.any(expanded[:,0]>=expanded[:,1]): raise ValueError('empty or malformed refreshed pair set')
# Exact unchanged seeds must survive in the refreshed set.
if not set(map(tuple,unaffected.tolist())).issubset(set(map(tuple,expanded.tolist()))): raise ValueError('unaffected historical pair identities were not preserved')
# Rebuild all rows at the new candidate coordinates, including touched face groups.
sp=importlib.util.spec_from_file_location('self_separation_026_refresh407',OWNER)
mod=importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
rows,records=mod.build_pair_rows(cur['points_world_m'].astype(np.float64)*1000.0,faces,expanded,margin_fraction=0.25)
max_rhs=max((float(r['lower']) for r in rows),default=0.0)
if max_rhs>2e-8: raise ValueError(f'zero-step rebuilt self rows infeasible: {max_rhs}')
save_npy_x(OUT/'merged-self-pair-seeds.npy',expanded)
save_json_x(OUT/'changed-face-replay.json',{'candidate_sha256':EXPECTED['candidate'],'base_candidate_sha256':EXPECTED['base'],'flip_report_sha256':EXPECTED['flip'],'changed_face_rows':changed.tolist(),'flip_groups':groups,'old_touched_pair_count':len(touched),'old_untouched_pair_count':len(unaffected),'touched_pair_replay':touched_ledger,'expanded_shared_edge_adjacency_drops':shared_edge_drops})
report={
 'schema':'numi.human.self-witness-refresh-407.v1','status':'PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY',
 'candidate_path':str(C),'candidate_sha256':EXPECTED['candidate'],'candidate_points_shape':list(cur['points_world_m'].shape),'candidate_faces_shape':list(faces.shape),
 'base_candidate_path':str(BASE),'base_candidate_sha256':EXPECTED['base'],'preflip_candidate_sha256':EXPECTED['pre'],'flip_report_path':str(FLIP),'flip_report_sha256':EXPECTED['flip'],
 'matched_audit_130_path':str(AUD130),'matched_audit_130_sha256':EXPECTED['audit130'],'matched_audit_159_path':str(AUD159),'matched_audit_159_sha256':EXPECTED['audit159'],'matched_audit_159_pairs':count_surface_pairs(s159),
 'input_seed_path':str(SEEDS),'input_seed_sha256':EXPECTED['seeds'],'input_pair_count':int(len(canonical)),'original_406_seed_pair_count':int(len(old)),'additional_406_failed_pass_pair_rows':int(sum(x['pair_count'] for x in extra_source_receipts)),'additional_406_failed_pass_unique_pairs':int(len(canonical)-len(old)),'additional_406_failed_pass_audits':extra_source_receipts,
 'changed_face_rows':changed.tolist(),'changed_face_group_count':len(groups),'changed_input_pair_count':int(len(touched)),'unchanged_input_pair_count':int(len(unaffected)),
 'replay_method':'Exact flip-operation row groups from the pinned report. For each historical pair touching a flipped row, replace that endpoint with both new triangles covering the same source/subface ancestry union; preserve unchanged pair identities; omit only resulting pairs sharing an intended edge. Rebuild every retained constraint from current candidate F32 coordinates using owner 026.',
 'expanded_unique_pair_count':int(len(expanded)),'replayed_unique_pair_count':int(len(expanded)-len(unaffected)),'shared_edge_adjacency_drops':int(len(shared_edge_drops)),
 'pair_owner_path':str(OWNER),'pair_owner_sha256':EXPECTED['owner'],'rebuilt_linear_row_count':int(len(rows)),'rebuilt_record_count':int(len(records)),'margin_fraction':0.25,'zero_step_max_rhs_mm':max_rhs,
 'output_pairs_path':str(OUT/'merged-self-pair-seeds.npy'),'output_pairs_sha256':sha(OUT/'merged-self-pair-seeds.npy'),'changed_face_replay_path':str(OUT/'changed-face-replay.json'),'changed_face_replay_sha256':sha(OUT/'changed-face-replay.json'),
 'complete_current_self_audit_zero':True,'native_admission':False,'accepted_physical_steps':0,
 'limitations':['Historical pair identities are construction guards only; they are not a complete self-intersection certificate.','The matched exact self audit is zero at the frozen candidate; all exact audits must rerun on every fitted trial.','No geometry admission, native or physical qualification is claimed.'],
 'input_hashes':EXPECTED,
}
save_json_x(OUT/'report.json',report)
print(json.dumps({k:report[k] for k in ['status','input_pair_count','changed_input_pair_count','unchanged_input_pair_count','expanded_unique_pair_count','shared_edge_adjacency_drops','rebuilt_linear_row_count','zero_step_max_rhs_mm','output_pairs_sha256']}))
