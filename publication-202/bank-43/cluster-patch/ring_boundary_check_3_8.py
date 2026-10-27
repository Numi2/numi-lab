#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027/b43-16/attempt-025-witness-cluster-topology-001'
BASE=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz'
PRE=W/'witness-cluster-preflight.json'
QUERY=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/audit-130/query'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE_SHA='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
PRE_SHA='93111ccab0e552476a8547983774fd02063c9f209829734573ef7f083e5f0971'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def topo(patch,F):
 ec=defaultdict(list)
 for fi in patch:
  tri=list(map(int,F[fi]))
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):ec[tuple(sorted((a,b)))].append(int(fi))
 edges=sorted(e for e,r in ec.items() if len(r)==1);ad=defaultdict(set)
 for a,b in edges:ad[a].add(b);ad[b].add(a)
 unseen=set(ad);loops=[]
 while unseen:
  q=[next(iter(unseen))];cc=set()
  while q:
   x=q.pop()
   if x in cc:continue
   cc.add(x);unseen.discard(x);q.extend(ad[x]-cc)
  loops.append(sorted(cc))
 hist={str(d):sum(len(v)==d for v in ad.values()) for d in sorted(set(map(len,ad.values())))}
 return edges,ad,loops,hist
def merge_patches(patches,nbr):
 parent=list(range(len(patches)))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[b]=a
 faceowners=defaultdict(set)
 for i,p in enumerate(patches):
  for f in p:faceowners[f].add(i)
 for owners in faceowners.values():
  owners=list(owners)
  for j in owners[1:]:union(owners[0],j)
 # Adjacent patch faces also merge to form connected local regions.
 for i,p in enumerate(patches):
  for f in p:
   for g in nbr[f]:
    for j in faceowners.get(g,()):
     if i!=j:union(i,j)
 groups=defaultdict(set)
 for i,p in enumerate(patches):groups[find(i)].update(p)
 return list(groups.values()),{find(i):i for i in range(len(patches))}
def exact_edges_all(edges,P,F,helper,exact):
 bankdir=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
 T,ids,gids,bodies,M,off,pos,keys,pins,_=helper.read_bank_inputs(bankdir,E)
 hit=[];cop=[];tested=0; local={}
 for body in sorted(set(map(int,ids[:,0]))):
  local[body]=(np.asarray(P,np.float64)-pos[body])@M[body].T-off[body]
 for body,bone in sorted(set((int(r[0]),int(r[1])) for r in ids)):
  mask=(ids[:,0]==body)&(ids[:,1]==bone);bank=np.asarray(T[mask],np.float64);lo=bank.min(1);hi=bank.max(1)
  btris=[];bnorm=[]
  for tri in bank:
   t=tuple(tuple(Fraction.from_float(float(x)) for x in pt) for pt in tri);btris.append(t);bnorm.append(exact._cross(exact._sub(t[1],t[0]),exact._sub(t[2],t[0])))
  for a,b in edges:
   pa=local[body][a];pb=local[body][b];elo=np.minimum(pa,pb);ehi=np.maximum(pa,pb)
   cand=np.flatnonzero(np.all(hi>=elo,axis=1)&np.all(lo<=ehi,axis=1))
   p=tuple(Fraction.from_float(float(x)) for x in pa);q=tuple(Fraction.from_float(float(x)) for x in pb)
   for ti in cand:
    tested+=1;tri=btris[ti];n=bnorm[ti]
    dp=exact._dot(n,exact._sub(p,tri[0]));dq=exact._dot(n,exact._sub(q,tri[0]))
    if dp==0 and dq==0:cop.append({'edge':[a,b],'body':body,'bone':bone,'triangle':int(ids[mask][ti,3])});continue
    if exact._segment_triangle(p,q,tri,n):hit.append({'edge':[a,b],'body':body,'bone':bone,'triangle':int(ids[mask][ti,3])})
 return {'tested_edge_triangle_pairs':tested,'exact_hit_count':len(hit),'coplanar_ambiguous_count':len(cop),'hits':hit[:100],'coplanar':cop[:100],'pass':not hit and not cop,'bank_count':len(keys),'bank_triangle_export_sha256':pins.get('exact-emitted.triangles.csv')}
def main():
 st=time.time()
 if sha(BASE)!=BASE_SHA or sha(PRE)!=PRE_SHA:raise ValueError('pinned base/preflight mismatch')
 with np.load(BASE,allow_pickle=False) as z:F=np.asarray(z['source_O_faces'],np.int64);P=np.asarray(z['points_world_m'],np.float32);ip=np.asarray(z['refinement_face_parent_ancestry_indptr']);ri=np.asarray(z['refinement_face_parent_ancestry_face_rows'])
 node_rows={int(r['query_row']):r for r in csv.DictReader((QUERY/'nodes.csv').open())}
 if len(node_rows)!=len(P):raise ValueError('a69 point query lacks complete node identities')
 nodes_npz=np.load(QUERY/'points-world.npz',allow_pickle=False)
 if not any(np.array_equal(np.asarray(nodes_npz[k],np.float32),P) for k in nodes_npz.files):raise ValueError('all-bank point query does not pin a69 point bits')
 pre=json.loads(PRE.read_text());components=pre['direct_edge_connected_witness_components'][1:]
 seedcomps=[set(map(int,c['seed_witness_face_rows'])) for c in components]
 edge_map=defaultdict(list)
 for fi,t in enumerate(F):
  a,b,c=map(int,t)
  for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):edge_map[e].append(fi)
 nbr=[set() for _ in range(len(F))]
 for rows in edge_map.values():
  if len(rows)>1:
   for f in rows:nbr[f].update(int(r) for r in rows if r!=f)
 hist=[];qualifying=[]
 for rings in range(3,9):
  patches=[]
  for seed in seedcomps:
   out=set(seed);front=set(seed)
   for _ in range(rings):
    nxt=set().union(*(nbr[f] for f in front))-out if front else set();out.update(nxt);front=nxt
   patches.append(out)
  merged,_=merge_patches(patches,nbr)
  level=[]
  for patch in merged:
   be,ad,loops,degh=topo(patch,F);bnodes=sorted(ad)
   outside=[v for v in bnodes if int(node_rows[v]['inside_banks'])==0 and int(node_rows[v]['ambiguous_banks'])==0 and int(node_rows[v]['invalid_banks'])==0]
   bad=[v for v in bnodes if v not in set(outside)]
   empty=[f for f in patch if int(ip[f])==int(ip[f+1])]
   parents=set()
   for f in patch:parents.update(map(int,ri[int(ip[f]):int(ip[f+1])]))
   row={'rings':rings,'face_count':len(patch),'seed_component_count':sum(bool(seed&patch) for seed in seedcomps),'boundary_node_count':len(bnodes),'boundary_edge_count':len(be),'boundary_loop_count':len(loops),'boundary_loop_sizes':[len(x) for x in loops],'degree_histogram':degh,'simple_single_loop':len(loops)==1 and bool(ad) and all(len(v)==2 for v in ad.values()),'all_boundary_nodes_outside_all185':len(outside)==len(bnodes),'nonclear_boundary_nodes':bad[:64],'nonclear_boundary_node_count':len(bad),'ancestry_empty_faces':len(empty),'immediate_parent_union_count':len(parents),'boundary_edges':be,'face_rows':sorted(patch)}
   level.append(row)
   if row['simple_single_loop'] and row['all_boundary_nodes_outside_all185'] and row['ancestry_empty_faces']==0:
    qualifying.append({'rings':rings,'patch':patch,'boundary_edges':be,'summary':row})
  hist.append({'rings':rings,'merged_component_count':len(level),'components':level})
 # Exact all-bank edge screen only on the smallest level with at least one simple clear component.
 edge_reps=[];chosen=None
 if qualifying:
  first=min(x['rings'] for x in qualifying)
  for q in [x for x in qualifying if x['rings']==first]:
   helper=mod('ring_allbank_helper',HELPER);owner=mod('ring_allbank_exact_owner',AUD159);exact=owner.exact_module()
   er=exact_edges_all(q['boundary_edges'],P,F,helper,exact)
   edge_reps.append({'rings':q['rings'],'component_summary':q['summary'],'exact_all185_boundary_edge_check':er})
   if er['pass'] and chosen is None:chosen={'rings':q['rings'],'component_summary':q['summary']}
 rep={'schema':'numi.human.b43-16.other-witness-ring-3-8-boundary-screen.v1','status':'CLEAR_SIMPLE_PATCH_FOUND' if chosen else 'NO_CLEAR_SIMPLE_PATCH_WITHIN_8_RINGS' if not qualifying else 'NODE_CLEAR_BUT_BOUNDARY_EDGE_OBSTRUCTED',
  'base_sha256':sha(BASE),'witness_preflight_sha256':sha(PRE),'tested_rings':list(range(3,9)),
  'seed_other_component_count':len(seedcomps),'per_ring_component_boundaries':hist,'candidate_clear_patches_at_minimum_level':edge_reps,
  'chosen_clear_patch':None if chosen is None else {k:v for k,v in chosen.items() if k!='patch'},
  'geometry_changed':False,'physical_steps':0,'geometry_admitted':False,
  'limitations':['All-bank node classifications are from the complete pinned a69 point query.','Exact boundary segment tests run against every typed bank triangle only for components with one simple boundary and all boundary nodes outside.'],
  'elapsed_seconds':time.time()-st}
 out=W/'other-witness-rings-3-8.json';out.write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':rep['status'],'rings':[{'rings':x['rings'],'merged_component_count':x['merged_component_count'],'simple_clear_count':sum(y['simple_single_loop'] and y['all_boundary_nodes_outside_all185'] and not y['ancestry_empty_faces'] for y in x['components']),'details':[{'face_count':y['face_count'],'boundary_nodes':y['boundary_node_count'],'loop_count':y['boundary_loop_count'],'simple':y['simple_single_loop'],'clear':y['all_boundary_nodes_outside_all185'],'nonclear_count':y['nonclear_boundary_node_count'],'nonclear_nodes':y['nonclear_boundary_nodes'][:12],'ancestry_empty':y['ancestry_empty_faces']} for y in x['components'] if y['simple_single_loop'] and y['all_boundary_nodes_outside_all185']]} for x in hist],'candidate_clear_patches_at_minimum_level':[{'rings':x['rings'],'edge_pass':x['exact_all185_boundary_edge_check']['pass'],'edge_hits':x['exact_all185_boundary_edge_check']['exact_hit_count'],'coplanar':x['exact_all185_boundary_edge_check']['coplanar_ambiguous_count']} for x in edge_reps]},indent=2))
 print('receipt_sha256',sha(out))
if __name__=='__main__':main()
