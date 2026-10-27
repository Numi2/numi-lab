#!/usr/bin/env python3
"""Topology-only complement check for bank 43/16 patch; no geometry changes."""
from pathlib import Path
from collections import defaultdict,deque,Counter
import json,hashlib
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/candidate.npz');A=C.parent/'audit-159/report.json';D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/ring-expansion.json');O=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/complement-fill-check.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(C,allow_pickle=False) as z:F=np.asarray(z['source_O_faces'],np.int64)
base=json.loads(A.read_text());bank=next(b for b in base['banks'] if (b['body'],b['stable_bone_id'])==(109,114));pairs=[tuple(map(int,p)) for p in bank['result']['triangle_pairs']];cross={sf for sf,bf in pairs}
nodefaces=defaultdict(list);edgefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t:nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
# Reconstruct exact step-2 patch: crossed faces + first vertex ring + two complete branch-vertex fan fills.
patch=set(cross)
for f in tuple(cross):
 for v in F[f]:patch.update(nodefaces[int(v)])
def boundaries(patch):
 out=[]
 for e,fs in edgefaces.items():
  inside=[f for f in fs if f in patch];outside=[f for f in fs if f not in patch]
  if inside and outside:out.append((e,outside))
  elif inside and len(fs)==1:out.append((e,[]))
 return out
def loop_sets(bedges):
 g=defaultdict(list);e_for_node=defaultdict(list)
 for e,_ in bedges:
  u,v=e;g[u].append(v);g[v].append(u);e_for_node[u].append(e);e_for_node[v].append(e)
 unseen=set(g);loops=[]
 while unseen:
  st=min(unseen);unseen.remove(st);q=[st];nodes=[]
  while q:
   v=q.pop();nodes.append(v)
   for w in g[v]:
    if w in unseen:unseen.remove(w);q.append(w)
  ns=set(nodes);es=sorted(e for e,_ in bedges if e[0] in ns and e[1] in ns)
  loops.append({'nodes':sorted(ns),'edges':es})
 return loops,g
for _ in range(2):
 b=boundaries(patch);loops,g=loop_sets(b);branched=[v for v in g if len(g[v])>2]
 add=set()
 for v in branched:add.update(nodefaces[int(v)])
 patch.update(add)
# Recompute boundary loops after prescribed expansions.
b=boundaries(patch);loops,g=loop_sets(b)
# Complement face adjacency components by shared edges.
comp_faces=set(range(len(F)))-patch;adj=defaultdict(set)
for e,fs in edgefaces.items():
 ins=[f for f in fs if f in comp_faces]
 if len(ins)==2:adj[ins[0]].add(ins[1]);adj[ins[1]].add(ins[0])
 elif len(ins)>2:
  for u in ins:
   adj[u].update(v for v in ins if v!=u)
comps=[];comp_of={};unseen=set(comp_faces)
while unseen:
 st=min(unseen);unseen.remove(st);q=[st];cf=[]
 while q:
  f=q.pop();cf.append(f)
  for n in adj[f]&unseen:unseen.remove(n);q.append(n)
 ci=len(comps)
 for f in cf:comp_of[f]=ci
 comps.append(sorted(cf))
# Vertex-only contact between distinct complement components is reported as ambiguity.
vertex_component_contacts=[]
for v,fs in nodefaces.items():
 cset=sorted({comp_of[f] for f in fs if f in comp_of})
 if len(cset)>1:vertex_component_contacts.append({'node':v,'components':cset})
loop_to_comp=[];comp_boundary=defaultdict(list)
for li,L in enumerate(loops):
 edge_records=[];cs=set()
 for e in L['edges']:
  out=[f for f in edgefaces[e] if f not in patch]
  ids=sorted({comp_of[f] for f in out if f in comp_of});cs.update(ids)
  edge_records.append({'edge':list(e),'outside_faces':out,'complement_components':ids})
  for cid in ids:comp_boundary[cid].append((li,e))
 loop_to_comp.append({'loop':li,'node_count':len(L['nodes']),'edge_count':len(L['edges']),'complement_components':sorted(cs),'edge_ownership':edge_records})
largest=max(range(len(comps)),key=lambda i:len(comps[i])) if comps else None
# A fill candidate must be a non-largest complement component with one loop, exact boundary coverage of that loop,
# no other component sharing one of its vertices, and <10% of the largest component.
selected=[];criteria=[]
for ci,fs in enumerate(comps):
 own=comp_boundary.get(ci,[]);loops0=sorted({li for li,e in own});sizes=[len(comps[j]) for j in range(len(comps)) if j!=ci]
 shared_vertex=any(ci in x['components'] and len(x['components'])>1 for x in vertex_component_contacts)
 loop_ok=False
 if len(loops0)==1:
  li=loops0[0];loop_edges=set(map(tuple,loops[li]['edges']));owned_edges={e for l,e in own if l==li};loop_ok=owned_edges==loop_edges
 largest_size=len(comps[largest]) if largest is not None else 0
 small=ci!=largest and largest_size>0 and len(fs)<0.10*largest_size
 ok=bool(loop_ok and not shared_vertex and small)
 criteria.append({'component':ci,'face_count':len(fs),'boundary_loops':loops0,'owned_boundary_edge_count':len(own),'single_loop_owns_full_boundary':loop_ok,'shares_vertex_with_other_complement_component':shared_vertex,'is_largest_body_component':ci==largest,'less_than_10_percent_largest':small,'unambiguous_small_hole_candidate':ok})
 if ok:selected.append(ci)
filled=set(patch)
for ci in selected:filled.update(comps[ci])
filled_boundary=boundaries(filled);filled_loops,fg=loop_sets(filled_boundary)
filled_cross_faces=cross&filled;filled_pairs=[p for p in pairs if p[0] in filled]
report={'schema':'numi.human.bank109_114.topology_only_complement_fill_check.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_sha256':sha(A),'ring_expansion_sha256':sha(D),'typed_bank':[109,114],'input_crossings':{'exact_pair_count':len(pairs),'crossing_skin_face_count':len(cross)},'base_patch':{'face_count':len(patch),'boundary_loop_count':len(loops),'boundary_loop_node_counts':sorted(len(x['nodes']) for x in loops),'boundary_degree_histogram':dict(Counter(len(g[v]) for v in g)),'boundary_non_degree_two_nodes':sorted((int(v),len(g[v])) for v in g if len(g[v])!=2)},'complement_components':{'count':len(comps),'largest_component_id':largest,'largest_component_face_count':len(comps[largest]) if largest is not None else 0,'vertex_only_component_contacts':vertex_component_contacts,'component_stats':[{'component':i,'face_count':len(fs),'adjacent_patch_boundary_loops':sorted({li for li,e in comp_boundary.get(i,[])}),'adjacent_patch_boundary_edges':len(comp_boundary.get(i,[]))} for i,fs in enumerate(comps)],'loop_to_complement':loop_to_comp,'fill_eligibility':criteria},'topology_only_fill':{'selected_complement_component_ids':selected,'added_face_count':sum(len(comps[i]) for i in selected),'added_face_rows':sorted(f for i in selected for f in comps[i]),'resulting_patch_face_count':len(filled),'resulting_boundary_loop_count':len(filled_loops),'resulting_boundary_loop_node_counts':sorted(len(x['nodes']) for x in filled_loops),'resulting_boundary_degree_histogram':dict(Counter(len(fg[v]) for v in fg)),'exact_crossing_faces_contained':len(filled_cross_faces),'exact_crossing_faces_total':len(cross),'exact_cross_pairs_owned':len(filled_pairs),'exact_cross_pairs_total':len(pairs),'all_crossings_still_inside':len(filled_pairs)==len(pairs)},'geometry_moved':False,'shared_candidate_modified':False,'limitation':'Face-set topology diagnostic only; no coordinates moved, no candidate edited, no anatomical repair or clearance claim.'}
O.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
np.save(Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/complement-filled-patch-face-rows.npy'),np.asarray(sorted(filled),np.int64))
