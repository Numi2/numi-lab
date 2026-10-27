#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict, deque
from fractions import Fraction
import hashlib, importlib.util, json, sys, time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027/b43-16/attempt-025-witness-cluster-topology-001'
BASE=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz'
FAILED=S/'per-bank-workers-20261027/b43-16/attempt-024-witness-constrained-fit-001/candidate-private.npz'
SELF=S/'per-bank-workers-20261027/b43-16/attempt-024-witness-constrained-fit-001/exact-self-check.json'
BONE=S/'per-bank-workers-20261027/b43-16/attempt-024-witness-constrained-fit-001/scoped-43-16-crossings.json'
SELF_SHA='093b2b4741108b528bd29ff327551c69f306526b0a1b0a8ee36ae3ed3b81b362'
BONE_SHA='d4bc7090cb916364aa2706704f94281981d8a9c3b68ce487023ba918cd82dfe2'
BASE_SHA='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
FAILED_SHA='d21241ea6b16f2119faebda48196d79d4312d1316f7af5f7817e3780f145d8bb'
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
EXACT=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
AUD159=ROOT/'audit_skin_bone_surfaces_159.py'
QUERY=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/audit-130/query'
LAYER=S/'per-bank-workers-20261027/physical-layer-diagnostic-001/guarded-65-quarter/skin-fat-layers-541.npz'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def boundary_info(patch,faces):
 ec=defaultdict(list)
 for fi in patch:
  tri=list(map(int,faces[fi]))
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   ec[tuple(sorted((a,b)))].append(int(fi))
 bedges=sorted(e for e,rows in ec.items() if len(rows)==1)
 adj=defaultdict(set)
 for a,b in bedges:adj[a].add(b);adj[b].add(a)
 unseen=set(adj);loops=[]
 while unseen:
  seed=next(iter(unseen));q=[seed];cc=set()
  while q:
   u=q.pop()
   if u in cc:continue
   cc.add(u);unseen.discard(u);q.extend(adj[u]-cc)
  loops.append(sorted(cc))
 degs={int(v):len(nbr) for v,nbr in adj.items()}
 return bedges,degs,loops
def fpoint(p):return tuple(Fraction.from_float(float(x)) for x in p)
def main():
 started=time.time()
 if sha(BASE)!=BASE_SHA or sha(FAILED)!=FAILED_SHA or sha(SELF)!=SELF_SHA or sha(BONE)!=BONE_SHA:raise ValueError('pinned input SHA mismatch')
 sr=json.loads(SELF.read_text());br=json.loads(BONE.read_text())
 with np.load(BASE,allow_pickle=False) as z:
  P=np.asarray(z['points_world_m'],np.float32).astype(np.float64);F=np.asarray(z['source_O_faces'],np.int64)
  ip=np.asarray(z['refinement_face_parent_ancestry_indptr'],np.int64);pr=np.asarray(z['refinement_face_parent_ancestry_face_rows'],np.int64)
  sf=np.asarray(z['face_source_ancestry_indptr'],np.int64);sfr=np.asarray(z['face_source_ancestry_face_rows'],np.int64)
  df=np.asarray(z['face_fine_detail_ancestry_indptr'],np.int64);dfr=np.asarray(z['face_fine_detail_ancestry_rows'],np.int64)
  ref=np.asarray(z['area_reference_037_points_world_m_f64'],np.float64);lf=np.asarray(z['area_reference_037_source_O_faces'],np.int64)
 with np.load(FAILED,allow_pickle=False) as z:
  FF=np.asarray(z['source_O_faces'],np.int64);Pfailed=np.asarray(z['points_world_m'],np.float32)
 if not np.array_equal(F,FF):raise ValueError('witness candidate topology differs from pinned a69')
 self_pairs=np.asarray(sr['exact_self_pairs'],np.int64).reshape(-1,2)
 raw_added=np.asarray(br['added_raw_pairs'],np.int64).reshape(-1,2)
 witness_self=set(map(int,self_pairs.ravel())); witness_bone=set(map(int,raw_added[:,0]))
 seed=sorted(witness_self|witness_bone)
 # Full-face edge adjacency.
 edge_faces=defaultdict(list)
 for fi,tri in enumerate(F):
  a,b,c=map(int,tri)
  for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):edge_faces[e].append(fi)
 nbr=[set() for _ in range(len(F))]
 for rows in edge_faces.values():
  if len(rows)>1:
   for f in rows:
    nbr[f].update(r for r in rows if r!=f)
 # Witness-induced components by shared edge, then bounded 1/2 ring patches.
 remaining=set(seed); components=[]
 while remaining:
  root=min(remaining);q=[root];cc=set()
  while q:
   f=q.pop()
   if f in cc:continue
   cc.add(f);remaining.discard(f);q.extend((nbr[f]&set(seed))-cc)
  components.append(sorted(cc))
 def expand(comp,k):
  out=set(comp);front=set(comp)
  for _ in range(k):
   nxt=set().union(*(nbr[f] for f in front))-out if front else set()
   out.update(nxt);front=nxt
  return out
 def comp_boundary_loops(patch):
  be,de,loops=boundary_info(patch,F)
  loop_simple=(len(loops)==1 and bool(de) and all(x==2 for x in de.values()))
  return be,de,loops,loop_simple
 # Reuse complete a69 point classification table for all vertices and exact bank helper for edge witnesses.
 node_rows=list(__import__('csv').DictReader((QUERY/'nodes.csv').open()))
 if len(node_rows)!=len(P):raise ValueError('baseline exact point-query table does not cover every vertex')
 query_counts={int(r['query_row']):r for r in node_rows}
 qnpz=np.load(QUERY/'points-world.npz',allow_pickle=False)
 if not any(np.array_equal(np.asarray(qnpz[k],np.float32),P.astype(np.float32)) for k in qnpz.files):raise ValueError('baseline point query does not match a69 coordinates')
 h=mod('cluster_bank_helper',HELPER);owner=mod('cluster_audit159',AUD159);exact=owner.exact_module()
 bankdir=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
 T,ids,gids,bodies,M,off,pos,keys,pins,_=h.read_bank_inputs(bankdir,E)
 mask=(ids[:,0]==43)&(ids[:,1]==16);bank=np.asarray(T[mask],np.float64)
 local=(P-pos[43])@M[43].T-off[43]
 bank_tris=[];bank_norms=[];bank_lo=[];bank_hi=[]
 for tri in bank:
  t=tuple(fpoint(x) for x in tri);n=exact._cross(exact._sub(t[1],t[0]),exact._sub(t[2],t[0]))
  bank_tris.append(t);bank_norms.append(n);bank_lo.append(tri.min(0));bank_hi.append(tri.max(0))
 bank_lo=np.asarray(bank_lo);bank_hi=np.asarray(bank_hi)
 # Compute a69 current area/angle for patch shape diagnostics; no geometry moved.
 def shape(patch):
  tri=P[F[list(patch)]]
  e2=np.stack([np.sum((tri[:,1]-tri[:,0])**2,1),np.sum((tri[:,2]-tri[:,1])**2,1),np.sum((tri[:,0]-tri[:,2])**2,1)],axis=1)
  areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
  lens=np.sqrt(e2); sem=.5*lens.sum(1)
  num=np.stack([lens[:,0]**2+lens[:,2]**2-lens[:,1]**2,lens[:,0]**2+lens[:,1]**2-lens[:,2]**2,lens[:,1]**2+lens[:,2]**2-lens[:,0]**2],axis=1)
  den=2*np.stack([lens[:,0]*lens[:,2],lens[:,0]*lens[:,1],lens[:,1]*lens[:,2]],axis=1)
  angles=np.degrees(np.arccos(np.clip(num/den,-1,1)))
  asp=e2.max(1)/(4*np.sqrt(3)*np.maximum(areas,1e-30))
  return {'face_count':len(patch),'min_area_m2':float(areas.min(initial=np.inf)),'min_angle_deg':float(angles.min(initial=np.inf)),'max_aspect':float(asp.max(initial=0.))}
 def edge_checks(edges):
  hits=[];coplanar=[];tested=0
  for a,b in edges:
   pa=local[a];pb=local[b];lo=np.minimum(pa,pb);hi=np.maximum(pa,pb)
   cand=np.flatnonzero(np.all(bank_hi>=lo,axis=1)&np.all(bank_lo<=hi,axis=1))
   p=fpoint(pa);q=fpoint(pb)
   for ti in cand:
    tested+=1;tri=bank_tris[ti];n=bank_norms[ti]
    dp=exact._dot(n,exact._sub(p,tri[0]));dq=exact._dot(n,exact._sub(q,tri[0]))
    if dp==0 and dq==0:
     coplanar.append({'edge':[int(a),int(b)],'bone_triangle':int(ti)})
     continue
    pts=exact._segment_triangle(p,q,tri,n)
    if pts:hits.append({'edge':[int(a),int(b)],'bone_triangle':int(ti),'exact_hit_count':len(pts)})
  return {'candidate_pairs_tested':tested,'exact_crossing_pair_count':len(hits),'exact_crossing_pairs':hits[:100],'exact_crossing_pairs_truncated':len(hits)>100,'coplanar_ambiguous_pair_count':len(coplanar),'coplanar_ambiguous_pairs':coplanar[:100],'pass':not hits and not coplanar}
 reports=[]
 for ci,comp in enumerate(components):
  wrows=set(comp)
  entry={'component':ci,'seed_witness_face_rows':comp,'seed_face_count':len(comp),'self_witness_faces':sorted(wrows&witness_self),'new_bone_witness_faces':sorted(wrows&witness_bone)}
  levels=[]
  for k in (0,1,2):
   patch=expand(comp,k);be,de,loops,simple=comp_boundary_loops(patch)
   bnodes=sorted(de)
   clear_nodes=[int(v) for v in bnodes if int(query_counts[v]['inside_banks'])==0 and int(query_counts[v]['ambiguous_banks'])==0 and int(query_counts[v]['invalid_banks'])==0]
   nonclear=[int(v) for v in bnodes if int(query_counts[v]['inside_banks']) or int(query_counts[v]['ambiguous_banks']) or int(query_counts[v]['invalid_banks'])]
   exact_edge=edge_checks(be)
   empty_parent=[int(f) for f in patch if int(ip[f])==int(ip[f+1])]
   parents=set();sources=set();details=set()
   for f in patch:
    parents.update(map(int,pr[int(ip[f]):int(ip[f+1])]))
    sources.update(map(int,sfr[int(sf[f]):int(sf[f+1])]))
    details.update(map(int,dfr[int(df[f]):int(df[f+1])]))
   levels.append({'ring_expansion':k,'patch_face_count':len(patch),'boundary_edge_count':len(be),'boundary_node_count':len(bnodes),'boundary_loop_count':len(loops),'boundary_loop_sizes':[len(x) for x in loops],'boundary_degree_histogram':{str(d):sum(v==d for v in de.values()) for d in sorted(set(de.values()))},'simple_single_loop_boundary':simple,'all_boundary_nodes_outside_all185_from_pinned_a69_point_query':len(clear_nodes)==len(bnodes),'nonclear_boundary_nodes':nonclear,'exact_boundary_edge_target43_16':exact_edge,'ancestry_empty_rows':empty_parent,'immediate_parent_union_count':len(parents),'source_union_count':len(sources),'detail_union_count':len(details),'shape_a69':shape(patch),'patch_face_rows':sorted(patch)})
  entry['levels']=levels;reports.append(entry)
 ops=[19814,19816,19818,19819]
 overlap={str(f):sorted(set(ops)&set(comp)) for f in ops}
 rep={'schema':'numi.human.b43-16.topology_witness_component_preflight.v1','status':'TOPOLOGY_PREFLIGHT_ONLY','geometry_admitted':False,'physical_steps':0,
  'pins':{'a69_sha256':sha(BASE),'failed_candidate_sha256':sha(FAILED),'self_receipt_sha256':sha(SELF),'scoped_bone_receipt_sha256':sha(BONE),'bank_triangle_export_sha256':pins.get('exact-emitted.triangles.csv')},
  'witness_counts':{'exact_self_pairs':len(self_pairs),'exact_self_witness_faces':len(witness_self),'new_43_16_raw_bone_pairs':len(raw_added),'new_bone_witness_faces':len(witness_bone),'union_unique_witness_faces':len(seed),'self_pairs':self_pairs.tolist(),'new_43_16_raw_pairs':raw_added.tolist()},
  'direct_edge_connected_witness_components':reports,
  'existing_two_flip_rows':[19814,19816,19818,19819],
  'existing_two_flip_row_overlap':overlap,
  'limitations':['Witnesses are from failed alpha-0.5 trial; component grouping maps those face identities onto the pinned a69 mesh.','Clear-node test reuses the complete pinned a69 all-bank point query; exact segment-vs-triangle test is only for target body 43 stable bone 16.','No candidate topology was changed in this preflight.'],
  'elapsed_seconds':time.time()-started}
 out=W/'witness-cluster-preflight.json';out.write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':rep['status'],'witness_counts':rep['witness_counts']|{'self_pairs':None,'new_43_16_raw_pairs':None},'components':[{'component':x['component'],'seed_face_count':x['seed_face_count'],'levels':[{'ring_expansion':l['ring_expansion'],'patch_face_count':l['patch_face_count'],'boundary_loop_count':l['boundary_loop_count'],'boundary_node_count':l['boundary_node_count'],'simple_single_loop_boundary':l['simple_single_loop_boundary'],'all_boundary_nodes_outside_all185_from_pinned_a69_point_query':l['all_boundary_nodes_outside_all185_from_pinned_a69_point_query'],'nonclear_boundary_nodes':l['nonclear_boundary_nodes'][:12],'boundary_edges_exact_pass':l['exact_boundary_edge_target43_16']['pass'],'exact_boundary_crossing_pair_count':l['exact_boundary_edge_target43_16']['exact_crossing_pair_count'],'coplanar_ambiguous_pair_count':l['exact_boundary_edge_target43_16']['coplanar_ambiguous_pair_count'],'ancestry_empty_rows':len(l['ancestry_empty_rows']),'shape':l['shape_a69']} for l in x['levels']]} for x in reports]},indent=2))
 print('receipt_sha256',sha(out))
if __name__=='__main__':main()
