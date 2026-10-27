from pathlib import Path
from fractions import Fraction
from collections import defaultdict,deque,Counter
import hashlib,importlib.util,json,sys,time,csv
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
S=ROOT/"D/surface-clearance-546/component-sat-selector-638"
W=S/"per-bank-workers-20261027"
D=W/"b109-114/attempt-031-alpha-scaled-progress-screen-001"
BASE=W/"combined-d071-b109-b61-001/candidate.npz"
AUD159=W/"combined-d071-b109-b61-001/audit-159/report.json"
AUD130=W/"combined-d071-b109-b61-001/audit-130/query/pairs.csv"
AUDIT=ROOT/"audit_skin_bone_surfaces_159.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BODY=109;BONE=114
SELF_PAIRS=[tuple(map(int,r)) for r in np.load(D/"audit-130/surface-self-pairs.npy")]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);x=importlib.util.module_from_spec(spec);sys.modules[name]=x;spec.loader.exec_module(x);return x
def frac(x):return Fraction.from_float(float(x))
def rational_rows(tri):
 return [tuple(frac(x) for x in row) for row in np.asarray(tri,dtype=np.float64)]
def bary(p,t):
 p=np.asarray([float(x) for x in p]);t=np.asarray([[float(x) for x in row] for row in t]);v0=t[1]-t[0];v1=t[2]-t[0];v2=p-t[0];d00=np.dot(v0,v0);d01=np.dot(v0,v1);d11=np.dot(v1,v1);d20=np.dot(v2,v0);d21=np.dot(v2,v1);den=d00*d11-d01*d01
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
 return [float(1-v-w),float(v),float(w)]
def fpoint(p):return [[int(x.numerator),int(x.denominator)] for x in p]
def bkey(p):return tuple((int(x.numerator),int(x.denominator)) for x in p)
def topology(F,seed,inside):
 edge_faces=defaultdict(list)
 for fi,t in enumerate(F):
  for j in range(3):edge_faces[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))].append(fi)
 adj=[set() for _ in range(len(F))]
 for uses in edge_faces.values():
  for f in uses:adj[f].update(g for g in uses if g!=f)
 def comps(fs):
  rem=set(fs);out=[]
  while rem:
   r=min(rem);rem.remove(r);q=deque([r]);c=[]
   while q:
    f=q.popleft();c.append(f)
    for g in adj[f]&rem:rem.remove(g);q.append(g)
   out.append(sorted(c))
  return sorted(out,key=lambda c:(-len(c),c[0]))
 rings=[];prev=set(seed)
 for radius in range(5):
  if radius==0:patch=set(seed)
  else:patch=prev
  cs=comps(patch)
  edge_counts=defaultdict(int)
  for fi in patch:
   t=F[fi]
   for j in range(3):edge_counts[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))]+=1
  be=[e for e,n in edge_counts.items() if n==1]
  vg=defaultdict(set)
  for u,v in be:vg[u].add(v);vg[v].add(u)
  unseen=set(vg);bc=[]
  while unseen:
   r=min(unseen);unseen.remove(r);q=deque([r]);c=[]
   while q:
    u=q.popleft();c.append(u)
    for v in vg[u]&unseen:unseen.remove(v);q.append(v)
   bc.append(sorted(c))
  rings.append({'radius':radius,'face_set':patch,'components':cs,'boundary_nodes':bc,'boundary_degree_histogram':{str(k):sum(len(v)==k for v in vg.values()) for k in sorted(set(map(len,vg.values())))} })
  if radius<4:prev=patch|set().union(*(adj[f] for f in patch))
 return rings
def graph_stats(edges):
 graph=defaultdict(set)
 for u,v in edges:
  if u==v:continue
  graph[u].add(v);graph[v].add(u)
 unseen=set(graph);comps=[]
 while unseen:
  r=min(unseen);unseen.remove(r);q=deque([r]);c=[]
  while q:
   u=q.popleft();c.append(u)
   for v in graph[u]&unseen:unseen.remove(v);q.append(v)
  comps.append(c)
 return {'vertex_count':len(graph),'unique_edge_count':len({tuple(sorted(e)) for e in edges if e[0]!=e[1]}),'degree_histogram':{str(k):sum(len(v)==k for v in graph.values()) for k in sorted(set(map(len,graph.values())))},'component_vertex_counts':sorted([len(c) for c in comps],reverse=True)}
def main():
 t0=time.monotonic()
 with np.load(BASE,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 X=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64);parents=np.asarray(Z['refinement_face_parent_face_row'],np.int64)
 aud=mod('segment_audit',AUDIT);helpmod=mod('segment_helper',HELPER);exact=aud.exact_module()
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helpmod.read_bank_inputs(BANKDIR,E)
 bm=np.asarray(T[(ids[:,0]==BODY)&(ids[:,1]==BONE)],np.float64)*1000.;bids=np.arange(len(bm),dtype=np.int64)
 local=((X.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY])*1000.
 saved=json.loads(AUD159.read_text());bank=next(q for q in saved['banks'] if (int(q['body']),int(q['stable_bone_id']))==(BODY,BONE))
 raw=sorted({(int(a),int(b)) for a,b in bank['result']['triangle_pairs']})
 target_faces={f for f,b in raw}
 targets=[]
 all_edges=[]
 endpoint_records={}
 for fi,bi in raw:
  skin=rational_rows(local[F[fi]]);bone=rational_rows(bm[bi])
  pts=[]
  for p in exact.triangle_intersection_points(skin,bone):
   if p not in pts:pts.append(p)
  pts=sorted(pts)
  if len(pts)>=2:
   pairs=[(pts[0],pts[-1])] if len(pts)==2 else [(pts[i],pts[(i+1)%len(pts)]) for i in range(len(pts))]
  else:pairs=[]
  for u,v in pairs:
   uk=bkey(u);vk=bkey(v);all_edges.append((uk,vk))
  targets.append({'skin_face_row':fi,'source_parent_face_row':int(parents[fi]),'bone_triangle_row':bi,'intersection_vertex_count':len(pts),'intersection_vertices':[{'exact_local_mm':fpoint(p),'skin_barycentric':bary(p,skin),'bone_barycentric':bary(p,bone)} for p in pts],'segment_pairs':[[fpoint(u),fpoint(v)] for u,v in pairs],'segment_lengths_mm':[float(np.linalg.norm(np.asarray([float(x) for x in u])-np.asarray([float(x) for x in v]))) for u,v in pairs]})
 exact_self=[]
 Xmm=X.astype(np.float64)*1000.
 self_edges=[]
 for fa,fb in SELF_PAIRS:
  ta=rational_rows(Xmm[F[fa]]);tb=rational_rows(Xmm[F[fb]])
  pts=[]
  for p in exact.triangle_intersection_points(ta,tb):
   if p not in pts:pts.append(p)
  pts=sorted(pts)
  if len(pts)>=2:
   pairs=[(pts[0],pts[-1])] if len(pts)==2 else [(pts[i],pts[(i+1)%len(pts)]) for i in range(len(pts))]
  else:pairs=[]
  for u,v in pairs:self_edges.append((bkey(u),bkey(v)))
  exact_self.append({'skin_face_rows':[fa,fb],'intersection_vertex_count':len(pts),'intersection_vertices':[{'exact_world_mm':fpoint(p),'face_a_barycentric':bary(p,ta),'face_b_barycentric':bary(p,tb)} for p in pts],'segment_pairs':[[fpoint(u),fpoint(v)] for u,v in pairs]})
 inside=[]
 with AUD130.open(newline='') as fh:
  for r in csv.DictReader(fh):
   if (r['bank_body'],r['bank_stable_id'],r['classification'])==('109','114','inside'):inside.append(int(r['query_row']))
 seed=set(target_faces)
 inside_set=set(inside)
 for fi,t in enumerate(F):
  if any(int(n) in inside_set for n in t):seed.add(fi)
 for fa,fb in SELF_PAIRS:seed.update((fa,fb))
 rings=topology(F,seed,inside)
 prepared=exact.prepare_point_location(aud.records(bm,bids,exact))
 ringloc=[]
 for r in rings[3:5]:
  comp_out=[]
  for ci,cfaces in enumerate(r['components']):
   fs=set(cfaces)
   bedges=defaultdict(int)
   for fi in fs:
    t=F[fi]
    for j in range(3):bedges[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))]+=1
   bnodes=sorted({n for e,k in bedges.items() if k==1 for n in e})
   locs=Counter()
   for node in bnodes:
    q=exact.point_location_prepared(tuple(frac(v) for v in local[node]),prepared)['location'];locs[q]+=1
   comp_out.append({'component_index':ci,'face_count':len(cfaces),'boundary_nodes':bnodes,'boundary_location_counts':dict(locs),'boundary_all_outside_target_bone':locs.get('outside',0)==len(bnodes),'target_crossing_pairs':sum(1 for f,b in raw if f in fs),'target_crossing_face_count':len({f for f,b in raw if f in fs}),'inside_sample_nodes':sorted(inside_set & set(map(int,np.unique(F[np.asarray(cfaces,np.int64)]))))})
  ringloc.append({'radius':r['radius'],'total_faces':sum(len(c) for c in r['components']),'boundary_degree_histogram':r['boundary_degree_histogram'],'components':comp_out})
 def clean_ring(r):return {'radius':r['radius'],'face_count':len(r['face_set']),'node_count':int(len(np.unique(F[np.asarray(sorted(r['face_set']),np.int64)]))),'component_face_counts':[len(c) for c in r['components']],'boundary_degree_histogram':r['boundary_degree_histogram']}
 receipt={'schema':'numi.human.b109-114.exact-segment-and-boundary-preflight.v1','candidate_path':str(BASE),'candidate_sha256':sha(BASE),'target_bank':[BODY,BONE],'raw_target_pair_count':len(raw),'target_face_count':len(target_faces),'target_source_parent_pair_count':len({(int(parents[f]),b) for f,b in raw}),'target_pairs':targets,'target_segment_graph':graph_stats(all_edges),'exact_self_witnesses':exact_self,'self_segment_graph':graph_stats(self_edges),'inside_sample_nodes':inside,'inside_sample_count':len(inside),'seed_face_count':len(seed),'rings':[clean_ring(r) for r in rings],'boundary_location_checks':ringloc,'no_geometry_changed':True,'construction_only':True,'physical_steps':0,'elapsed_s':time.monotonic()-t0}
 out=D/'exact-segment-boundary-preflight.json';out.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report':str(out),'report_sha256':sha(out),'raw_target_pairs':len(raw),'target_faces':len(target_faces),'segment_graph':receipt['target_segment_graph'],'self_segment_graph':receipt['self_segment_graph'],'rings':receipt['rings'],'boundary_location_checks':ringloc},indent=2,allow_nan=False))
if __name__=='__main__':main()
