#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
from collections import defaultdict,Counter
import ast,hashlib,importlib.util,sys,json,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz';A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json';PATCH=S/'per-bank-workers-20261027/b43-16/attempt-001/complement-filled-patch-face-rows.npy';OUT=S/'per-bank-workers-20261027/b43-16/attempt-003/ring-clearance-check-rerun.json';AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py');HELPER_AUDIT=S/'per-bank-workers-20261027/b43-16/attempt-002/boundary_feasibility.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
# Reuse only tested local geometry predicates from the prior private receipt script; do not execute its audit.
tree=ast.parse(HELPER_AUDIT.read_text());wanted={'ff','dot','sub','add','mul','cross','in_tri3','orient2','on_seg2','seg_intersect2','in_tri2','exact_segment_triangle','point_seg_dist','point_tri_dist','seg_seg_dist','seg_tri_dist','winding'}
funcs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted]
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(HELPER_AUDIT),'exec'),globals())
started=time.monotonic();assert sha(C)=='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
P=Z['points_world_m'].astype(np.float64);F=Z['source_O_faces'].astype(np.int64);patch=set(map(int,np.load(PATCH,allow_pickle=False)))
a=mod('a43ring_audit',AUDIT);h=mod('a43ring_helper',a.BANKHELPER);bankdir=a.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';T,ids,gids,bodies,M,off,pos,bankkeys,pins,unused=h.read_bank_inputs(bankdir,a.E);mask=(ids[:,0]==43)&(ids[:,1]==16);BT=np.asarray(T[mask],float);BID=np.asarray(ids[mask,3],np.int64);local=(P-pos[43])@M[43].T-off[43];BTlo=BT.min(1);BThi=BT.max(1);BTfrac=[tuple(tuple(ff(x) for x in v) for v in t) for t in BT]
edgefaces=defaultdict(list);nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t:nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
def get_boundary(PATCH):
 out=[]
 for e,fs in edgefaces.items():
  ii=[f for f in fs if f in PATCH];oo=[f for f in fs if f not in PATCH]
  if ii and oo:out.append((e,ii,oo))
  elif ii and len(fs)==1:out.append((e,ii,[]))
 bg=defaultdict(list)
 for e,_,_ in out:bg[e[0]].append(e[1]);bg[e[1]].append(e[0])
 comps=[];unseen=set(bg)
 while unseen:
  st=min(unseen);unseen.remove(st);q=[st];cc=[]
  while q:
   v=q.pop();cc.append(v)
   for w in bg[v]:
    if w in unseen:unseen.remove(w);q.append(w)
  comps.append(cc)
 return out,bg,comps
allbanks=json.loads(A.read_text())['banks'];rows=[]
for level in range(3):
 bedges,bg,bcomps=get_boundary(patch);bn=set(bg);bset={e for e,_,_ in bedges};inbf={f for _,ii,_ in bedges for f in ii};outbf={f for _,_,oo in bedges for f in oo};adjf={f for n in bn for f in nodefaces[n]}
 edge_hits=[];edge_mins=[]
 for e,ii,oo in bedges:
  u,v=map(int,e);p0=tuple(ff(x) for x in local[u]);p1=tuple(ff(x) for x in local[v]);lo=np.minimum(local[u],local[v]);hi=np.maximum(local[u],local[v]);cand=np.flatnonzero(np.all(BThi>=lo,axis=1)&np.all(BTlo<=hi,axis=1));ex=[]
  for j in cand:
   q=exact_segment_triangle(p0,p1,BTfrac[j])
   if q and q['hit']:ex.append({'bone_face':int(BID[j]),'kind':q['kind'],'point_local_m':[str(x) for x in q['point']] if q['point'] else None})
  if ex:edge_hits.append({'edge':[u,v],'inside_faces':ii,'outside_faces':oo,'hits':ex})
  ds=[seg_tri_dist(local[u],local[v],tr) for tr in BT];j=int(np.argmin(ds));edge_mins.append({'edge':[u,v],'inside_faces':ii,'outside_faces':oo,'nearest_bone_face':int(BID[j]),'distance_mm':float(ds[j]*1000)})
 node_mins=[];inside_nodes=[]
 for n in sorted(bn):
  w=winding(local[n],BT);ds=[point_tri_dist(local[n],tr) for tr in BT];j=int(np.argmin(ds));rec={'node':int(n),'inside':bool(abs(w)>.5),'winding':w,'nearest_bone_face':int(BID[j]),'distance_mm':float(ds[j]*1000)};node_mins.append(rec)
  if rec['inside']:inside_nodes.append(rec)
 xhits=[]
 for br in allbanks:
  if not br.get('result'):continue
  key=[int(br['body']),int(br['stable_bone_id'])]
  for sf,bf in br['result']['triangle_pairs']:
   sf=int(sf);bf=int(bf);fv=set(map(int,F[sf]));sn=sorted(fv&bn);se=[];t=F[sf]
   for a0,b0 in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
    ee=tuple(sorted((int(a0),int(b0))))
    if ee in bset:se.append(list(ee))
   if sf in adjf or sn or se:
    rel=[]
    if sf in inbf:rel.append('inside_boundary_face')
    if sf in outbf:rel.append('outside_boundary_adjacent_face')
    if sf in adjf:rel.append('boundary_node_incident_face')
    if se:rel.append('shares_boundary_edge')
    xhits.append({'bank':key,'skin_face':sf,'bone_face':bf,'relation':rel,'shared_nodes':sn,'shared_edges':se})
 row={'ring_level':level,'patch_faces':len(patch),'boundary_edges':len(bedges),'boundary_nodes':len(bn),'boundary_components':len(bcomps),'boundary_component_node_counts':sorted(map(len,bcomps)),'boundary_degree_histogram':dict(Counter(len(bg[n]) for n in bg)),'simple_single_loop':len(bcomps)==1 and all(len(bg[n])==2 for n in bg),'boundary_inside_node_count':len(inside_nodes),'minimum_boundary_node_distance_mm':min(node_mins,key=lambda x:x['distance_mm'])['distance_mm'],'minimum_boundary_node':min(node_mins,key=lambda x:x['distance_mm']),'exact_target_bone_boundary_edge_hits':len(edge_hits),'target_bone_edge_hit_witnesses':edge_hits,'minimum_boundary_edge_distance_mm':min(edge_mins,key=lambda x:x['distance_mm'])['distance_mm'],'minimum_boundary_edge':min(edge_mins,key=lambda x:x['distance_mm']),'stored_crossing_touch_count':len(xhits),'stored_crossing_touch_by_bank':{f'{b[0]}/{b[1]}':sum(x['bank']==b for x in xhits) for b in sorted({tuple(x['bank']) for x in xhits})},'stored_crossing_touch_witnesses':xhits}
 rows.append(row)
 if level<2:
  new_faces={f for _,_,outside in bedges for f in outside}-patch;row['added_faces_for_next_ring']=len(new_faces);patch.update(new_faces)
report={'schema':'numi.human.bank43_16.ring_clearance_screen.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_sha256':sha(A),'seed_patch_face_rows_sha256':sha(PATCH),'target_bank':[43,16],'ring_expansion':'1 and 2 contiguous edge-adjacent face rings; no coordinates moved','rings':rows,'elapsed_seconds':time.monotonic()-started,'limitations':['Stored neighbor crossings come from audit-159; no new all-bank audit. Exact boundary-segment intersection is only against target bone 43/16. Float64 distances are diagnostic; no threshold imposed.']}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'rings':[{k:v for k,v in r.items() if k not in ('stored_crossing_touch_witnesses','target_bone_edge_hit_witnesses')} for r in rows],'elapsed_seconds':report['elapsed_seconds']},indent=2,allow_nan=False))
