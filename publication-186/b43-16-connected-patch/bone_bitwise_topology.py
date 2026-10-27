from pathlib import Path
import importlib.util,sys,json,hashlib
from collections import defaultdict,deque,Counter
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz';A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json';D=S/'per-bank-workers-20261027/b43-16/attempt-001/local-diagnostic.json';OUT=S/'per-bank-workers-20261027/b43-16/attempt-001/bone-bitwise-topology.json'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
def mod(name,p):s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=mod('audit_b43_topology',AUDIT);h=mod('helper_b43_topology',a.BANKHELPER);E=a.E;bankdir=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';T,ids,*_=h.read_bank_inputs(bankdir,E);mask=(ids[:,0]==43)&(ids[:,1]==16);BT=np.asarray(T[mask],np.float64);BID=np.asarray(ids[mask,3],np.int64)
# Exact-bit corner classes only; no coordinate bins. Export has no explicit bone-corner ID columns.
keys=[];vid={}
for tri in BT:
 row=[]
 for p in tri:
  key=tuple(map(int,np.asarray(p,dtype=np.float32).view(np.uint32)))
  if key not in vid:vid[key]=len(vid)
  row.append(vid[key])
 keys.append(row)
V=np.asarray(keys,np.int64);edges=defaultdict(list)
for i,t in enumerate(V):
 for j,(u,v) in enumerate(((t[0],t[1]),(t[1],t[2]),(t[2],t[0]))):edges[tuple(sorted((int(u),int(v))))].append((i,int(u),int(v)))
bound=[(e,len(x)) for e,x in edges.items() if len(x)==1];nonman=[(e,len(x)) for e,x in edges.items() if len(x)>2]
orientation_conflicts=[]
for e,x in edges.items():
 if len(x)==2:
  _,u0,v0=x[0];_,u1,v1=x[1]
  if (u0,v0)==(u1,v1):orientation_conflicts.append(e)
Adata=json.loads(A.read_text());b=next(x for x in Adata['banks'] if (x['body'],x['stable_bone_id'])==(43,16));cross_bones=sorted({int(p[1]) for p in b['result']['triangle_pairs']});row_by_face={int(fid):i for i,fid in enumerate(BID)}
bi=[row_by_face[f] for f in cross_bones];unseen=set(bi);comps=[]
while unseen:
 st=min(unseen);unseen.remove(st);q=[st];comp=[]
 while q:
  i=q.pop();comp.append(int(BID[i]))
  for u,v in ((V[i,0],V[i,1]),(V[i,1],V[i,2]),(V[i,2],V[i,0])):
   for j,_,_ in edges[tuple(sorted((int(u),int(v))))]:
    if j in unseen:unseen.remove(j);q.append(j)
 comps.append(sorted(comp))
R={'schema':'numi.human.bank43_16.bone_triangle_bitwise_adjacency.v1','bank':[43,16],'triangle_count':len(BT),'corner_count':int(V.size),'unique_exact_bitwise_corner_classes':len(vid),'edge_count':len(edges),'edge_valence_histogram':dict(Counter(len(x) for x in edges.values())),'boundary_edge_count':len(bound),'nonmanifold_edge_count':len(nonman),'orientation_conflicting_two_face_edges':len(orientation_conflicts),'crossing_bone_face_component_count':len(comps),'crossing_bone_face_components':comps,'vertex_identity_source':'Exact Float32 bit tuples from the pinned triangle export only; export has no explicit bone-corner IDs. This is an exact-value adjacency proxy, not a certified source topology ID map.','candidate_sha256':sha(C),'audit159_sha256':sha(A),'input_pins':b'','limitations':['No coordinate-bin merging. No full/all-bank/self audit or construction. A coincident disconnected bone vertex pair could be conflated by bitwise coordinate equality because explicit corner IDs are absent.']}
R['input_pins']=h.read_bank_inputs(bankdir,E)[8]
OUT.write_text(json.dumps(R,indent=2)+'\n');print(json.dumps(R,indent=2))
