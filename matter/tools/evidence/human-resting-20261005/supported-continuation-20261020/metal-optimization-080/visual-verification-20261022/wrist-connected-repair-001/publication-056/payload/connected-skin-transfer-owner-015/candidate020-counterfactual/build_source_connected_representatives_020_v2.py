#!/usr/bin/env python3
import hashlib,json,pathlib,struct
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
D=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/E/connected-skin-transfer-owner-015")
O=D/"candidate020-counterfactual"; inp=O/"actual-013-to-294-020-input.bin"; diag=O/"owner-diagnostic-precise.tsv"
outbin=O/"representative-source-graph-020-v2.bin"; reportpath=O/"representative-source-graph-020-v2.json"
if outbin.exists() or reportpath.exists(): raise SystemExit("refusing overwrite")
b=inp.read_bytes(); ns,nf,no,nof=struct.unpack_from("<4Q",b,0); off=32
def arr(dt,n):
 global off
 z=np.frombuffer(b,dtype=dt,count=n,offset=off).copy(); off+=np.dtype(dt).itemsize*n; return z
x=arr("<f4",ns*3).reshape(ns,3).astype(float); faces=arr("<u4",nf*3).reshape(nf,3).astype(int); w=arr("<f4",ns*86).reshape(ns,86); bind=arr("<u4",86).astype(int); outer=arr("<f4",no*3).reshape(no,3).astype(float); ofaces=arr("<u4",nof*3).reshape(nof,3).astype(int)
if off!=len(b): raise SystemExit("input trailing bytes")
donor=bind[np.argmax(w,axis=1)]
miss=[]; missdonor=[]; oDon=np.full(no,-1,dtype=int); oFace=np.full(no,-1,dtype=int); oBary=np.zeros((no,3),dtype=float)
for line in diag.read_text().splitlines():
 q=line.split("\t")
 if q[0]=="F": miss.append(int(q[1])); missdonor.append(int(q[2]))
 elif q[0]=="O": oDon[int(q[1])]=int(q[2]); oFace[int(q[1])]=int(q[3]); oBary[int(q[1])]=np.asarray(q[4:7],dtype=float)
miss=np.asarray(miss,dtype=int); missdonor=np.asarray(missdonor,dtype=int)
if len(miss)!=12 or np.any(missdonor!=donor[miss]): raise SystemExit("owner fallback list mismatch")
edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])); edges.sort(axis=1); edges=np.unique(edges,axis=0); lens=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)
rr=np.r_[edges[:,0],edges[:,1]]; cc=np.r_[edges[:,1],edges[:,0]]; dd=np.r_[lens,lens]; graph=coo_matrix((dd,(rr,cc)),shape=(ns,ns)).tocsr(); graph.sum_duplicates(); ncomp,labels=connected_components(graph,directed=False)
dist,pred=dijkstra(graph,indices=miss,directed=False,return_predecessors=True)
xc=float(np.median(x[:,0])); rows=[]
for k,src in enumerate(miss):
 db=int(missdonor[k]); candidates=np.flatnonzero(oFace>=0)
 if not len(candidates): raise SystemExit(f"no physical O rows for donor {db} target {src}")
 sf=oFace[candidates]
 if np.any((sf<0)|(sf>=nf)): raise SystemExit("O embedding source face out of range")
 af=faces[sf]; connector=np.linalg.norm(outer[candidates,None,:]-x[af],axis=2); costs=dist[k,af]+connector
 corner=np.argmin(costs,axis=1); score=costs[np.arange(len(candidates)),corner]
 order=np.argsort(score,kind="stable"); chosen=None
 for j in order:
  oi=int(candidates[j]); anchor=int(af[j,corner[j]]); cost=float(score[j])
  back=[anchor]; cur=anchor
  while cur!=int(src):
   cur=int(pred[k,cur])
   if cur<0 or len(back)>ns: raise SystemExit("broken source graph predecessor")
   back.append(cur)
  path=list(reversed(back)); pathx=x[path,0]
  if db==102 and (pathx.min()<=xc or outer[oi,0]<=xc): continue
  if labels[src]!=labels[anchor]: continue
  chosen=(oi,int(sf[j]),anchor,path,cost); break
 if chosen is None: raise SystemExit(f"no same-donor/source-component representative for {src} donor {db}")
 oi,faceRow,anchor,path,cost=chosen; direct=float(np.linalg.norm(x[src]-outer[oi])); euclid=float(np.linalg.norm(x[src]-x[anchor])+np.linalg.norm(x[anchor]-outer[oi]))
 rows.append({"source_vertex":int(src),"dominant_donor_body_index":db,"representative_o_local":oi,"representative_o_donor_body_index":int(oDon[oi]),"anchor_source_face_row":faceRow,"anchor_barycentric":oBary[oi].tolist(),"anchor_vertex":anchor,"source_graph_path_vertex_ids":path,"source_graph_plus_anchor_connector_m":cost,"direct_source_to_representative_m":direct,"path_euclidean_upper_m":euclid,"graph_path_vertex_count":len(path),"source_component":int(labels[src]),"anchor_component":int(labels[anchor]),"x_side_filter":"positive path/representative required" if db==102 else "side-neutral donor"})
with outbin.open("xb") as f:
 f.write(struct.pack("<Q",len(rows)))
 for r in rows: f.write(struct.pack("<II",r["source_vertex"],r["representative_o_local"]))
report={"schema":"numi.human.connected-skin-transfer-015.source-connected-representatives.candidate020.v1","counterfactual_only":True,"input_sha256":hashlib.sha256(b).hexdigest(),"owner_diagnostic_sha256":hashlib.sha256(diag.read_bytes()).hexdigest(),"registered_surface_graph":{"vertex_count":ns,"face_count":nf,"unique_edges":len(edges),"component_count":int(ncomp),"metric":"Euclidean edge lengths on exact source294 face graph using identity-004 current 020 replayed Float32 coordinates","source_x_median_m":xc},"selection":{"fallback_count":len(rows),"fallback_donor_counts":{str(int(d)):int(np.sum(missdonor==d)) for d in np.unique(missdonor)},"rule":"preserve target dominant donor ID; anchor through current owner-selected O-to-source face; require same source-face graph component; choose deterministic minimum graph distance plus face-corner connector; existing owner searches the representative one-ring while retaining the target donor, and imposes the representative-distance and projection caps; no scene-wide remap","configured_owner_projection_cap_m":0.05,"graph_distance_cap":"none; reported as locality diagnostic only"},"rows":rows,"physical_or_native_admission":False}
reportpath.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"representative_map":str(outbin),"map_sha256":hashlib.sha256(outbin.read_bytes()).hexdigest(),"report":str(reportpath),"report_sha256":hashlib.sha256(reportpath.read_bytes()).hexdigest(),"selection":report["selection"],"graph":report["registered_surface_graph"],"rows":[{k:v for k,v in r.items() if k!="source_graph_path_vertex_ids"} for r in rows]},indent=2))
