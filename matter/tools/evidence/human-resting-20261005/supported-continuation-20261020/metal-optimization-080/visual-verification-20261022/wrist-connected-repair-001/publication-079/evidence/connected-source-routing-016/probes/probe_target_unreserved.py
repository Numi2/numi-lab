#!/usr/bin/env python3
import hashlib, heapq, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE=Path('/Users/n/Work/numi-human-connected-surface-069')
RUN=BASE/'connected-source-patch-cut-123/run-016-best-reroute-expanded-cut'
DETAIL=BASE/'publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz'
OUT=BASE/'connected-source-patch-cut-123/route-probe-after-016'
REPORT=OUT/'target-unreserved-path-traces.json'
EXPECTED={
 'cut_report':'',
 'cut_npz':'21e2b65d85ba9bac176aaa670b916bc13189975a295c9ab328d1e835bb2e0b78',
 'detail':'21f1869b418a2eca1e5a0522059c8f9cc6f8afe1a270ccc7c7d99c9c19fdd2a4'
}
TARGET=(0,0)
START_OFFSETS=(0,7,14,21)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def edge(a,b):
 a=int(a); b=int(b)
 return (a,b) if a<b else (b,a)

def dijkstra(adj,start,target,forbid_nodes,forbid_edges,max_pops=250000):
 dist={start:0.0}; prev={}; heap=[(0.0,int(start))]; done=set(); pops=0
 while heap:
  d,u=heapq.heappop(heap)
  if u in done: continue
  done.add(u); pops+=1
  if pops>max_pops: return None,None,pops
  if u==target: break
  for v,wgt in adj[u].items():
   if v in forbid_nodes and v!=target: continue
   if v==start or v in done: continue
   if edge(u,v) in forbid_edges: continue
   nd=d+wgt; old=dist.get(v)
   if old is None or nd<old-1e-15:
    dist[v]=nd; prev[v]=u; heapq.heappush(heap,(nd,v))
   elif abs(nd-old)<=1e-15 and u<prev.get(v,10**18):
    prev[v]=u
 if target not in dist: return None,None,pops
 path=[target]
 while path[-1]!=start:
  if path[-1] not in prev: return None,None,pops
  path.append(prev[path[-1]])
 path.reverse()
 return path,float(dist[target]),pops

def main():
 t0=time.time(); OUT.mkdir(parents=True,exist_ok=True)
 report_path=RUN/'source-patch-cut-report.json'; cut_path=RUN/'source-patch-cut.npz'
 hashes={'cut_report':sha(report_path),'cut_npz':sha(cut_path),'detail':sha(DETAIL)}
 if hashes['cut_npz']!=EXPECTED['cut_npz'] or hashes['detail']!=EXPECTED['detail']:
  raise RuntimeError(f'input pin mismatch {hashes}')
 report=json.loads(report_path.read_text())
 arr=np.load(cut_path,allow_pickle=False); detail=np.load(DETAIL,allow_pickle=False)
 anchors_record=report['patch_loops'][0]
 if (anchors_record['component'],anchors_record['loop'])!=TARGET:
  raise RuntimeError('target loop identity changed')
 anchors=[int(x) for x in anchors_record['anchor_nodes']]
 coarse_loop=[int(x) for x in anchors_record['coarse_loop']]
 if len(anchors)!=27 or len(coarse_loop)!=27: raise RuntimeError('expected exact 27-anchor loop')
 faces=np.asarray(detail['source_surface_faces_detail_rows'],dtype=np.int64)
 points=np.asarray(detail['detail_baseline_world_m'],dtype=np.float64)
 stable=np.asarray(detail['detail_source_vertex_id'],dtype=np.int64)
 if faces.shape!=(108987,3) or points.shape!=(61950,3) or stable.shape!=(61950,):
  raise RuntimeError('pinned detailed source topology schema changed')
 if len(set(map(int,stable)))!=len(stable): raise RuntimeError('duplicate detail source IDs')
 if np.any(np.asarray(anchors)<0) or np.any(np.asarray(anchors)>=len(points)): raise RuntimeError('anchor row out of range')
 adj=[{} for _ in range(len(points))]
 for tri in faces:
  a,b,c=map(int,tri)
  for u,v in ((a,b),(b,c),(c,a)):
   d=float(np.linalg.norm(points[u]-points[v]))
   if d<=0 or not math.isfinite(d): raise RuntimeError('invalid source surface edge length')
   old=adj[u].get(v)
   if old is None or d<old: adj[u][v]=d; adj[v][u]=d
 all_anchors=set()
 for rec in report['patch_loops']:
  all_anchors.update(int(x) for x in rec.get('anchor_nodes',[]))
 route_indptr=np.asarray(arr['route_path_indptr'],dtype=np.int64)
 route_nodes=np.asarray(arr['route_path_nodes'],dtype=np.int64)
 route_edges=set(); reserved_nodes=set()
 for j in range(len(route_indptr)-1):
  path=route_nodes[route_indptr[j]:route_indptr[j+1]].tolist()
  if len(path)<2: raise RuntimeError('stored successful route has no edge')
  reserved_nodes.update(path[1:-1])
  route_edges.update(edge(path[k],path[k+1]) for k in range(len(path)-1))
 # Keep all other successful loops fixed; reroute only original seed-1 component-0 loop 0.
 if len(route_edges)!=int(np.asarray(arr['routed_cut_edges']).shape[0]):
  raise RuntimeError('stored route-edge ancestry does not match cut edge set')
 attempts=[]; best=None
 n=len(anchors)
 for rev in (False,True):
  for start in START_OFFSETS:
   order=[(start+k)%n for k in range(n)]
   if rev: order=list(reversed(order))
   used_nodes=set(); used_edges=set(); staged={}; failures=[]
   for si in order:
    a=anchors[si]; b=anchors[(si+1)%n]
    forbidden=(all_anchors-{a,b})|used_nodes
    path,length,pops=dijkstra(adj,a,b,forbidden,used_edges)
    if path is None or len(path)<2:
     free,free_len,free_pops=dijkstra(adj,a,b,all_anchors-{a,b},set())
     fail={'segment':int(si),'coarse_edge':[coarse_loop[si],coarse_loop[(si+1)%n]],
           'start_row':a,'end_row':b,'start_source_id':int(stable[a]),'end_source_id':int(stable[b]),
           'reason':'blocked_by_reserved_routes' if free is not None else 'no_anchor_safe_surface_path',
           'free_shortest_length_m':free_len,'attempt_pops':pops,'free_pops':free_pops}
     failures.append(fail); break
    internal=set(path[1:-1])
    if internal & all_anchors:
     failures.append({'segment':si,'reason':'nonterminal_anchor_interior','rows':sorted(internal&all_anchors)}); break
    pedges={edge(path[k],path[k+1]) for k in range(len(path)-1)}
    if internal & used_nodes or pedges & used_edges:
     failures.append({'segment':si,'reason':'reserved_vertex_or_edge_reuse','nodes':sorted(internal&used_nodes),'edges':[list(e) for e in sorted(pedges&used_edges)]}); break
    used_nodes|=internal; used_edges|=pedges
    staged[si]={'path_rows':path,'length_m':length,'heap_pops':pops}
   record={'start_offset':start,'reverse_order':rev,'segments_routed':len(staged),
           'failed':failures[0] if failures else None}
   attempts.append(record)
   score=len(staged)
   if best is None or score>best['segments_routed']:
    best={'segments_routed':score,'staged':staged,'attempt':record,'failures':failures}
   if not failures:
    ring=[]
    for si in range(n):
     ring.extend(staged[si]['path_rows'][:-1])
    if len(ring)!=len(set(ring)):
     record['failed']={'reason':'ring_repeats_vertex','repeated_rows':sorted(x for x in set(ring) if ring.count(x)>1)}
     continue
    record['status']='simple_cycle'
    output_paths=[]
    for si in range(n):
     p=staged[si]['path_rows']
     output_paths.append({'segment':si,'coarse_edge':[coarse_loop[si],coarse_loop[(si+1)%n]],
       'detail_rows':p,'detail_source_vertex_ids':[int(stable[x]) for x in p],
       'length_m':staged[si]['length_m'],'heap_pops':staged[si]['heap_pops']})
    result={'schema':'source_fine_graph_reroute_probe_v1','status':'SIMPLE_CYCLE_FOUND',
      'target':{'component':0,'loop':0,'anchor_count':n,'boundary_edges':n},
      'attempt':record,'simple_ring_detail_rows':ring,
      'simple_ring_detail_source_vertex_ids':[int(stable[x]) for x in ring],
      'routed_segment_paths':output_paths,'reserved_other_loop_edges':len(route_edges),
      'reserved_other_loop_internal_nodes':len(reserved_nodes),'input_sha256':hashes,
      'scope':'Graph path proposal only; no geometric crossing, source-patch extraction, or physical admission gate.',
      'elapsed_seconds':time.time()-t0}
    REPORT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'attempt':record,'report':str(REPORT),'segments':n,'path_lengths_mm':[1000*x['length_m'] for x in output_paths]},indent=2))
    return
 result={'schema':'source_fine_graph_reroute_probe_v1','status':'NO_SIMPLE_CYCLE_WITH_OTHER_LOOPS_UNRESERVED',
   'target':{'component':0,'loop':0,'anchor_count':n,'boundary_edges':n},
   'attempts':attempts,'best_partial_attempt':{'segments_routed':best['segments_routed'],'attempt':best['attempt']},
   'best_partial_paths':[{'segment':int(si),'coarse_edge':[int(coarse_loop[si]),int(coarse_loop[(si+1)%n])],
      'detail_rows':st['path_rows'],'detail_source_vertex_ids':[int(stable[x]) for x in st['path_rows']],
      'length_m':float(st['length_m'])} for si,st in sorted(best['staged'].items())],
   'reserved_other_loop_edges':len(route_edges),'reserved_other_loop_internal_nodes':len(reserved_nodes),
   'input_sha256':hashes,'scope':'Bounded graph-only search; no geometry changed and no proof of global infeasibility.',
   'elapsed_seconds':time.time()-t0}
 REPORT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':result['status'],'report':str(REPORT),'attempts':attempts},indent=2))

if __name__=='__main__':main()
