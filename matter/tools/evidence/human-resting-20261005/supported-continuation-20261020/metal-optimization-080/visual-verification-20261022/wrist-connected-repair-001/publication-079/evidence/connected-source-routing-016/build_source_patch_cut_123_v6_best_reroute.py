#!/usr/bin/env python3
import csv, gzip, hashlib, heapq, json, math, copy
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay, QhullError

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069")
OUT = BASE / "connected-source-patch-cut-123"
RUN = OUT / "run-016-best-reroute-expanded-cut"
RUN.mkdir(parents=True, exist_ok=False)
PROP = BASE / "connected-boundary-refinement-114/run-002-source-normal-all-146/surface-proposal.npz"
COARSE = BASE / "publication-072/evidence/whole-reference-TetGen-084/whole-skin-fat-reference-084-candidate.npz"
DETAIL = BASE / "publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
COARSE_CSV = BASE / "publication-071/evidence/exact-transfer-075/run-001/source-preflight/accepted-coarse-reference.csv.gz"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def edge(a,b):
    a=int(a); b=int(b)
    return (a,b) if a<b else (b,a)

prop=np.load(PROP,allow_pickle=False); coarse=np.load(COARSE,allow_pickle=False); detail=np.load(DETAIL,allow_pickle=False)
F=np.asarray(coarse["source_O_faces"],dtype=np.int64)
E=np.asarray(prop["selected_outer_edges"],dtype=np.int64)
xyz=np.asarray(detail["detail_baseline_world_m"],dtype=np.float64)
fine_faces=np.asarray(detail["source_surface_faces_detail_rows"],dtype=np.int64)
stable_ids=np.asarray(detail["detail_source_vertex_id"],dtype=np.int64)
id_to_row={int(s):i for i,s in enumerate(stable_ids)}
if len(id_to_row)!=len(stable_ids): raise RuntimeError("duplicate stable detailed vertex IDs")

rows=[None]*2411
with gzip.open(COARSE_CSV,"rt",newline="") as f:
    for rec in csv.DictReader(f):
        i=int(rec["coarse_row"])
        if 0<=i<len(rows): rows[i]=rec
if any(r is None for r in rows): raise RuntimeError("coarse reference rows are incomplete")

w=np.asarray(detail["transfer_weights"],dtype=np.float64)
coarse_ids=np.asarray(detail["coarse_vertex_ids"],dtype=np.int64)
active=np.asarray(detail["active"],dtype=bool)
detail_src=np.asarray(detail["detail_source_vertex_id"],dtype=np.int64)
onehot=active & (np.count_nonzero(w,axis=1)==1) & (np.max(w,axis=1)==1.0)
anchor_candidates=defaultdict(list)
for r in np.flatnonzero(onehot):
    j=int(np.argmax(w[r])); cr=int(coarse_ids[r,j]); sr=int(detail_src[r])
    if sr not in id_to_row: raise RuntimeError(f"direct anchor stable ID {sr} has no detail row")
    anchor_candidates[cr].append(int(id_to_row[sr]))
# Geometry anchors come from accepted registered source mapping, not transfer one-hot support.
direct_anchor={}
direct_anchor_residual={}
for cr,rec in enumerate(rows):
    sid=int(rec["source_direct_id"])
    if sid < 0 or sid not in id_to_row: continue
    dr=int(id_to_row[sid]); direct_anchor[cr]=dr
    cp=np.array([float(rec[f"world_{a}_m"]) for a in "xyz"],dtype=np.float64)
    direct_anchor_residual[cr]=float(np.linalg.norm(xyz[dr]-cp))
if len(set(direct_anchor.values())) != len(direct_anchor):
    raise RuntimeError("accepted direct anchors are not injective")
anchor_mapping_audit={
    "accepted_csv_direct_anchor_count":len(direct_anchor),
    "onehot_candidate_count":int(sum(len(v) for v in anchor_candidates.values())),
    "onehot_candidate_coarse_rows":len(anchor_candidates),
    "onehot_duplicate_coarse_rows":int(sum(len(v)>1 for v in anchor_candidates.values())),
    "direct_anchor_max_residual_m":max(direct_anchor_residual.values(),default=0.0),
    "direct_anchor_p99_residual_m":float(np.quantile(list(direct_anchor_residual.values()),.99)) if direct_anchor_residual else 0.0,
    "direct_anchor_wrong_min_onehot_rows":[],
}
for cr,candidates in anchor_candidates.items():
    if cr in direct_anchor and direct_anchor[cr] != min(candidates):
        cp=np.array([float(rows[cr][f"world_{a}_m"]) for a in "xyz"],dtype=np.float64)
        anchor_mapping_audit["direct_anchor_wrong_min_onehot_rows"].append({
            "coarse_row":int(cr),"csv_direct_detail_row":int(direct_anchor[cr]),
            "min_onehot_detail_row":int(min(candidates)),
            "csv_residual_m":direct_anchor_residual[int(cr)],
            "min_onehot_residual_m":float(np.linalg.norm(xyz[min(candidates)]-cp))
        })

edge_faces=defaultdict(list)
for fi,tri in enumerate(F):
    a,b,c=map(int,tri)
    for u,v in ((a,b),(b,c),(c,a)): edge_faces[edge(u,v)].append(fi)
face_adj=[set() for _ in range(len(F))]
nonmanifold_edges=[]
for e,uses in edge_faces.items():
    if len(uses)==2:
        face_adj[uses[0]].add(uses[1]); face_adj[uses[1]].add(uses[0])
    elif len(uses)>2: nonmanifold_edges.append([*e,len(uses)])
seed_faces=set(); missing_selected=[]
for e in E:
    ee=edge(*e); uses=edge_faces.get(ee,[])
    if not uses: missing_selected.append(list(ee))
    seed_faces.update(uses)
if missing_selected: raise RuntimeError(f"selected edges absent from compact source O faces: {missing_selected[:8]}")

seed_components=[]; unseen=set(seed_faces)
while unseen:
    start=min(unseen); unseen.remove(start); comp={start}; q=[start]
    while q:
        x=q.pop()
        for y in sorted(face_adj[x] & unseen):
            unseen.remove(y); comp.add(y); q.append(y)
    seed_components.append(comp)
seed_components.sort(key=lambda c:(min(c),len(c)))
# Target failed exact-self component 2 and three unroutable components 1, 7, 31.
# Expand each target by one coarse face ring, include any original seed component
# touched by that ring, then recompute connected components on the union.
original_seed_components=[set(c) for c in seed_components]
target_original_component_ids=[1,2,7,31]
if max(target_original_component_ids)>=len(original_seed_components):
    raise RuntimeError("target seed component index is outside frozen v3 component domain")
expanded=set()
for ci in target_original_component_ids:
    comp=original_seed_components[ci]
    expanded.update(comp)
    for fi in comp:
        expanded.update(face_adj[fi])
touched_original=[ci for ci,c in enumerate(original_seed_components) if c & expanded]
for ci in touched_original:
    expanded.update(original_seed_components[ci])
expanded_face_count_before_merge=len(expanded)
expanded_components=[]; unseen=set(expanded)
while unseen:
    start=min(unseen); unseen.remove(start); comp={start}; q=[start]
    while q:
        x=q.pop()
        for y in sorted(face_adj[x] & unseen):
            unseen.remove(y); comp.add(y); q.append(y)
    expanded_components.append(comp)
expanded_components.sort(key=lambda c:(min(c),len(c)))
expanded_component_origins=[]
for c in expanded_components:
    expanded_component_origins.append([ci for ci,old in enumerate(original_seed_components) if c & old])
seed_components=expanded_components
seed_faces=set(expanded)

def boundary_loops(comp):
    bedges=[]
    for fi in comp:
        tri=F[fi]
        for u,v in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            if sum(x in comp for x in edge_faces[edge(u,v)])==1: bedges.append(edge(u,v))
    graph=defaultdict(set)
    for u,v in bedges: graph[u].add(v); graph[v].add(u)
    bad={int(v):len(n) for v,n in graph.items() if len(n)!=2}
    if bad: return bedges,graph,[],bad
    loops=[]; left=set(bedges)
    while left:
        e0=min(left); start=e0[0]; prev=None; cur=start; loop=[start]
        while True:
            nbs=sorted(graph[cur]); nxt=nbs[0] if nbs[0]!=prev else nbs[1]; ek=edge(cur,nxt)
            if ek not in left and nxt!=start: return bedges,graph,loops,{"reused_edge":ek}
            left.discard(ek)
            if nxt==start: break
            loop.append(nxt); prev,cur=cur,nxt
            if len(loop)>len(graph)+1: return bedges,graph,loops,{"walk_limit":len(loop)}
        if len(loop)<3: return bedges,graph,loops,{"short_loop":loop}
        loops.append(loop)
    return bedges,graph,loops,{}

def oriented_boundary_loops(comp):
    directed=[]
    for fi in comp:
        tri=list(map(int,F[fi]))
        for i in range(3):
            u,v=tri[i],tri[(i+1)%3]
            if sum(x in comp for x in edge_faces[edge(u,v)])==1: directed.append((u,v))
    out=defaultdict(list); inc=defaultdict(list)
    for u,v in directed: out[u].append(v); inc[v].append(u)
    bad={int(v):{"out":len(out[v]),"in":len(inc[v])}
         for v in set(out)|set(inc) if len(out[v])!=1 or len(inc[v])!=1}
    if bad: return directed,[],bad
    loops=[]; unused=set(directed)
    while unused:
        first=min(unused); start=first[0]; cur=start; loop=[start]
        while True:
            pair=(cur,out[cur][0])
            if pair not in unused:
                if pair[1]==start: break
                return directed,loops,{"reused_directed_edge":pair}
            unused.remove(pair); nxt=pair[1]
            if nxt==start: break
            loop.append(nxt); cur=nxt
            if len(loop)>len(directed)+1: return directed,loops,{"walk_limit":len(loop)}
        if len(loop)<3: return directed,loops,{"short_loop":loop}
        loops.append(loop)
    return directed,loops,{}

def missing_boundary(comp):
    _,_,loops,bad=boundary_loops(comp)
    verts={v for loop in loops for v in loop}
    return sorted(verts-set(direct_anchor)),bad,len(loops)

expansion=[]
for ci,comp in enumerate(seed_components):
    cur=set(comp); stats=[]
    for ring in range(5):
        miss,bad,nloop=missing_boundary(cur)
        stats.append({"ring":ring,"faces":len(cur),"boundary_missing_direct":miss,
                      "boundary_missing_count":len(miss),"loop_count":nloop,"boundary_cycle_error":bad})
        if ring<4:
            neigh=set()
            for f in cur: neigh.update(face_adj[f])
            cur |= neigh
    expansion.append(stats)

virtual=[]; coarse_anchor={}
for ci,comp in enumerate(seed_components):
    _,loops,bad=oriented_boundary_loops(comp)
    if bad: continue
    for loop in loops:
        for cr in loop:
            cr=int(cr)
            if cr in coarse_anchor: continue
            if cr in direct_anchor:
                coarse_anchor[cr]=int(direct_anchor[cr]); continue
            rec=rows[cr]
            sfids=[int(rec[f"source_face_v{k}"]) for k in range(3)]
            bary=np.array([float(rec[f"source_bary{k}"]) for k in range(3)],dtype=np.float64)
            if any(s not in id_to_row for s in sfids): raise RuntimeError(f"virtual anchor coarse row {cr}: source vertex missing")
            if not np.isfinite(bary).all() or np.min(bary)<-1e-12 or abs(float(bary.sum())-1)>1e-10:
                raise RuntimeError(f"virtual anchor coarse row {cr}: invalid barycentrics {bary}")
            fr=np.array([id_to_row[s] for s in sfids],dtype=np.int64); pt=bary @ xyz[fr]
            k=int(np.argmax(bary))
            if bary[k]==1.0 and np.count_nonzero(bary)==1:
                coarse_anchor[cr]=int(fr[k])
            else:
                vi=len(xyz)+len(virtual)
                virtual.append({"node":vi,"coarse_row":cr,"source_vertex_ids":sfids,
                                "source_face_detail_rows":fr.tolist(),"barycentric":bary.tolist(),
                                "point_world_m":pt.tolist(),"coarse_world_m":[float(rec["world_x_m"]),float(rec["world_y_m"]),float(rec["world_z_m"])],
                                "coarse_to_virtual_residual_m":float(np.linalg.norm(pt-np.array([float(rec["world_x_m"]),float(rec["world_y_m"]),float(rec["world_z_m"])])))})
                coarse_anchor[cr]=vi
if len(virtual)>10000: raise RuntimeError(f"unreasonable virtual-anchor count {len(virtual)}")

N=len(xyz)+len(virtual)
coords=np.vstack([xyz,np.asarray([v["point_world_m"] for v in virtual],dtype=np.float64)]) if virtual else xyz.copy()
adj=[{} for _ in range(N)]; fine_edge_faces=defaultdict(list)
for fi,tri in enumerate(fine_faces):
    a,b,c=map(int,tri)
    for u,v in ((a,b),(b,c),(c,a)):
        fine_edge_faces[edge(u,v)].append(fi)
        d=float(np.linalg.norm(xyz[u]-xyz[v]))
        if d<=0 or not math.isfinite(d): raise RuntimeError(f"invalid fine edge {edge(u,v)}")
        adj[u][v]=d; adj[v][u]=d
for v in virtual:
    node=int(v["node"]); fr=v["source_face_detail_rows"]; bary=np.asarray(v["barycentric"],dtype=np.float64)
    active_idx=np.flatnonzero(bary>1e-12)
    if len(active_idx)==2:
        u=int(fr[active_idx[0]]); q=int(fr[active_idx[1]])
        if q in adj[u]: del adj[u][q]; del adj[q][u]
        for k in active_idx:
            r=int(fr[k]); d=float(np.linalg.norm(coords[node]-xyz[r])); adj[node][r]=d; adj[r][node]=d
    elif len(active_idx)==3:
        for r in fr:
            r=int(r); d=float(np.linalg.norm(coords[node]-xyz[r])); adj[node][r]=d; adj[r][node]=d
    else: raise RuntimeError(f"unsupported virtual anchor barycentric support {bary}")

loop_records=[]; all_ring_anchors=set()
for ci,comp in enumerate(seed_components):
    directed,loops,bad=oriented_boundary_loops(comp)
    if bad:
        loop_records.append({"component":ci,"coarse_faces":sorted(map(int,comp)),
                             "status":"boundary_not_manifold_cycle","error":bad})
        continue
    for li,loop in enumerate(loops):
        aa=[int(coarse_anchor[v]) for v in loop]
        if len(set(aa))!=len(aa):
            loop_records.append({"component":ci,"loop":li,"status":"anchor_collision",
                                 "coarse_faces":sorted(map(int,comp)),
                                 "coarse_loop":list(map(int,loop)),"anchor_nodes":aa})
            continue
        all_ring_anchors.update(aa)
        loop_records.append({"component":ci,"loop":li,"coarse_faces":sorted(map(int,comp)),
                             "coarse_loop":list(map(int,loop)),"anchor_nodes":aa,
                             "boundary_edge_count":len(loop),"status":"pending"})
def dijkstra_path(start,target,forbid_nodes,forbid_edges):
    dist={start:0.0}; prev={}; heap=[(0.0,int(start))]; done=set()
    while heap:
        d,u=heapq.heappop(heap)
        if u in done: continue
        done.add(u)
        if u==target: break
        for v,wgt in adj[u].items():
            if v in forbid_nodes and v!=target: continue
            if v==start or v in done: continue
            if edge(u,v) in forbid_edges: continue
            nd=d+wgt; old=dist.get(v)
            if old is None or nd<old-1e-15:
                dist[v]=nd; prev[v]=u; heapq.heappush(heap,(nd,v))
            elif abs(nd-old)<=1e-15 and u<prev.get(v,10**18): prev[v]=u
    if target not in dist: return None,None
    path=[target]
    while path[-1]!=start: path.append(prev[path[-1]])
    path.reverse()
    return path,float(dist[target])

# Deterministic bounded rip-up/reroute across all expanded target loops.
# Each attempt resets reservations and changes loop/edge ordering; paths remain
# vertex-disjoint except at their own endpoints and may not hit other anchors.
routeable=[r for r in loop_records if r["status"]=="pending"]
def loop_priority(rec):
    loop=rec["coarse_loop"]; perimeter=0.0
    for i,u in enumerate(loop):
        v=loop[(i+1)%len(loop)]
        pu=np.array([float(rows[u][f"world_{a}_m"]) for a in "xyz"])
        pv=np.array([float(rows[v][f"world_{a}_m"]) for a in "xyz"])
        perimeter+=float(np.linalg.norm(pv-pu))
    return (-perimeter,int(rec["component"]),int(rec["loop"]))
def alternating(items, key, reverse_first=True):
    vals=sorted(items,key=key,reverse=reverse_first)
    out=[]; lo=0; hi=len(vals)-1; take_lo=True
    while lo<=hi:
        if take_lo: out.append(vals[lo]);lo+=1
        else: out.append(vals[hi]);hi-=1
        take_lo=not take_lo
    return out
def edge_order(rec, mode, failed_hint=None):
    n=len(rec["coarse_loop"]); rows0=[]
    for si in range(n):
        s0=int(rec["anchor_nodes"][si]);t0=int(rec["anchor_nodes"][(si+1)%n])
        rows0.append((float(np.linalg.norm(coords[s0]-coords[t0])),si,s0,t0))
    if mode==0: rows0.sort(key=lambda q:(-q[0],q[1]))
    elif mode==1: rows0.sort(key=lambda q:(q[0],q[1]))
    elif mode==2: rows0=alternating(rows0,key=lambda q:(q[0],q[1]),reverse_first=True)
    else:
        rows0.sort(key=lambda q:(-q[0],q[1]))
        if failed_hint and (int(rec["component"]),int(rec["loop"]))==tuple(failed_hint[:2]):
            si=int(failed_hint[2]); rows0.sort(key=lambda q:(q[1]!=si,-q[0],q[1]))
    return rows0
loop_values=sorted(routeable,key=loop_priority)
attempt_specs=[
    (loop_values,0,"longest_loops_longest_edges"),
    (list(reversed(loop_values)),1,"shortest_loops_shortest_edges"),
    (alternating(loop_values,key=loop_priority,reverse_first=True),2,"alternating_loops_edges"),
]
last_failure=None
failed_first=sorted(routeable,key=lambda r:(0 if last_failure and (r["component"],r["loop"])==last_failure[:2] else 1,loop_priority(r)))
attempt_specs.append((failed_first,3,"failed_loop_and_edge_first"))
segment_records=[]; route_attempt_records=[]; used_internal=set(); used_edges=set()
best_route_records=None;best_segments=None;best_used_nodes=set();best_used_edges=set();best_routed_count=-1;best_attempt_number=None
for attempt_id,(ordered_loops,mode,label) in enumerate(attempt_specs):
    if label=="failed_loop_and_edge_first" and last_failure is not None:
        ordered_loops=sorted(routeable,key=lambda r:(0 if (int(r["component"]),int(r["loop"]))==tuple(last_failure[:2]) else 1,loop_priority(r)))
    for rec in routeable:
        rec.pop("path_failure",None); rec["status"]="pending"
    attempt_segments=[]; attempt_used_nodes=set(); attempt_used_edges=set(); failures=[]
    for rec in ordered_loops:
        aa=list(map(int,rec["anchor_nodes"])); n=len(aa)
        pending=edge_order(rec,mode,last_failure); local_internal=set();local_edges=set();staged={};fail=None
        for dist0,si,s0,t0 in pending:
            forbidden=(all_ring_anchors-{s0,t0}) | (attempt_used_nodes|local_internal)
            path,dist=dijkstra_path(s0,t0,forbidden,attempt_used_edges|local_edges)
            if path is None or len(path)<2:
                free_path,free_dist=dijkstra_path(s0,t0,all_ring_anchors-{s0,t0},set())
                blocked_nodes=[];blocked_edges=[]
                if free_path is not None:
                    blocked_nodes=sorted(set(free_path[1:-1])&(attempt_used_nodes|local_internal))
                    fp={edge(free_path[j],free_path[j+1]) for j in range(len(free_path)-1)}
                    blocked_edges=sorted(fp&(attempt_used_edges|local_edges))
                fail={"segment":int(si),"start_anchor":int(s0),"end_anchor":int(t0),
                      "reason":"blocked_by_reserved_paths" if free_path is not None else "no_anchor_safe_source_graph_path",
                      "free_shortest_length_m":free_dist,"blocking_reserved_nodes":blocked_nodes[:100],
                      "blocking_reserved_edges":[list(e) for e in blocked_edges[:100]]}
                last_failure=(int(rec["component"]),int(rec["loop"]),int(si)); break
            internal=set(path[1:-1]); badnodes=internal&all_ring_anchors
            if badnodes:
                fail={"segment":int(si),"reason":"nonterminal_ring_anchor_hit","nodes":sorted(badnodes)}
                last_failure=(int(rec["component"]),int(rec["loop"]),int(si)); break
            pedges={edge(path[j],path[j+1]) for j in range(len(path)-1)}
            if pedges&(attempt_used_edges|local_edges):
                fail={"segment":int(si),"reason":"path_edge_reuse","edges":[list(e) for e in sorted(pedges&(attempt_used_edges|local_edges))]}
                last_failure=(int(rec["component"]),int(rec["loop"]),int(si)); break
            if internal&local_internal:
                fail={"segment":int(si),"reason":"path_node_reuse","nodes":sorted(internal&local_internal)}
                last_failure=(int(rec["component"]),int(rec["loop"]),int(si)); break
            local_internal|=internal;local_edges|=pedges
            staged[si]={"component":int(rec["component"]),"loop":int(rec["loop"]),"segment":int(si),
                        "coarse_start":int(rec["coarse_loop"][si]),"coarse_end":int(rec["coarse_loop"][(si+1)%n]),
                        "start_anchor":int(s0),"end_anchor":int(t0),"path_nodes":path,
                        "path_edge_count":len(path)-1,"length_m":float(dist)}
        if fail is None:
            ring=[]
            for si in range(n): ring.extend(staged[si]["path_nodes"][:-1])
            if len(ring)!=len(set(ring)):
                fail={"reason":"ring_self_touch_or_intersection","repeated_nodes":sorted(v for v,c in Counter(ring).items() if c>1)}
            else:
                attempt_used_nodes|=local_internal;attempt_used_edges|=local_edges
                attempt_segments.extend(staged[i] for i in sorted(staged))
                rec["status"]="closed_disjoint_source_edge_cycle";rec["fine_ring_node_count"]=len(ring)
                rec["fine_ring_edge_count"]=len(local_edges)
                rec["fine_ring_length_m"]=float(sum(staged[i]["length_m"] for i in staged))
        if fail is not None:
            rec["status"]="path_failure";rec["path_failure"]=fail
            failures.append({"component":int(rec["component"]),"loop":int(rec["loop"]),"failure":fail})
    route_attempt_records.append({"attempt":attempt_id+1,"policy":label,
        "routed_loops":sum(r["status"]=="closed_disjoint_source_edge_cycle" for r in routeable),
        "total_loops":len(routeable),"failed_loops":failures})
    routed_now=sum(r["status"]=="closed_disjoint_source_edge_cycle" for r in routeable)
    if routed_now>best_routed_count:
        best_routed_count=routed_now;best_attempt_number=attempt_id+1
        best_route_records=copy.deepcopy(routeable);best_segments=copy.deepcopy(attempt_segments)
        best_used_nodes=set(attempt_used_nodes);best_used_edges=set(attempt_used_edges)
    segment_records=attempt_segments;used_internal=attempt_used_nodes;used_edges=attempt_used_edges
    if not failures: break
if best_route_records is not None:
    for rec,snapshot in zip(routeable,best_route_records):
        rec.clear();rec.update(snapshot)
    segment_records=best_segments;used_internal=best_used_nodes;used_edges=best_used_edges
# Build locally subdivided source faces around virtual anchors for dual-graph cut testing.
face_by_key={tuple(sorted(map(int,f))):i for i,f in enumerate(fine_faces)}
virtual_face_nodes=defaultdict(set)
for v in virtual:
    node=int(v["node"]); fr=np.asarray(v["source_face_detail_rows"],dtype=np.int64); bary=np.asarray(v["barycentric"],dtype=np.float64)
    key=tuple(sorted(map(int,fr)))
    if key not in face_by_key: raise RuntimeError(f"virtual anchor source face {key} absent from fine source topology")
    fi=face_by_key[key]; virtual_face_nodes[fi].add(node)
    zeros=np.flatnonzero(bary<=1e-12)
    if len(zeros)==1:
        en=[int(fr[k]) for k in range(3) if k!=int(zeros[0])]
        for afi in fine_edge_faces[edge(*en)]: virtual_face_nodes[int(afi)].add(node)

subfaces=[]; subface_parent=[]
for fi,tri in enumerate(fine_faces):
    vnodes=sorted(virtual_face_nodes.get(fi,set()))
    if not vnodes:
        subfaces.append(tuple(map(int,tri))); subface_parent.append(fi); continue
    a,b,c=xyz[np.asarray(tri,dtype=np.int64)]; u=b-a; ul=np.linalg.norm(u)
    if ul<=0: raise RuntimeError(f"degenerate fine source face {fi}")
    u=u/ul; nn=np.cross(b-a,c-a); nl=np.linalg.norm(nn)
    if nl<=0: raise RuntimeError(f"degenerate fine source face {fi}")
    nn=nn/nl; vv=np.cross(nn,u)
    local=np.asarray([a,b,c]+[coords[n] for n in vnodes],dtype=np.float64)
    xy=np.column_stack(((local-a)@u,(local-a)@vv))
    try: dt=Delaunay(xy)
    except QhullError as exc: raise RuntimeError(f"virtual-anchor subdivision failed on fine face {fi}: {exc}") from exc
    M=np.column_stack((b-a,c-a))
    for simplex in dt.simplices:
        tri_nodes=[int(tri[k]) if k<3 else int(vnodes[k-3]) for k in simplex]
        cent=coords[np.asarray(tri_nodes,dtype=np.int64)].mean(axis=0)
        uv,_,_,_=np.linalg.lstsq(M,cent-a,rcond=None)
        if uv[0]>=-1e-10 and uv[1]>=-1e-10 and uv.sum()<=1+1e-10:
            subfaces.append(tuple(tri_nodes)); subface_parent.append(fi)
subfaces=np.asarray(subfaces,dtype=np.int64); subface_parent=np.asarray(subface_parent,dtype=np.int32)
for fi in range(len(subfaces)):
    tri=subfaces[fi]; src=fine_faces[int(subface_parent[fi])]
    n0=np.cross(coords[tri[1]]-coords[tri[0]],coords[tri[2]]-coords[tri[0]])
    ns=np.cross(xyz[src[1]]-xyz[src[0]],xyz[src[2]]-xyz[src[0]])
    if float(np.dot(n0,ns)) < 0.0: subfaces[fi,[1,2]]=subfaces[fi,[2,1]]
subedge_faces=defaultdict(list); subarea=np.empty(len(subfaces),dtype=np.float64)
for fi,tri in enumerate(subfaces):
    a,b,c=coords[tri]; subarea[fi]=0.5*np.linalg.norm(np.cross(b-a,c-a))
    for u,v in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])): subedge_faces[edge(u,v)].append(fi)
bad_subedges={e:len(q) for e,q in subedge_faces.items() if len(q)>2}

def face_components(cut):
    seen=set(); comps=[]
    for start in range(len(subfaces)):
        if start in seen: continue
        q=[start]; seen.add(start); comp=[]
        while q:
            x=q.pop(); comp.append(x); tri=subfaces[x]
            for u,v in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                ee=edge(u,v)
                if ee in cut: continue
                for y in subedge_faces[ee]:
                    if y!=x and y not in seen: seen.add(y); q.append(y)
        comps.append(comp)
    return comps

for rec in loop_records:
    if rec["status"]!="closed_disjoint_source_edge_cycle": continue
    cut=set()
    for seg in segment_records:
        if seg["component"]==rec["component"] and seg["loop"]==rec["loop"]:
            pth=seg["path_nodes"]; cut.update(edge(pth[i],pth[i+1]) for i in range(len(pth)-1))
    missing_cut=sorted(e for e in cut if e not in subedge_faces)
    comps=face_components(cut)
    sizes=sorted((len(c) for c in comps),reverse=True); areas=sorted((float(subarea[c].sum()) for c in comps),reverse=True)
    rec["fine_cut_status"]="cut_edges_present" if not missing_cut else "cut_edges_missing_from_virtual_subdivision"
    rec["fine_cut_edge_count"]=len(cut); rec["fine_face_component_count_after_cut"]=len(comps)
    rec["fine_face_component_face_counts"]=sizes[:20]; rec["fine_face_component_areas_m2"]=areas[:20]
    rec["fine_cut_missing_edges"]=[list(e) for e in missing_cut[:20]]

# Check route segments that cross in the interior of a fine source face (virtual chords only can do this).
edge_face_membership=defaultdict(set)
for e,fs in fine_edge_faces.items(): edge_face_membership[e].update(map(int,fs))
for v in virtual:
    n=int(v["node"]); fr=list(map(int,v["source_face_detail_rows"])); bary=np.asarray(v["barycentric"])
    fi=face_by_key[tuple(sorted(fr))]
    for r in fr: edge_face_membership[edge(n,r)].add(fi)
    zeros=np.flatnonzero(bary<=1e-12)
    if len(zeros)==1:
        en=[fr[k] for k in range(3) if k not in zeros]
        for fidx in fine_edge_faces[edge(*en)]:
            for r in fr: edge_face_membership[edge(n,r)].add(int(fidx))
route_edge_owner={}
for si,seg in enumerate(segment_records):
    pth=seg["path_nodes"]
    for j in range(len(pth)-1): route_edge_owner[edge(pth[j],pth[j+1])]=(seg["component"],seg["loop"],seg["segment"],si)
face_route_edges=defaultdict(list)
for ee,owner in route_edge_owner.items():
    for fi in edge_face_membership.get(ee,()): face_route_edges[int(fi)].append((ee,owner))
virtual_same_face_crossings=[]
for fi,entries in face_route_edges.items():
    if len(entries)<2: continue
    tri=fine_faces[fi]; a,b,c=xyz[tri]; u=b-a; u=u/np.linalg.norm(u); nn=np.cross(b-a,c-a); nn=nn/np.linalg.norm(nn); vv=np.cross(nn,u)
    def xy_for(n):
        p=coords[int(n)]-a
        return np.array([p@u,p@vv])
    def orient(p,q,r): return float((q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]))
    def proper(e1,e2):
        a1,b1=map(xy_for,e1); a2,b2=map(xy_for,e2)
        o1=orient(a1,b1,a2); o2=orient(a1,b1,b2); o3=orient(a2,b2,a1); o4=orient(a2,b2,b1); eps=1e-14
        if abs(o1)<eps and abs(o2)<eps and abs(o3)<eps and abs(o4)<eps:
            if set(e1)&set(e2): return False
            axis=int(abs(b1-a1).argmax()); lo=max(min(a1[axis],b1[axis]),min(a2[axis],b2[axis])); hi=min(max(a1[axis],b1[axis]),max(a2[axis],b2[axis]))
            return hi-lo>1e-14
        return o1*o2 < -eps*eps and o3*o4 < -eps*eps
    for i in range(len(entries)):
        for j in range(i+1,len(entries)):
            e1,o1=entries[i]; e2,o2=entries[j]
            if o1[:3]==o2[:3]: continue
            if proper(e1,e2):
                virtual_same_face_crossings.append({"fine_face_row":fi,"edge_a":list(e1),"edge_b":list(e2),"route_a":list(o1[:3]),"route_b":list(o2[:3])})

all_boundary_vertices=set()
for comp in seed_components:
    _,_,loops,bad=boundary_loops(comp)
    for loop in loops: all_boundary_vertices.update(map(int,loop))
initial_missing=sorted(all_boundary_vertices-set(direct_anchor))
report={
 "status":"completed_topology_cut_feasibility",
 "interpretation":"source-topology feasibility only; no TetGen, material remap, capacity, self-intersection, or physical admission",
 "inputs":{"builder_source":str(Path(__file__)),"builder_source_sha256":sha(Path(__file__)), "prior_v3_source_status":"unavailable_exact_original_source_hash; this run uses a new targeted derivative", "proposal":str(PROP),"proposal_sha256":sha(PROP),"compact084":str(COARSE),"compact084_sha256":sha(COARSE),
           "direct_detail075":str(DETAIL),"direct_detail075_sha256":sha(DETAIL),
           "accepted_coarse_reference":str(COARSE_CSV),"accepted_coarse_reference_sha256":sha(COARSE_CSV)},
 "counts":{"selected_outer_edges":int(len(E)),"seed_source_O_faces":int(len(seed_faces)),
           "seed_face_components":int(len(seed_components)),"fine_detail_vertices":int(len(xyz)),
           "fine_detail_faces":int(len(fine_faces)),"direct_onehot_coarse_rows":int(len(direct_anchor)),
           "direct_anchor_duplicate_rows":int(sum(1 for v in anchor_candidates.values() if len(v)>1)),
           "seed_boundary_vertices":int(len(all_boundary_vertices)),"missing_direct_boundary_vertices":initial_missing,
           "virtual_exact_bary_anchor_count":int(len(virtual)),"virtual_anchor_rows":[v["coarse_row"] for v in virtual],
           "nonmanifold_coarse_edges":nonmanifold_edges[:50],"nonmanifold_coarse_edge_count":len(nonmanifold_edges),
           "subdivided_fine_face_count":int(len(subfaces)),"virtual_face_crossings":len(virtual_same_face_crossings)},
 "route_attempts":route_attempt_records,"selected_best_route_attempt":best_attempt_number,"selected_best_routed_loops":best_routed_count,"expansion_ring_diagnostics":expansion,"target_expansion":{"original_seed_component_ids":target_original_component_ids,"touched_original_component_ids":touched_original,"one_ring_plus_touched_seed_face_count":expanded_face_count_before_merge,"expanded_face_count_after_touched_component_merge":len(seed_faces),"expanded_component_original_sources":expanded_component_origins},"anchor_mapping_audit":anchor_mapping_audit,
 "virtual_anchors":virtual,"patch_loops":loop_records,
 "routed_segment_count":len(segment_records),
 "routed_segments":[{k:v for k,v in s.items() if k!="path_nodes"} for s in segment_records],
 "source_path_intersection_witnesses":virtual_same_face_crossings[:100],
 "fine_mesh_edge_incidence":{"edges":len(subedge_faces),"boundary_edge_count":sum(1 for x in subedge_faces.values() if len(x)==1),
                             "nonmanifold_edge_count":len(bad_subedges),"nonmanifold_edges":[[list(k),v] for k,v in list(bad_subedges.items())[:50]]},
 "patch_feasibility":{
   "all_seed_components_have_manifold_boundary":all(r["status"]!="boundary_not_manifold_cycle" for r in loop_records),
   "all_boundary_vertices_have_direct_or_virtual_exact_source_anchor":len(virtual)==len(initial_missing) and len(virtual)<=4,
   "all_loops_routed":all(r["status"]=="closed_disjoint_source_edge_cycle" for r in loop_records),
   "all_loops_cut_fine_face_graph":all(r.get("fine_cut_status")=="cut_edges_present" for r in loop_records if r["status"]=="closed_disjoint_source_edge_cycle"),
   "no_detected_path_crossings":len(virtual_same_face_crossings)==0,
   "feasible_for_source_patch_extraction":all(r["status"]=="closed_disjoint_source_edge_cycle" and r.get("fine_cut_status")=="cut_edges_present" and r.get("fine_face_component_count_after_cut",0)>=2 for r in loop_records) and len(virtual_same_face_crossings)==0
 }
}
np.savez_compressed(RUN/"source-patch-cut.npz",
    seed_source_O_face_rows=np.array(sorted(seed_faces),dtype=np.int32),
    patch_component_indptr=np.array([0]+list(np.cumsum([len(c) for c in seed_components])),dtype=np.int64),
    patch_component_face_rows=np.array([f for c in seed_components for f in sorted(c)],dtype=np.int32),
    virtual_anchor_nodes=np.array([v["node"] for v in virtual],dtype=np.int32),
    virtual_anchor_coarse_rows=np.array([v["coarse_row"] for v in virtual],dtype=np.int32),
    virtual_anchor_points_world_m=np.array([v["point_world_m"] for v in virtual],dtype=np.float64).reshape((-1,3)),
    virtual_anchor_source_face_detail_rows=np.array([v["source_face_detail_rows"] for v in virtual],dtype=np.int32).reshape((-1,3)),
    virtual_anchor_barycentric=np.array([v["barycentric"] for v in virtual],dtype=np.float64).reshape((-1,3)),
    route_segment_component=np.array([s["component"] for s in segment_records],dtype=np.int32),
    route_segment_loop=np.array([s["loop"] for s in segment_records],dtype=np.int32),
    route_segment_coarse_endpoints=np.array([[s["coarse_start"],s["coarse_end"]] for s in segment_records],dtype=np.int32).reshape((-1,2)),
    route_path_indptr=np.array([0]+list(np.cumsum([len(s["path_nodes"]) for s in segment_records])),dtype=np.int64),
    route_path_nodes=np.array([n for s in segment_records for n in s["path_nodes"]],dtype=np.int32),
    routed_cut_edges=np.array(sorted(used_edges),dtype=np.int32).reshape((-1,2)),
    fine_subfaces=subfaces.astype(np.int32),
    fine_subface_parent_rows=subface_parent.astype(np.int32))
report["outputs"]={"source_patch_cut_npz":str(RUN/"source-patch-cut.npz"),"source_patch_cut_sha256":sha(RUN/"source-patch-cut.npz")}
(RUN/"source-patch-cut-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"out":str(RUN),"report":report["patch_feasibility"],"counts":report["counts"],
                  "loop_status_counts":dict(Counter(r["status"] for r in loop_records)),
                  "routed_segments":len(segment_records),"path_crossings":len(virtual_same_face_crossings),
                  "hashes":{"report":sha(RUN/"source-patch-cut-report.json"),"npz":sha(RUN/"source-patch-cut.npz")}},indent=2))

