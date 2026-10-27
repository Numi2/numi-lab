#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict, Counter
from fractions import Fraction
import ast, hashlib, importlib.util, json, sys, time
import numpy as np

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027/b65-99/attempt-001'
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
HELPER=S/'per-bank-workers-20261027/b43-16/attempt-002/boundary_feasibility.py'
OUT=W/'topology-boundary-check.json'
BANK=[65,99]
EXPECTED_C='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
EXPECTED_A='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def ff(x): return Fraction.from_float(float(x))

# Reuse only tested local exact/distance predicates; do not execute the prior audit.
tree=ast.parse(HELPER.read_text())
wanted={'dot','sub','add','mul','cross','in_tri3','orient2','on_seg2','seg_intersect2','in_tri2','exact_segment_triangle','point_seg_dist','point_tri_dist','seg_seg_dist','seg_tri_dist'}
funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(HELPER),'exec'),globals())

started=time.monotonic()
assert sha(C)==EXPECTED_C and sha(A)==EXPECTED_A
with np.load(C,allow_pickle=False) as z: Z={k:z[k].copy() for k in z.files}
P=Z['points_world_m'].astype(np.float32).astype(np.float64)
F=Z['source_O_faces'].astype(np.int64)
audit=mod('bank65_audit159',AUDIT)
exact=audit.exact_module()
h=mod('bank65_helper159',audit.BANKHELPER)
bankdir=audit.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,bankkeys,pins,unused=h.read_bank_inputs(bankdir,audit.E)
mask=(ids[:,0]==BANK[0])&(ids[:,1]==BANK[1])
BT=np.asarray(T[mask],dtype=np.float64)
BID=np.asarray(ids[mask,3],dtype=np.int64)
local=(P-pos[BANK[0]])@M[BANK[0]].T-off[BANK[0]]
bank_records=audit.records(BT,BID,exact)
BTfrac=[tuple(tuple(ff(x) for x in v) for v in tri) for tri in BT]
# Exact coordinate-key edge incidence is a topology proxy because this bank export has no explicit corner IDs.
corner_key=lambda p: tuple(float(x).hex() for x in p)
bone_edges=defaultdict(list)
for j,tri in enumerate(BT):
    ks=[corner_key(v) for v in tri]
    for a,b in ((0,1),(1,2),(2,0)):
        ka,kb=ks[a],ks[b]; key=tuple(sorted((ka,kb)))
        bone_edges[key].append((j,ka,kb))
valence=Counter(len(v) for v in bone_edges.values())
bone_boundary=int(valence.get(1,0))
bone_nonmanifold=sum(n for d,n in valence.items() if d>2)
bone_orientation_conflicts=sum(1 for vals in bone_edges.values() if len(vals)==2 and vals[0][1:]==vals[1][1:])

br=next(x for x in json.loads(A.read_text())['banks'] if [int(x['body']),int(x['stable_bone_id'])]==BANK)
pairs=[tuple(map(int,p)) for p in br['result']['triangle_pairs']]
seed_faces=set(f for f,b in pairs)
# Exact edge adjacency and face fans for the skin.
edgefaces=defaultdict(list);nodefaces=defaultdict(list)
for fi,t in enumerate(F):
    for v in t: nodefaces[int(v)].append(fi)
    for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
        edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
face_adj=defaultdict(set)
for e,fs in edgefaces.items():
    for f in fs: face_adj[f].update(g for g in fs if g!=f)
# Crossing components are edge-connected; do not merge vertex-only contacts.
unseen=set(seed_faces); components=[]
while unseen:
    st=min(unseen);unseen.remove(st);q=[st];cc=[]
    while q:
        f=q.pop();cc.append(f)
        for g in face_adj[f]:
            if g in unseen and g in seed_faces: unseen.remove(g);q.append(g)
    components.append(sorted(cc))
components.sort(key=lambda x:x[0])
patches=[set(c) for c in components]

def boundary(patch):
    rows=[]
    for e,fs in edgefaces.items():
        ins=[f for f in fs if f in patch]
        outs=[f for f in fs if f not in patch]
        if ins and outs: rows.append((e,ins,outs))
        elif ins and len(fs)==1: rows.append((e,ins,[]))
    graph=defaultdict(list)
    for e,_,_ in rows:
        graph[e[0]].append(e[1]);graph[e[1]].append(e[0])
    unseen=set(graph);loops=[]
    while unseen:
        st=min(unseen);unseen.remove(st);q=[st];nodes=[]
        while q:
            n=q.pop();nodes.append(n)
            for m in graph[n]:
                if m in unseen:unseen.remove(m);q.append(m)
        loops.append(sorted(nodes))
    return rows,graph,loops

def ancestry(sf):
    rec={'skin_face_row':int(sf)}
    for k in ('parent_source_face_marker','parent_coarse_face_row','parent_fine_detail_face_row','parent_fine_subface_row'):
        if k in Z: rec[k]=int(Z[k][sf])
    for ptrkey,rowkey,outkey in (
        ('face_source_ancestry_indptr','face_source_ancestry_face_rows','source_face_ancestry_rows'),
        ('face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows','fine_detail_ancestry_rows'),
        ('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows','fine_subface_ancestry_rows'),
        ('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows','refinement_parent_ancestry_rows')):
        if ptrkey in Z and rowkey in Z:
            lo,hi=map(int,Z[ptrkey][sf:sf+2]);rec[outkey]=list(map(int,Z[rowkey][lo:hi]))
    return rec

def local_boundary_check(patch, comp_id, level):
    bedges,bg,loops=boundary(patch);bn=set(bg)
    degree_hist=dict(Counter(len(bg[n]) for n in bg))
    inside=[];node_records=[]
    for n in sorted(bn):
        q=tuple(ff(x) for x in local[n])
        locres=exact.point_location(q,bank_records)
        ds=[point_tri_dist(local[n],tr) for tr in BT]
        j=int(np.argmin(ds))
        r={'skin_node_row':int(n),'exact_location':locres.get('location','unknown'),'exact_location_ray':locres.get('ray'),'exact_crossings':locres.get('crossings'),'nearest_target_bone_face_row':int(BID[j]),'distance_mm':float(ds[j]*1000)}
        node_records.append(r)
        if r['exact_location'] in ('inside','boundary'):inside.append(r)
    edge_hits=[];edge_mins=[]
    for e,ins,outs in bedges:
        u,v=map(int,e);p0=tuple(ff(x) for x in local[u]);p1=tuple(ff(x) for x in local[v])
        lo=np.minimum(local[u],local[v]);hi=np.maximum(local[u],local[v])
        cand=np.flatnonzero(np.all(BT.max(1)>=lo,axis=1)&np.all(BT.min(1)<=hi,axis=1))
        hits=[]
        for j in cand:
            r=exact_segment_triangle(p0,p1,BTfrac[j])
            if r['hit']:hits.append({'target_bone_face_row':int(BID[j]),'kind':r['kind'],'point_local_m':[str(x) for x in r['point']] if r.get('point') else None})
        if hits:edge_hits.append({'edge_node_rows':[u,v],'inside_faces':ins,'outside_faces':outs,'hits':hits})
        ds=[seg_tri_dist(local[u],local[v],tr) for tr in BT]
        j=int(np.argmin(ds))
        edge_mins.append({'edge_node_rows':[u,v],'inside_faces':ins,'outside_faces':outs,'nearest_target_bone_face_row':int(BID[j]),'distance_mm':float(ds[j]*1000)})
    node_min=min(node_records,key=lambda r:r['distance_mm']) if node_records else None
    edge_min=min(edge_mins,key=lambda r:r['distance_mm']) if edge_mins else None
    boundary_nodes=set(bn);boundary_edges={tuple(sorted(e)) for e,_,_ in bedges}
    incident_faces={f for n in boundary_nodes for f in nodefaces[n]}
    inside_patch_boundary={f for _,ii,_ in bedges for f in ii}
    outside_boundary={f for _,_,oo in bedges for f in oo}
    other_patch_pairs=[];other_boundary_pairs=[]
    for bank in json.loads(A.read_text())['banks']:
        key=[int(bank['body']),int(bank['stable_bone_id'])]
        if key==BANK or not bank.get('result'):continue
        for sf,bf in bank['result']['triangle_pairs']:
            sf=int(sf);bf=int(bf);fv=set(map(int,F[sf]));sn=sorted(fv&boundary_nodes)
            se=[];t=F[sf]
            for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
                ee=tuple(sorted((int(a),int(b))))
                if ee in boundary_edges:se.append(list(ee))
            rel=[]
            if sf in patch: rel.append('inside_patch_crossing')
            if sf in inside_patch_boundary: rel.append('inside_patch_boundary_face')
            if sf in outside_boundary: rel.append('outside_boundary_adjacent_face')
            if sf in incident_faces: rel.append('boundary_node_incident_face')
            if se: rel.append('shares_boundary_edge')
            if sn: rel.append('shares_boundary_node')
            if rel:
                rec={'typed_bank':key,'skin_face_row':sf,'bone_face_row':bf,'relation':rel,'shared_boundary_nodes':sn,'shared_boundary_edges':se}
                other_patch_pairs.append(rec)
                if sf in incident_faces or sn or se:other_boundary_pairs.append(rec)
    simple=bool(loops) and all(all(len(bg[n])==2 for n in loop) for loop in loops)
    return {'component_id':comp_id,'ring_level':level,'patch_face_count':len(patch),'boundary_edge_count':len(bedges),'boundary_node_count':len(bn),'boundary_loop_count':len(loops),'boundary_loop_node_counts':sorted(map(len,loops)),'boundary_degree_histogram':degree_hist,'all_boundary_loops_simple':simple,'boundary_exact_inside_or_on_node_count':len(inside),'boundary_inside_or_on_witnesses':inside,'minimum_boundary_node':node_min,'exact_target_bone_boundary_edge_hit_count':len(edge_hits),'exact_target_bone_boundary_edge_hit_witnesses':edge_hits,'minimum_boundary_edge':edge_min,'stored_other_bank_pairs_in_patch':len(other_patch_pairs),'stored_other_bank_pairs_touching_boundary':len(other_boundary_pairs),'stored_other_bank_boundary_witnesses':other_boundary_pairs[:200],'boundary_node_records':node_records,'boundary_edge_records':edge_mins}

initial_component_ids=[set(c) for c in components]
levels=[];merge_block=None
for level in range(3):
    level_rows=[local_boundary_check(p,ci,level) for ci,p in enumerate(patches)]
    levels.append(level_rows)
    if level==2:break
    adds=[]
    for ci,p in enumerate(patches):
        b,_,_=boundary(p)
        outside={f for _,_,outs in b for f in outs}
        outside-=set().union(*(initial_component_ids[j] for j in range(len(initial_component_ids)) if j!=ci))
        adds.append(outside)
    # Never merge separate crossing components; if the next edge ring overlaps or touches another patch, stop expansion.
    conflict=None
    for i in range(len(patches)):
        for j in range(i+1,len(patches)):
            if adds[i]&patches[j] or adds[j]&patches[i] or adds[i]&adds[j]:
                conflict={'component_pair':[i,j],'reason':'shared_candidate_face'};break
            if any(g in patches[j] for f in adds[i] for g in face_adj[f]) or any(g in patches[i] for f in adds[j] for g in face_adj[f]):
                conflict={'component_pair':[i,j],'reason':'edge_ring_would_connect_patches'};break
            ni={int(v) for f in patches[i]|adds[i] for v in F[f]}
            nj={int(v) for f in patches[j]|adds[j] for v in F[f]}
            if ni & nj:
                conflict={'component_pair':[i,j],'reason':'ring_patches_would_share_skin_node'};break
        if conflict:break
    if conflict:
        merge_block={'after_ring_level':level,'conflict':conflict}
        break
    patches=[p|adds[i] for i,p in enumerate(patches)]

# Preserve the full per-crossing-face source ancestry mapping.
pair_rows=[{'skin_face_row':sf,'target_bone_face_row':bf} for sf,bf in pairs]
ancestry_rows=[ancestry(sf) for sf in sorted(seed_faces)]
source_marker_counts=Counter(str(x.get('parent_source_face_marker')) for x in ancestry_rows)
coarse_counts=Counter(str(x.get('parent_coarse_face_row')) for x in ancestry_rows)
fine_counts=Counter(str(x.get('parent_fine_detail_face_row')) for x in ancestry_rows)
report={'schema':'numi.human.bank65_99.topology_boundary_feasibility.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_path':str(A),'audit159_sha256':sha(A),'exact_predicate_sha256':sha(audit.EXACT),'target_bank':BANK,'target_bone_face_count':int(len(BT)),'target_bone_corner_adjacency_proxy':{'triangle_count':len(BT),'edge_valence_histogram':dict(valence),'boundary_edge_count':bone_boundary,'nonmanifold_edge_count':bone_nonmanifold,'orientation_conflicting_two_face_edges':bone_orientation_conflicts,'identity_basis':'exact emitted local coordinate tuples, since the bank table does not expose corner IDs'},'stored_crossing_pair_count':len(pairs),'stored_crossing_skin_face_count':len(seed_faces),'edge_connected_component_count':len(components),'components':[{'component_id':i,'skin_face_count':len(c),'skin_face_rows':c,'target_bone_face_ids':sorted({bf for sf,bf in pairs if sf in set(c)})} for i,c in enumerate(components)],'stored_exact_crossing_pairs':pair_rows,'source_ancestry_by_crossing_skin_face':ancestry_rows,'source_ancestry_counts':{'source_face_marker_top20':source_marker_counts.most_common(20),'coarse_face_top20':coarse_counts.most_common(20),'fine_detail_face_top20':fine_counts.most_common(20)},'ring_expansion':'topology-only edge-adjacent face rings per edge-connected crossing component; components kept distinct; no vertices moved','boundary_checks_by_ring_level':levels,'ring_expansion_stopped_to_preserve_components':merge_block,'native_admission':False,'accepted_physical_steps':0,'whole_body_or_all_bank_audit_run':False,'coordinate_rule':'emitted Float32 world points promoted to Float64 and transformed by pinned body 65 map; exact predicates use Fraction.from_float on those transformed stored values','distance_predicate':'Float64 Euclidean point/segment-to-target-triangle distance; distances are diagnostic with no thickness threshold','limitations':['Neighbor-bank witnesses come from the pinned stored audit-159 report only; no bank rerun.','Boundary point inside/on classification uses the exact rational point_location predicate on the target bank triangles.','Bone corner connectivity is an exact-coordinate proxy because the export has no explicit corner IDs.','This is a topology/boundary feasibility diagnostic, not a geometry repair, tissue-thickness proof, or physical admission.'],'elapsed_seconds':time.monotonic()-started}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
summary={'status':'COMPLETE_LOCAL_TOPOLOGY_BOUNDARY_DIAGNOSTIC','crossing_pairs':len(pairs),'crossing_skin_faces':len(seed_faces),'components':[{'id':i,'faces':len(c)} for i,c in enumerate(components)],'bone_topology':report['target_bone_corner_adjacency_proxy'],'rings':[[{k:r[k] for k in ['component_id','ring_level','patch_face_count','boundary_edge_count','boundary_node_count','boundary_loop_count','boundary_loop_node_counts','boundary_degree_histogram','all_boundary_loops_simple','boundary_exact_inside_or_on_node_count','minimum_boundary_node','exact_target_bone_boundary_edge_hit_count','minimum_boundary_edge','stored_other_bank_pairs_in_patch','stored_other_bank_pairs_touching_boundary']} for r in rows] for rows in levels],'merge_block':merge_block,'elapsed_seconds':report['elapsed_seconds']}
print(json.dumps(summary,indent=2,allow_nan=False))
