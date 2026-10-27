from pathlib import Path
p=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/steiner-multinode-004-cdt-guided/cdt_patch_trial.py')
s=p.read_text()
s=s.replace("out=S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided'","out=S/'per-bank-workers-20261027/b74-104/steiner-multinode-004-cdt-guided'")
s=s.replace('va=d3*d6-d3*d5','va=d3*d6-d5*d4')
start=s.index('# Exact crossing endpoints form one closed degree-2 contour.')
end=s.index('# Outward normal for exact contour vertices, certified by exact point location on both sides.')
replacement=r"""# Validate the supplied SciPy constrained-Delaunay arrays independently before any 3D node fitting.
cdt_path=out/'planar-triangulation.npz'
cz=np.load(cdt_path,allow_pickle=False)
cp=np.asarray(cz['points_barycentric'],np.float64)
ct=np.asarray(cz['triangles'],np.int64)
ccap=np.asarray(cz['cap'],bool)
edges=[tuple(map(int,e)) for e in graph['segments_point_indices']]
npts=len(graph['intersection_points_barycentric'])
if cp.shape!=(31,3) or ct.shape!=(54,3) or ccap.shape!=(54,):raise RuntimeError('constrained-Delaunay payload has unexpected shapes')
if not np.array_equal(np.asarray(cz['outer_nodes'],np.int64),np.asarray(loop,np.int64)):raise RuntimeError('CDT outer-node ordering differs from certified boundary loop')
if not np.allclose(cp[:6],np.asarray([node_bary[n] for n in loop]),rtol=0,atol=1e-12):raise RuntimeError('CDT fixed-boundary barycentric rows do not map to their original identities')
if not np.allclose(cp[6:],np.asarray(graph['intersection_points_barycentric'],np.float64),rtol=0,atol=1e-12):raise RuntimeError('CDT contour rows do not match original exact segment-graph point IDs')
if len(cp)!=len(np.unique(np.round(cp,13),axis=0)):raise RuntimeError('CDT contains duplicate barycentric nodes')
if not np.allclose(cp.sum(axis=1),1.0,rtol=0,atol=1e-12) or np.min(cp)<-1e-9 or np.max(cp)>1.0+1e-9:raise RuntimeError('CDT barycentric nodes leave the source face')
if np.min(ct)<0 or np.max(ct)>=len(cp) or any(len(set(map(int,t)))!=3 for t in ct):raise RuntimeError('CDT triangle connectivity contains invalid or repeated node rows')
used=sorted(set(map(int,ct.ravel())))
if used!=list(range(len(cp))):raise RuntimeError(f'CDT has unused nodes: {sorted(set(range(len(cp)))-set(used))}')
outer_edges_in=np.asarray(cz['outer_edges'],np.int64);contour_edges_in=np.asarray(cz['contour_edges'],np.int64)
if outer_edges_in.shape!=(6,2) or contour_edges_in.shape!=(25,2):raise RuntimeError('CDT constraints have unexpected counts')
outer_constraint_set={tuple(sorted(map(int,e))) for e in outer_edges_in}
expected_outer_set={tuple(sorted((i,(i+1)%6))) for i in range(6)}
contour_graph_set={tuple(sorted((6+a,6+b))) for a,b in edges}
if outer_constraint_set!=expected_outer_set:raise RuntimeError('CDT outer constraints do not preserve all six certified boundary edges')
if {tuple(sorted(map(int,e))) for e in contour_edges_in}!=contour_graph_set:raise RuntimeError('CDT contour constraints do not match all 25 exact input segments')
outer_bary=cp[:6].copy();inner_bary=cp[6:].copy()
outer_xy=outer_bary[:,1:];inner_bary=inner_bary.copy()
inner_xy_all=inner_bary[:,1:]
# Derive the exact contour cycle from exact segment adjacency; keep CDT node rows in original report order.
adj=[[] for _ in range(npts)]
for a,b in edges:adj[a].append(b);adj[b].append(a)
degrees=[len(q) for q in adj]
if any(q!=2 for q in degrees):raise RuntimeError(f'exact segment graph is not a degree-2 loop: {Counter(degrees)}')
cycle=[0];prev=-1;cur=0
while True:
 opts=[q for q in adj[cur] if q!=prev]
 if not opts:raise RuntimeError('exact contour graph is open')
 nxt=opts[0]
 if nxt==cycle[0]:break
 if nxt in cycle:raise RuntimeError('exact contour graph has an early cycle')
 cycle.append(nxt);prev,cur=cur,nxt
if len(cycle)!=npts:raise RuntimeError('exact contour graph has multiple components')
inner_xy=inner_xy_all[cycle]
outer_area=abs(area2(outer_xy));inner_area=abs(area2(inner_xy));center=poly_centroid(inner_xy)
outer_star,outer_angle_diffs=is_star(outer_xy,center);inner_star,inner_angle_diffs=is_star(inner_xy,center)
def row_token(r):return ('o',int(r)) if int(r)<6 else ('i',int(r)-6)
base_tris=[[row_token(q) for q in tri] for tri in ct]
cap=[base_tris[i] for i in range(len(base_tris)) if ccap[i]]
ann=[base_tris[i] for i in range(len(base_tris)) if not ccap[i]]
def token_bary(t):return outer_bary[t[1]] if t[0]=='o' else inner_bary[t[1]]
def token_node(t):return loop[t[1]] if t[0]=='o' else len(X)+t[1]
def token_xy(t):return token_bary(t)[1:]
def edgekey(a,b):return tuple(sorted((a,b)))
inc=Counter()
for tri in base_tris:
 for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):inc[edgekey(a,b)]+=1
outer_expected={edgekey(('o',i),('o',(i+1)%6)) for i in range(6)}
inner_expected={edgekey(('i',a),('i',b)) for a,b in edges}
constraint_incidence_ok=all(inc[e]==1 for e in outer_expected) and all(inc[e]==2 for e in inner_expected) and all(v==2 for e,v in inc.items() if e not in outer_expected)
xy=np.asarray(cp[:,1:],np.float64)
def onseg(p,a,b,tol=1e-12):
 ab=b-a;ap=p-a
 return abs(ab[0]*ap[1]-ab[1]*ap[0])<=tol and float(ap@ab)>tol and float(ap@ab)<float(ab@ab)-tol
t_junctions=[]
all_edges={tuple(sorted((int(a),int(b))) for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])) for t in ct}
for e in all_edges:
 for v in range(len(cp)):
  if v in e:continue
  if onseg(xy[v],xy[e[0]],xy[e[1]]):t_junctions.append([v,list(e)])
crossings=[]
edge_list=sorted(all_edges)
for ii0,e in enumerate(edge_list):
 for f in edge_list[ii0+1:]:
  if set(e)&set(f):continue
  if proper_intersects(xy[e[0]],xy[e[1]],xy[f[0]],xy[f[1]]):crossings.append([list(e),list(f)])
areas=np.abs(np.asarray([cross2(xy[t[0]],xy[t[1]],xy[t[2]])*0.5 for t in ct]))
cap_area=float(areas[ccap].sum());ann_area=float(areas[~ccap].sum());tri_area=float(areas.sum())
def point_in_poly0(p,poly):return point_in_poly(p,poly)
domain_ok=True
for ri,t in enumerate(ct):
 cxy=xy[t].mean(axis=0);inside=point_in_poly0(cxy,inner_xy)
 if bool(ccap[ri])!=bool(inside):domain_ok=False
ancestry_ok=all(abs(float(np.sum(b))-1.0)<=1e-12 and np.min(b)>=-1e-12 and np.max(b)<=1.0+1e-12 for b in cp)
simple_crossings=len(crossings)
planar_ok=(abs(tri_area-outer_area)<=1e-12 and abs(cap_area-inner_area)<=1e-12 and abs(ann_area-(outer_area-inner_area))<=1e-12 and constraint_incidence_ok and not t_junctions and not crossings and domain_ok and ancestry_ok)
planar={'schema':'numi.human.b74_104_cdt_independent_planar_check.v1','status':'PASS' if planar_ok else 'FAIL','candidate_sha256':basehash,'root_planar_report_path':str(out/'root-planar-report.json'),'root_planar_report_sha256':hashlib.sha256((out/'root-planar-report.json').read_bytes()).hexdigest(),'cdt_npz_path':str(cdt_path),'cdt_npz_sha256':hashlib.sha256(cdt_path.read_bytes()).hexdigest(),'segment_graph_report_sha256':hashlib.sha256((out/'segment-graph-report.json').read_bytes()).hexdigest(),'outer_node_identity_mapping':[{'row':i,'node':loop[i],'barycentric':cp[i].tolist()} for i in range(6)],'contour_node_identity_mapping':[{'row':6+i,'graph_point_id':i,'output_node':len(X)+i,'barycentric':cp[6+i].tolist()} for i in range(npts)],'fixed_boundary_edge_count':len(outer_expected),'exact_contour_segment_count':len(inner_expected),'edge_incidence_ok':constraint_incidence_ok,'unused_nodes':[],'t_junctions':t_junctions,'nonadjacent_edge_crossings':crossings,'outer_area_barycentric2':outer_area,'contour_area_barycentric2':inner_area,'triangulated_area_barycentric2':tri_area,'annulus_area_barycentric2':ann_area,'cap_area_barycentric2':cap_area,'coverage_error_barycentric2':tri_area-outer_area,'minimum_triangle_area_barycentric2':float(areas.min()),'triangle_count':len(ct),'domain_centroid_classification_pass':domain_ok,'source_face_barycentric_ancestry_pass':ancestry_ok,'immediate_parent_face_row':22189,'fine_detail_face_row':10342,'fine_subface_row':10344}
(out/'planar-triangulation-report.json').write_text(json.dumps(planar,indent=2)+'\n')
if not planar_ok:raise RuntimeError('independent CDT coverage/constraint/nonoverlap/ancestry gate failed')
"""
s=s[:start]+replacement+s[end:]
s=s.replace("for i,b in enumerate(inner_bary):new_specs.append(('segment',tuple(sorted(incident[cycle[i]]))))","for i,b in enumerate(inner_bary):new_specs.append(('segment',tuple(sorted(incident[i]))))")
p.write_text(s)
print('installed independent CDT validation and geometry mapping')

