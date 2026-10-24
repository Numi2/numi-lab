from pathlib import Path
import importlib.util,sys,json,hashlib,numpy as np,time
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'global-inner-p1-active-layer-refinement-008'
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load(D/'active_layer_refinement_008.py','h008')
a=load(E/'global-inner-p1-field-owner-integration-007/field_owner_integration_007.py','a007r')
c=load(E/'robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py','c302r');r=h.owner403()
z=np.load(D/'round-000.npz');X=z['points_world_m'];T=z['tetrahedra'];O=z['physical_outer_node_ids'];S=z['physical_shell_faces']
bank=np.load(E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz');tri=bank['bank_0004_vertices'][bank['bank_0004_faces'][2562]]
results=[]
for level in range(4):
 cons,polys,meta=a.clip_typed_triangle_to_mesh(X,T,tri,c.triangle_tet_barycentric_polygon)
 caps=a.fixed_zero_capacity_report(T,cons,O,len(X))
 bad=np.asarray(sorted(set(x['tet_row'] for x in caps if x['max_free_capacity']<.341-1e-12)),np.int64)
 edges=h.select_internal_outer_free_edges(T,O,S,bad,len(X))
 X,T,p,ep=r.split_edges_globally(X,T,edges)
 expected=np.load(D/('round-%03d.npz'%(level+1)))
 ok=np.array_equal(X,expected['points_world_m']) and np.array_equal(T,expected['tetrahedra'])
 assert ok
 results.append({'round':level+1,'edges':len(edges),'candidate_arrays_exact':ok})
records={}
for p in [Path('/tmp/refine_active_layer_008.py'),Path('/tmp/finalize_active_layer_008.py'),D/'active_layer_refinement_008.py',E/'global-inner-p1-edge-support-403/refine_internal_o_edges_403.py',E/'global-inner-p1-field-owner-integration-007/field_owner_integration_007.py',E/'robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py',E/'global-inner-p1-field-402-absolute-variable-integration/absolute_field_402.py']:
 records[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
report={'pass':True,'all_four_rounds_exact':True,'rounds':results,'sources':records,'native_admission':False}
(D/'helper-replay-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

