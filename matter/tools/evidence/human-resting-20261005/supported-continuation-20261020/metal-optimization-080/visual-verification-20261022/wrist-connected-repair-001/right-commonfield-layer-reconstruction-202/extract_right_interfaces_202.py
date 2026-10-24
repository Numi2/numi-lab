#!/usr/bin/env python3
from pathlib import Path
import sys
import importlib.util, hashlib, json, time
from collections import Counter
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE=E/'right-commonfield-layer-reconstruction-202';OUT=BASE/(sys.argv[1] if len(sys.argv)>1 else 'attempt-002')
spec=importlib.util.spec_from_file_location('geom',BASE/'tools/human_skin_patch_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
z=np.load(OUT/'right-commonfield-calibrated-202.npz',allow_pickle=False)
X=z['points_world_f32_m'];T=z['tetrahedra'];phi=z['phi']; start=time.monotonic();report={'physical_admission':False,'source_sha256':hashlib.sha256((OUT/'right-commonfield-calibrated-202.npz').read_bytes()).hexdigest(),'interfaces':{}}
for role,level,ring in zip(['I','D'],z['levels'],[z['I_cut_ids'],z['D_cut_ids']]):
 V,F,K,W=g.extract_p1_tetrahedral_interface(X,T,phi,level)
 used=np.unique(F); remap=np.full(len(V),-1,np.int64);remap[used]=np.arange(len(used));F=remap[F];V=V[used];K=K[used];W=W[used]
 cnt=Counter();dire=Counter()
 for f in F:
  for a,b in zip(f,np.roll(f,-1)):
   a,b=int(a),int(b);key=(min(a,b),max(a,b));cnt[key]+=1;dire[key]+=1 if a<b else -1
 boundary=[e for e,n in cnt.items() if n==1]
 exact_ids={int(k[0]):i for i,k in enumerate(K) if k[0]==k[1]}
 inherited={int(i):exact_ids.get(int(i)) for i in ring}
 expected={tuple(sorted((exact_ids[int(a)],exact_ids[int(b)]))) for a,b in zip(ring,np.roll(ring,-1)) if int(a) in exact_ids and int(b) in exact_ids}
 exact_cut=all(i is not None for i in inherited.values()) and set(boundary)==expected
 p=V.astype(float)[F]; areas=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2
 info={'level':float(level),'vertices':len(V),'faces':len(F),'boundary_edges':len(boundary),'incidence_histogram':dict(Counter(cnt.values())),'inconsistent_interior_edges':sum(n==2 and dire[e]!=0 for e,n in cnt.items()),'exact_inherited_cut_ring':bool(exact_cut),'missing_inherited_nodes':[int(i) for i,j in inherited.items() if j is None],'minimum_area_m2':float(areas.min()),'source_edge_ancestry_rows':len(K),'discarded_unreferenced_vertices':int(len(used)<len(remap))}
 np.savez_compressed(OUT/f'right-{role}-interface-202.npz',vertices_world_f32_m=V,faces=F,source_volume_edge_nodes=K,source_edge_endpoint_one_weight=W,inherited_cut_volume_node_ids=ring)
 report['interfaces'][role]=info;print(json.dumps({'role':role,**info}),flush=True)
report['preparation_pass']=all(i['exact_inherited_cut_ring'] and i['inconsistent_interior_edges']==0 and max(map(int,i['incidence_histogram']))<=2 for i in report['interfaces'].values())
report['wall_seconds']=time.monotonic()-start;(OUT/'interface-preflight-report-202.json').write_text(json.dumps(report,indent=2)+'\n')
if not report['preparation_pass']: raise SystemExit('interface preflight failed')
