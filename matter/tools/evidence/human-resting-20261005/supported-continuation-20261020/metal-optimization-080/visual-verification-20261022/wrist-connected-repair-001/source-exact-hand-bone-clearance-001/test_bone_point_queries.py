from pathlib import Path
import sys,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bone_point_queries import TrianglePointQuery,closest
rng=np.random.default_rng(77126);tri=rng.normal(size=(90,3,3));tri=np.concatenate((tri,[[[-100,-100,.01],[100,-100,.01],[0,100,.01]]]))
q=TrianglePointQuery(tri);checks=[]
for p in np.vstack((np.zeros((1,3)),rng.normal(size=(64,3))*2)):
 d,x,j,n=q.query(p);all_d,all_p=closest(p,tri);assert abs(d-all_d.min())<1e-13;assert abs(np.linalg.norm(p-x)-d)<1e-13;checks.append(n)
for data in (np.empty((0,3,3)),np.full((1,3,3),np.nan)):
 try:TrianglePointQuery(data)
 except ValueError:pass
 else:raise AssertionError('invalid triangles admitted')
for p in (np.array([np.nan,0,0]),np.zeros(2)):
 try:q.query(p)
 except ValueError:pass
 else:raise AssertionError('invalid point admitted')
report={'status':'PASS','seed':77126,'brute_force_comparison_cases':len(checks),'contains_large_triangle_with_far_centroid':True,'invalid_input_cases':4,'helper_sha256':hashlib.sha256(Path(__file__).with_name('bone_point_queries.py').read_bytes()).hexdigest(),'scope':'CPU unsigned geometry preparation query; not closed-solid containment or native dynamics'}
Path(__file__).with_name('bone-point-query-tests.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
