from pathlib import Path
import numpy as np,json,sys,hashlib
from scipy.optimize import linprog
D=Path("/Users/numi/Work/numi-human-connected-surface-069");P=D/'source-layer-feasibility-009';P.mkdir(exist_ok=False)
sys.path.insert(0,str(D/'canonical/matter/tools'));import human_skin_fat_connected_repair as repair
I=Path('/Users/n/Work/294-source-frame-audit-001/connected-interior-reference-fit-001/retessellation-001/volume-cavity-001/oid-cavity-replacement.npz')
z=np.load(I);T=z['candidate_tetrahedra'];R=z['reference_points_AP_m'];X=z['target_points_AP_m']
J,G,q,Q=repair.cell_values_gradients(X,R,T)
rows=np.flatnonzero(np.sum(T<2411,axis=1)==3);groups={}
for row in rows:
 where=np.flatnonzero(T[row]>=2411);assert len(where)==1
 k=int(where[0]);node=int(T[row,k]);assert node<4822 and node-2411 in T[row]
 g=G[row,k];g=g/np.linalg.norm(g)
 groups.setdefault(node,[]).append((int(row),g))
results=[]
for node,items in sorted(groups.items()):
 A=np.array([np.r_[-g,1.]for row,g in items])
 sol=linprog([0,0,0,-1.],A_ub=A,b_ub=np.zeros(len(A)),bounds=[(-1,1)]*3+[(-2,2)],method='highs')
 assert sol.success,sol.message
 results.append({'I_node':node,'O_node':node-2411,'incident_tet_rows':[row for row,g in items],'maximum_inward_margin':float(sol.x[3]),'inward_direction':sol.x[:3].tolist(),'infeasible_strict_inward_cone':bool(sol.x[3]<=1e-9)})
out={'input_sha256':hashlib.sha256(I.read_bytes()).hexdigest(),'fixed_O_face_tets':len(rows),'tested_I_nodes':len(results),'strict_inward_cone_infeasible_nodes':sum(r['infeasible_strict_inward_cone']for r in results),'minimum_inward_margin':min(r['maximum_inward_margin']for r in results),'results':results,'geometry_admitted':False,'accepted_physical_steps':0,'limitations':['Necessary per-star strict halfspace feasibility only. Does not establish full tetrahedral feasibility, skin/fat closure, contact or native physics.']}
(P/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='results'},indent=2))
