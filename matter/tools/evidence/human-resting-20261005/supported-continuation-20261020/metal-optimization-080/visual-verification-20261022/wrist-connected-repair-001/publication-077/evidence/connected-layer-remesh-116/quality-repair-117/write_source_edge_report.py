import csv,json,hashlib,itertools
from pathlib import Path
import numpy as np
base=Path("/Users/numi/Work/numi-human-connected-surface-069/connected-layer-remesh-116")
outdir=base/"quality-repair-117"
src=base/"run-001/candidate.npz"; final=outdir/"run-002/candidate-tetrahedra.npz"
coarse=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/accepted-transfer-row-export-075/run-001/source-preflight/accepted-coarse-faces.csv")
poly=base/"run-001/whole-skin-fat-reference.poly"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=np.load(src);X=z["points_world_f32"].astype(np.float64);T=z["tetrahedra"];F=z["surface_faces_zero_based"];FM=z["surface_face_markers"];srcrow=z["source_input_row"];outid=z["output_node_ids"]
bad=[]
for r in (14022,23439,30475):
 t=list(map(int,T[r]));edges=sorted([(float(np.linalg.norm(X[a]-X[b])),a,b) for a,b in itertools.combinations(t,2)],reverse=True)
 surf=[]
 for j,f in enumerate(F):
  if all(a in f for a in t):surf.append({"surface_face_row":j,"marker":int(FM[j]),"nodes":list(map(int,f))})
 bad.append({"input_tet_row":r,"cell_uid":int(z["cell_uids"][r]),"material_id":int(z["material_ids"][r]),"nodes":t,"longest_edges":[{"length_m":l,"nodes":[a,b],"source_input_rows":[int(srcrow[a]),int(srcrow[b])],"output_node_ids":[int(outid[a]),int(outid[b])]} for l,a,b in edges[:3]],"tet_faces_in_boundary":surf})
# Locate surface facets containing the key long edge and parse source coarse faces.
edge=(344,2332); face_rows=[]
for i,f in enumerate(F):
 if edge[0] in f and edge[1] in f:face_rows.append({"row":i,"marker":int(FM[i]),"nodes":list(map(int,f))})
coarse_rows=[]
with coarse.open(newline="") as f:
 for r in csv.DictReader(f):
  tri=[int(r[f"vertex{i}"]) for i in range(3)]
  if 344 in tri and 2332 in tri:coarse_rows.append({"coarse_face_row":int(r["coarse_face_row"]),"stable_source_face_id":int(r["stable_source_face_id"]),"vertices":tri})
poly_matches=[]
lines=poly.read_text().splitlines()
for i,line in enumerate(lines):
 parts=line.split()
 if len(parts)>=3 and parts[0]=="1" and parts[1]=="0" and parts[2] in ("100717","100718"):
  poly_matches.append({"poly_line_1based":i+1,"marker":int(parts[2]),"one_based_point_ids":list(map(int,lines[i+1].split()[1:4]))})
report={"input_path":str(src),"input_sha256":sha(src),"coarse_faces_075_path":str(coarse),"coarse_faces_075_sha256":sha(coarse),"input_poly_path":str(poly),"input_poly_sha256":sha(poly),"candidate_run002_path":str(final),"candidate_run002_sha256":sha(final),"bad_tets":bad,"key_outer_coarse_edge":{"node_rows":[344,2332],"output_node_ids":[int(outid[344]),int(outid[2332])],"source_input_rows":[int(srcrow[344]),int(srcrow[2332])],"length_m":float(np.linalg.norm(X[344]-X[2332])),"coarse_face_rows":coarse_rows,"output_surface_facets":face_rows,"poly_parent_facets":poly_matches,"interpretation":"Original O-layer coarse edge participates in both source parent facets; it is also a marked boundary edge in the 116 mesh, including a small child facet [2332,344,7318]. This points to a source-boundary refinement trigger based on cell quality as well as bone-hit witnesses; no source boundary point was moved here."},"classification":"diagnostic only; no physical or mass admission"}
p=outdir/"source-edge-q-witness.json";p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))
