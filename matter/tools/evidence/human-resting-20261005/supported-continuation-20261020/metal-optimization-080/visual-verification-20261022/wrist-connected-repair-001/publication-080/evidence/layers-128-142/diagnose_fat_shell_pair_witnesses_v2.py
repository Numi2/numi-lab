from pathlib import Path
import gzip,csv,json,hashlib,importlib.util,numpy as np,sys
from fractions import Fraction
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
run=D/"nonmatching-layer-comparison-142/run-005-full146-baseline084"
asm=D/"coupled-layer-trial-128/assemble_layers_128.py"
spec=importlib.util.spec_from_file_location("assembler",asm);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
pred=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
sys.path.insert(0,str(m.PREDICATE_IMPORTS)); name="numilab_human.cardiac_cavity_intersections"; ps=importlib.util.spec_from_file_location(name,pred); p=importlib.util.module_from_spec(ps); sys.modules[name]=p; ps.loader.exec_module(p)
C=run/"candidate.npz"; z=np.load(C); X=z["points_world_m"].astype(np.float32); n=8776; nc=2411
Fc,_=m.face_csv(m.DEFAULT_FACE_CSV_GZ); Fc=Fc[:,[0,2,1]]
boundary=np.asarray([(v[0][1],v[0][2]) for v in m.edges_of_faces(Fc).values() if len(v)==1],dtype=np.int64).reshape((-1,2))
fat=z["fat_shell_faces"]; face_count=len(Fc); capbase=2*face_count
csvpath=m.DEFAULT_FACE_CSV_GZ
with gzip.open(csvpath,"rt") as f: source_rows=list(csv.DictReader(f))
stable={int(x["coarse_face_row"]):int(x["stable_source_face_id"]) for x in source_rows}
modified=set(map(int,np.load(D/"fine-layer-fields-127/capacity-115-coarse-to-detail-layer-fields.npz")["capacity115_modified_D_source_rows"]))
def face_info(fi):
 ids=fat[fi].astype(int)
 def layerrow(v):
  if n<=v<n+nc: return {"layer":"I","row":int(v-n)}
  if v>=n+nc: return {"layer":"D","row":int(v-n-nc)}
  return {"layer":"O","row":int(v)}
 if fi<face_count:
  typ="I"; row=fi; ancestry={"coarse_face_row":int(row),"stable_source_face_id":stable[int(row)]}
 elif fi<capbase:
  typ="D"; row=fi-face_count; ancestry={"coarse_face_row":int(row),"stable_source_face_id":stable[int(row)],"modified_by_capacity115":bool(any(int(v) in modified for v in Fc[row]))}
 else:
  k=fi-capbase; erow=k//2; edge=boundary[erow].tolist(); occ=m.edges_of_faces(Fc)[tuple(sorted(edge))]
  ancestry={"cap_boundary_edge_row":int(erow),"cap_triangle_in_quad":int(k%2),"coarse_boundary_directed_edge":edge,"stable_adjacent_face_ids":[stable[int(x[0])] for x in occ]}
  typ="cap"
 return {"shell_face":int(fi),"kind":typ,**ancestry,"point_ids":ids.tolist(),"point_layer_rows":[layerrow(int(v)) for v in ids],"vertices_world_f32_m":X[ids].astype(float).tolist()}
pairs=np.load(run/"fat_shell-pairs.npy").astype(int); witnesses=[]
for aa,bb in pairs:
 idsA=fat[aa].astype(int); idsB=fat[bb].astype(int); shared=sorted(set(idsA)&set(idsB))
 records,_=p.float32_surface_records(X,np.asarray([idsA,idsB],dtype=np.int64),source_seam_groups=())
 triA=[[Fraction(float(v)) for v in xyz] for xyz in X[idsA]]; triB=[[Fraction(float(v)) for v in xyz] for xyz in X[idsB]]; hits=sorted(set(p.triangle_intersection_points(triA,triB)))
 coincident=[]
 for ia in idsA:
  for ib in idsB:
   if ia!=ib and np.array_equal(X[ia],X[ib]): coincident.append([int(ia),int(ib)])
 def area(ids): return float(np.linalg.norm(np.cross(X[ids[1]].astype(float)-X[ids[0]],X[ids[2]].astype(float)-X[ids[0]]))/2)
 only_shared=bool(hits and shared and all(any(tuple(pt)==tuple(Fraction(float(x)) for x in X[v]) for v in shared) for pt in hits))
 witnesses.append({"pair":[int(aa),int(bb)],"same_indexed_vertices":[int(v) for v in shared],"coincident_distinct_ids":coincident,"exact_intersection_point_count":len(hits),"exact_intersection_points_world_m":[[float(c) for c in pt] for pt in hits],"intersection_only_at_common_indexed_vertex":only_shared,"triangle_areas_m2":[area(idsA),area(idsB)],"triangle_vertex_min_distance_m":float(np.linalg.norm(X[idsA][:,None,:].astype(float)-X[idsB][None,:,:].astype(float),axis=2).min()),"face_a":face_info(aa),"face_b":face_info(bb)})
report={"candidate_sha256":hashlib.sha256(C.read_bytes()).hexdigest(),"pair_file_sha256":hashlib.sha256((run/"fat_shell-pairs.npy").read_bytes()).hexdigest(),"predicate_sha256":hashlib.sha256(pred.read_bytes()).hexdigest(),"coarse_face_csv_sha256":hashlib.sha256(m.DEFAULT_FACE_CSV_GZ.read_bytes()).hexdigest(),"classification":"Exact Float32 predicate witnesses. These are not waived as seams; source IDs and exact intersection coordinates are retained.","witnesses":witnesses}
out=run/"fat_shell_pair_witnesses_exact_unique.json"; out.write_text(json.dumps(report,indent=2)+"\n"); print(out); print(json.dumps(report,indent=2))
