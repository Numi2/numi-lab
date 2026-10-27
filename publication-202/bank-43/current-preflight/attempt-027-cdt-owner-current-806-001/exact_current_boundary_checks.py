#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import ast,csv,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
B=S/"per-bank-workers-20261027/combined-d071-b109-b61-001"
D=Path(__file__).resolve().parent
C=B/"candidate.npz";A=B/"audit-159/report.json";I=B/"audit-130/summary.json";PREF=D/"bulk-patch-topology.json"
AUDIT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py")
BANKHELP=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
BANKDIR=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXPORT=E/"E/generic-static-query-042/prepare_generic_query_points.py"
QUERYBIN=Path("/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query")
HELPER_PRED=S/"per-bank-workers-20261027/b43-16/attempt-002/boundary_feasibility.py"
OUT=D/"exact-source-boundary-checks.json"
PINS={"candidate":"806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756","audit159":"4cc8fd5d5e8fb651733fd380001e2b9847a0040960d7c8865c37f25fe5b8a09d","audit130":"1291c50ddd6ac65e031f04bc586dfd7d7d3806b16338e245dbe65f9aabfc0eb4"}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mod(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise ImportError(path)
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def main():
 start=time.monotonic()
 pins={"candidate":sha(C),"audit159":sha(A),"audit130":sha(I)}
 if pins!=PINS:raise ValueError(f"input pins changed: {pins}")
 pre=json.loads(PREF.read_text())
 top=pre["final_topology"]
 if pre["baseline_sha256"]!=PINS["candidate"] or not top["boundary_simple_loops"] or not pre["final_ancestry_complete"]:raise ValueError("bulk boundary preflight is not valid")
 boundary_edges=[tuple(map(int,x["edge"])) for x in top["boundary_edges"]]
 boundary_nodes=np.asarray(top["boundary_nodes"],dtype=np.int64)
 with np.load(C,allow_pickle=False) as z:
  world=np.asarray(z["points_world_m"],dtype=np.float32)
 # Exact rational segment/triangle test functions from the previous pinned boundary checker.
 tree=ast.parse(HELPER_PRED.read_text())
 wanted={"ff","dot","sub","add","mul","cross","in_tri3","orient2","on_seg2","seg_intersect2","in_tri2","exact_segment_triangle"}
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
 ns={"Fraction":Fraction}
 exec(compile(ast.Module(body=funcs,type_ignores=[]),str(HELPER_PRED),"exec"),ns)
 ff=ns["ff"];exact_segment_triangle=ns["exact_segment_triangle"]
 helper=mod("source8265_boundary_bank_helper",BANKHELP)
 T,ids,gids,bodies,M,offset,pos,keys,input_pins,unused=helper.read_bank_inputs(BANKDIR,E)
 bankkeys=[(int(x[0]),int(x[1])) for x in keys]
 if len(bankkeys)!=185:raise ValueError(f"expected 185 typed banks, got {len(bankkeys)}")
 edge_hits=[];edge_tests=0;bank_edge_stats=[]
 for body,bone in bankkeys:
  mask=(ids[:,0]==body)&(ids[:,1]==bone)
  bt=np.asarray(T[mask],dtype=np.float64);bids=np.asarray(ids[mask,3],dtype=np.int64)
  if not len(bt):raise ValueError(f"empty bank {body}/{bone}")
  local=(world.astype(np.float64)-pos[body])@M[body].T-offset[body]
  blo=bt.min(axis=1);bhi=bt.max(axis=1)
  bf=[tuple(tuple(ff(v) for v in xyz) for xyz in tri) for tri in bt]
  bank_tests=0;bank_hits=0
  for u,v in boundary_edges:
   p0=local[u];p1=local[v];lo=np.minimum(p0,p1);hi=np.maximum(p0,p1)
   cand=np.flatnonzero(np.all(bhi>=lo,axis=1)&np.all(blo<=hi,axis=1))
   for j in cand:
    edge_tests+=1;bank_tests+=1
    q=exact_segment_triangle(tuple(ff(x) for x in p0),tuple(ff(x) for x in p1),bf[int(j)])
    if q["hit"]:
     bank_hits+=1
     point=q.get("point")
     edge_hits.append({"edge_node_rows":[u,v],"bank_body":body,"stable_bone_id":bone,"bone_triangle_row":int(bids[int(j)],),"intersection_kind":q["kind"],"intersection_point_local_m":[str(x) for x in point] if point is not None else None})
  bank_edge_stats.append({"bank_body":body,"stable_bone_id":bone,"triangle_count":len(bt),"exact_aabb_candidate_tests":bank_tests,"exact_segment_triangle_hits":bank_hits})
 # Exact all-185 node query with the pinned native static-query binary.
 qdir=D/"boundary-point-query-rerun-001";qdir.mkdir(exist_ok=True)
 qnpz=qdir/"points-world-f32.npz";np.savez_compressed(qnpz,points_world_f32=world[boundary_nodes])
 qcsv=qdir/"points.csv";qpairs=qdir/"pairs.csv";qnodes=qdir/"nodes.csv"
 cp=subprocess.run([sys.executable,str(EXPORT),"--candidate-npz",str(qnpz),"--input-sha256",sha(qnpz),"--field","points_world_f32","--output",str(qcsv)],capture_output=True,text=True)
 (qdir/"prepare.log").write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError(f"point export failed {cp.returncode}: {cp.stderr}")
 cp=subprocess.run([str(QUERYBIN),str(BANKDIR/"exact-emitted.triangles.csv"),str(BANKDIR/"exact-emitted.bodies.csv"),str(qcsv),str(qpairs),str(qnodes),"185"],capture_output=True,text=True)
 (qdir/"query.log").write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError(f"exact all-bank node query failed {cp.returncode}: {cp.stderr}")
 points=list(csv.DictReader(qcsv.open()));node_rows=list(csv.DictReader(qnodes.open()));pair_rows=list(csv.DictReader(qpairs.open()))
 if not (len(points)==len(node_rows)==len(boundary_nodes)):raise ValueError("query row count mismatch")
 bits=world.view(np.uint32)
 for i,(p,nr) in enumerate(zip(points,node_rows)):
  node=int(boundary_nodes[i])
  if int(p["query_row"])!=i or int(nr["query_row"])!=i:raise ValueError("query order mismatch")
  if any(int(p[k])!=int(bits[node,j]) for j,k in enumerate(("x_bits_u32","y_bits_u32","z_bits_u32"))):raise ValueError("query point differs from baseline F32 node")
  if int(nr["total_bone_banks"])!=185 or int(nr["ambiguous_banks"])!=0 or int(nr["invalid_banks"])!=0:raise ValueError(f"incomplete or ambiguous all-185 query for node {node}: {nr}")
  if sum(int(nr[k]) for k in ("aabb_proven_outside_banks","parity_outside_banks","inside_banks","ambiguous_banks","invalid_banks"))!=185:raise ValueError(f"bank count mismatch for node {node}")
 node_inside={int(n):[] for n in boundary_nodes}
 for r in pair_rows:
  if int(r["ambiguous"])!=0 or r["classification"] not in ("inside","outside"):raise ValueError(f"ambiguous pair query: {r}")
  if r["classification"]=="inside":node_inside[int(boundary_nodes[int(r["query_row"])])].append({"bank_body":int(r["bank_body"]),"stable_bone_id":int(r["bank_stable_id"]),"classification":r["classification"]})
 node_summary=[]
 for i,nr in enumerate(node_rows):
  node=int(boundary_nodes[i]);node_summary.append({"node_row":node,"inside_bank_count":sum(1 for x in node_inside[node]),"inside_banks":node_inside[node],"ambiguous_banks":int(nr["ambiguous_banks"]),"invalid_banks":int(nr["invalid_banks"]),"total_bone_banks":int(nr["total_bone_banks"])})
 out={"schema":"numi.human.b43-16.source8265-exact-boundary-checks.v1","status":"BOUNDARY_CHECKS_COMPLETE","baseline_candidate_path":str(C),"baseline_candidate_sha256":pins["candidate"],"audit159_sha256":pins["audit159"],"audit130_sha256":pins["audit130"],"source_boundary_preflight_path":str(PREF),"source_boundary_preflight_sha256":sha(PREF),"expanded_source_union":pre["final_source_union"],"expanded_detail_union":pre["final_detail_union"],"expanded_patch_face_count":pre["final_patch_face_count"],"boundary_node_ids":boundary_nodes.tolist(),"boundary_node_count":len(boundary_nodes),"all_185_point_query_binary_sha256":sha(QUERYBIN),"all_185_point_query_exporter_sha256":sha(EXPORT),"all_185_bank_count_per_node_verified":True,"boundary_nodes_with_inside_banks":[x for x in node_summary if x["inside_bank_count"]],"boundary_node_query_results":node_summary,"exact_boundary_edge_scope":"all bulk-patch boundary edges against every triangle in all 185 typed bone banks using exact rational segment-triangle predicates after body-local Float64 transform of baseline F32 points and pinned bank triangles","exact_boundary_edge_count":len(boundary_edges),"exact_edge_aabb_candidate_tests":edge_tests,"exact_boundary_segment_triangle_hits":edge_hits,"exact_boundary_edge_hit_count":len(edge_hits),"per_bank_edge_stats":bank_edge_stats,"elapsed_seconds":time.monotonic()-start,"geometry_changed":False,"candidate_emitted":False,"physical_admission":False,"accepted_physical_steps":0}
 OUT.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({k:out[k] for k in ("status","boundary_node_count","boundary_nodes_with_inside_banks","exact_boundary_edge_count","exact_edge_aabb_candidate_tests","exact_boundary_edge_hit_count","exact_boundary_segment_triangle_hits","elapsed_seconds")},indent=2,sort_keys=True))
if __name__=="__main__":main()
