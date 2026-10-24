#!/usr/bin/env python3
import csv,hashlib,json,math,subprocess,time,runpy,sys,inspect,ast
from fractions import Fraction
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.cardiac_cavity_intersections import _audit_record_pairs

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
OLD=E/"ankle-parent-tet-refinement-026/generated-224-commonfield-tetgen-volume-actual007/build_commonfield_224.py"
SHELLDIR=E/"bilateral-commonfield-shell-272"
BANK=E/"registered-surrounding-tissue-native007-190/registered-tissue-banks.npz"
INV=E/"registered-surrounding-tissue-native007-190/inventory.json"
POSE=E/"native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json"
CAND=E/"bilateral-source-cut-plane-271/candidate-271-max-cut-distance-bilateral-collar.npz"
TETGEN=E/"arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen"
OUT=E/"tissue-aware-commonfield-280-cached-output"
MAXVOL=1.5e-7
TARGET={"left":{"skin":0.0003310,"fat":0.0011475},"right":{"skin":0.0003491,"fat":0.0011945}}
OBSTACLE_MARGIN_BONE_M=0.00025
OBSTACLE_MARGIN_SOFT_M=0.00010
if OUT.exists():raise SystemExit(f"refusing to overwrite {OUT}")
OUT.mkdir(parents=True)

# Load the already-tested 224 exact predicates/geometry helpers without executing its main.
src=OLD.read_text()
src=src.replace('OUT=P/"generated-224-commonfield-tetgen-volume-actual007"',
                'OUT=P/"tissue-aware-commonfield-263"')
src=src.replace('src=COMMON.read_text()\nsrc=src.replace',
                'src=COMMON.read_text()\nsrc=src.replace("if vol<0:F=F[:,[0,2,1]];vol=-vol","")\nsrc=src.replace')
lib={"__name__":"tissue_aware_commonfield_263_lib"}
exec(compile(src,str(OLD),"exec"),lib)
# Add exact Float32-degeneracy witnesses to the retained helper failure.
def _install_level_witness():
 fn=lib["ns"]["extract_level"]
 source_path=Path(fn.__code__.co_filename)
 text=source_path.read_text()
 tree=ast.parse(text)
 node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="extract_level")
 original="\n".join(text.splitlines()[node.lineno-1:node.end_lineno])
 target=next((line for line in original.splitlines() if "F32 degenerate level faces" in line),None)
 replacement=' if np.any(area2<=1e-16):\n  bad=np.flatnonzero(area2<=1e-16)\n  witness=[]\n  for wi in bad[:8]:\n   tri=F[int(wi)];current_pts=[];stable_pts=[];edge_rows=[]\n   for pid0 in tri:\n    pid=int(pid0);lo,hi=map(int,pedge[pid]);tt=float(et[pid])\n    # Reproduce the helper\'s exact original arithmetic before its emitted F32 cast.\n    current=((1-tt)*V[lo]+tt*V[hi]).astype(np.float32)\n    # Compare with a numerically stable double interpolation followed by one F32 cast.\n    stable=((1.0-tt)*np.asarray(V[lo],np.float64)+tt*np.asarray(V[hi],np.float64)).astype(np.float32)\n    current_pts.append(current);stable_pts.append(stable)\n    edge_rows.append({"point_id":pid,"edge_node_ids":[lo,hi],"edge_t":tt,"endpoint_world_m":np.asarray([V[lo],V[hi]],np.float64).tolist()})\n   cp=np.asarray(current_pts,np.float32);sp=np.asarray(stable_pts,np.float32)\n   def area2_local(q):return float(np.linalg.norm(np.cross(q[1].astype(np.float64)-q[0].astype(np.float64),q[2].astype(np.float64)-q[0].astype(np.float64))))\n   uc=int(len(np.unique(cp,axis=0)));us=int(len(np.unique(sp,axis=0)));sa=area2_local(sp)\n   if uc<3 and us==3 and sa>1e-16:classification="current_f32_arithmetic_collapse_stable_cast_noncollinear"\n   elif uc<3:classification="exact_emitted_f32_coordinate_coincidence"\n   elif sa<=1e-16:classification="three_distinct_emitted_f32_points_collinear_after_stable_cast"\n   elif area2_local(cp)<=1e-16:classification="current_arithmetic_near_collinear_stable_cast_noncollinear"\n   else:classification="degenerate_threshold_witness_unclassified"\n   witness.append({"face_index":int(wi),"point_ids":tri.astype(int).tolist(),"classification":classification,\n    "current_f32_unique_points":uc,"stable_cast_f32_unique_points":us,"current_f32_points_m":cp.astype(float).tolist(),\n    "stable_float64_then_f32_points_m":sp.astype(float).tolist(),"current_area2_m2":float(area2[wi]),"stable_cast_area2_m2":sa,"parent_intersection_edges":edge_rows})\n  record={"level":float(level),"degenerate_face_count":int(len(bad)),"witnesses":witness,\n    "policy":"diagnostic only: no coordinate welding or face removal; exact emitted-F32 coincidences require identity/ancestry and closed-manifold proof before any quotient"}\n  globals()["last_f32_degenerate_level"] = record\n  raise ValueError("F32 degenerate level faces "+str(int(len(bad)))+"; witness_classifications="+repr([x["classification"] for x in witness]))\n'
 if target is None:raise RuntimeError("degeneracy diagnostic anchor missing")
 exec(compile(original.replace(target,replacement),fn.__code__.co_filename,"exec"),lib["ns"])
# exact-level extractor below supersedes the prior diagnostic helper

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_level_p1_280 import extract_level_exact
lib["ns"]["extract_level"] = extract_level_exact
CACHE_PATH=E/"tissue-aware-commonfield-279/field-cache-left.npz"
CACHE_RECEIPT=E/"tissue-aware-commonfield-279/field-cache-left.json"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def write_poly(path,V,F,seed):
 with Path(path).open("x") as f:
  f.write(f"{len(V)} 3 0 1\n")
  for i,p in enumerate(V,1):f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 1\n")
  f.write(f"{len(F)} 1\n")
  for tri in F:f.write(f"1 0 1\n3 {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")
  f.write("0\n1\n1 {0:.17g} {1:.17g} {2:.17g} 1 {3:.17g}\n".format(*seed,MAXVOL))
def face_inc(T):
 inc=defaultdict(list)
 for ti,t in enumerate(np.asarray(T,np.int64)):
  a,b,c,d=map(int,t)
  for q in ((a,b,c),(a,d,b),(b,d,c),(c,d,a)):
   inc[tuple(sorted(q))].append((q,ti))
 return inc
def _orient2(a,b,c):
 return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def _onseg2(a,b,p):
 return _orient2(a,b,p)==0 and min(a[0],b[0])<=p[0]<=max(a[0],b[0]) and min(a[1],b[1])<=p[1]<=max(a[1],b[1])
def _seg_touch2(a,b,c,d):
 o1,o2,o3,o4=_orient2(a,b,c),_orient2(a,b,d),_orient2(c,d,a),_orient2(c,d,b)
 if o1==0 and _onseg2(a,b,c):return True
 if o2==0 and _onseg2(a,b,d):return True
 if o3==0 and _onseg2(c,d,a):return True
 if o4==0 and _onseg2(c,d,b):return True
 return (o1>0)!=(o2>0) and (o3>0)!=(o4>0)
def _loop2(points):
 return [(Fraction.from_float(float(p[0])),Fraction.from_float(float(p[1]))) for p in np.asarray(points,float)]
def _loop_self_cross(poly):
 n=len(poly);hits=[]
 for i in range(n):
  a,b=poly[i],poly[(i+1)%n]
  for j in range(i+1,n):
   if j==(i+1)%n or (j+1)%n==i:continue
   if _seg_touch2(a,b,poly[j],poly[(j+1)%n]):hits.append((i,j))
 return hits
def _loops_cross(a,b):
 return [(i,j) for i in range(len(a)) for j in range(len(b)) if _seg_touch2(a[i],a[(i+1)%len(a)],b[j],b[(j+1)%len(b)])]
def _inside_loop(p,poly):
 inside=False
 for i,a in enumerate(poly):
  b=poly[(i+1)%len(poly)]
  if _onseg2(a,b,p):return None
  if (a[1]>p[1]) != (b[1]>p[1]):
   x=a[0]+(p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
   if p[0]<x:inside=not inside
 return inside
def _area2(poly):
 return sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))
def shell_boundary_preflight(side,V,F,Fp,Fc,raw):
 V=np.asarray(V,np.float32);F=np.asarray(F,np.int64)
 row_by_face={tuple(sorted(map(int,t))):i for i,t in enumerate(F)}
 outer_rows=[row_by_face[tuple(sorted(map(int,t)))] for t in Fp]
 cap_rows=[row_by_face[tuple(sorted(map(int,t)))] for t in Fc]
 all_records=lib["records"](V,F)
 cap_self=lib["_audit_self_pairs_touching_faces"](all_records,cap_rows)
 outer_rec=[all_records[i] for i in outer_rows];cap_rec=[all_records[i] for i in cap_rows]
 outer_idx=lib["_prepare_surface_aabb"](outer_rec)
 def cross_candidates():
  for cr in cap_rec:
   yield from lib["_query_surface_aabb"]([cr],outer_idx)
 outer_cap=_audit_record_pairs(cross_candidates(),same_surface=True)
 n=np.asarray(raw["plane_normal_actual007"],float);axis=int(np.argmax(np.abs(n)));keep=[i for i in range(3) if i!=axis]
 d=float(np.dot(n,np.asarray(raw["plane_origin_actual007_m"],float))+float(raw["plane_offset_m"]))
 rings={k:np.asarray(raw[k],np.int64) for k in ("outer_ring","skin_inner_cut_ring","fat_deep_cut_ring")}
 projected={name:_loop2(V[ids][:,keep]) for name,ids in rings.items()}
 plane_res={name:float(np.max(np.abs(V[ids].astype(float)@n-d))) for name,ids in rings.items()}
 simple={name:len(_loop_self_cross(poly))==0 for name,poly in projected.items()}
 pair_hits={};names=list(projected)
 for i,a in enumerate(names):
  for b in names[i+1:]:pair_hits[a+"_vs_"+b]=_loops_cross(projected[a],projected[b])
 nesting={"outer_contains_skin_inner":_inside_loop(projected["skin_inner_cut_ring"][0],projected["outer_ring"]),
          "skin_inner_contains_fat_deep":_inside_loop(projected["fat_deep_cut_ring"][0],projected["skin_inner_cut_ring"]),
          "outer_contains_fat_deep":_inside_loop(projected["fat_deep_cut_ring"][0],projected["outer_ring"])}
 loop_pass=all(simple.values()) and all(not v for v in pair_hits.values()) and nesting=={"outer_contains_skin_inner":True,"skin_inner_contains_fat_deep":True,"outer_contains_fat_deep":True} and max(plane_res.values())<=2e-5
 cand=np.load(CAND,allow_pickle=False);cr=np.asarray(raw["candidate271_face_rows"],np.int64)
 candV=np.asarray(cand["outer_world_actual007_m"],np.float32);candF=np.asarray(cand["surface_faces"],np.int64)[cr]
 signs=np.zeros(len(Fp),np.int8);perm_parity={(0,1,2):1,(0,2,1):-1,(1,0,2):-1,(1,2,0):1,(2,0,1):1,(2,1,0):-1}
 for i,(sf,cf) in enumerate(zip(Fp,candF)):
  a=V[sf];b=candV[cf]
  for perm,sgn in perm_parity.items():
   if np.array_equal(a,b[list(perm)]):signs[i]=sgn;break
 winding_path=OUT/f"shell-{side}-physical-winding-map.npz"
 np.savez_compressed(winding_path,candidate271_face_rows=cr,physical_source_face_ids=raw["physical_source_face_ids"],winding_sign_vs_candidate271=signs)
 winding={"same_order":int(np.sum(signs==1)),"reversed_order":int(np.sum(signs==-1)),"unmatched":int(np.sum(signs==0)),"map_path":str(winding_path),"map_sha256":sha(winding_path)}
 report={"side":side,"exact_predicate":"cardiac_cavity_intersections exact rational triangle intersections; same-index shared edge/vertex contacts are allowed only when all intersection points lie on that shared feature",
  "cap_self":cap_self,"physical_outer_vs_cap":outer_cap,
  "cross_section_loops":{"projection_dropped_axis":axis,"max_plane_residual_m":plane_res,
   "signed_double_area_m2":{"outer":float(_area2(projected["outer_ring"])),"skin_inner":float(_area2(projected["skin_inner_cut_ring"])),"fat_deep":float(_area2(projected["fat_deep_cut_ring"]))},
   "simple":simple,"pair_segment_intersections":{k:len(v) for k,v in pair_hits.items()},"nesting":nesting,"pass":bool(loop_pass)},
  "physical_outer_winding_vs_candidate271":winding,
  "pass":bool(cap_self["count"]==0 and outer_cap["count"]==0 and loop_pass and winding["unmatched"]==0)}
 p=OUT/f"shell-{side}-exact-preflight.json";p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 if not report["pass"]:raise ValueError((side,"cap/outer/loop exact preflight failed",report))
 return report
def shell_mesh(side):
 if side != "left":raise ValueError("bounded field run is left-first")
 shell=np.load(SHELLDIR/f"{side}-source-shell.npz",allow_pickle=False)
 srcV=np.asarray(shell["vertices_world_m"],np.float32)
 F0=np.asarray(shell["faces"],np.int64)
 used=np.unique(F0)
 remap=np.full(len(srcV),-1,np.int64);remap[used]=np.arange(len(used))
 F=remap[F0]
 OV=srcV[used].astype(np.float32).astype(np.float64)
 OF=remap[np.asarray(shell["physical_faces"],np.int64)]
 Fc=remap[np.asarray(shell["cap_faces"],np.int64)]
 hist=lib["face_edge_hist"](F)
 vol=lib["signed_volume"](OV,F)
 if hist!={2:sum(hist.values())} or vol<=0:
  raise ValueError(f"{side} shell not closed/outward: {hist}, volume={vol}")
 center,cv=lib["ns"]["volume_centroid"](OV,F)
 if abs(cv-vol)>max(1e-12,1e-9*vol):
  raise ValueError(f"{side} shell volume/centroid mismatch: {vol}, {cv}")
 cached=E/"tissue-aware-commonfield-272"
 expected={
  "outer-left.poly":"aacc7831ed72f361c222ea14ff92fb80ec7231fa54d5041f559eb96d508a208d",
  "outer-left.1.node":"2fd4d538d8943f68e7423bf5071c254d5aebbe5057b3f3c06fbfb76e61bec3b5",
  "outer-left.1.ele":"a07aba03882d91c309c24c64ae73d68441b7778be9a288e6c54340a2d6736fa1",
  "outer-left.1.face":"ec83d772240450ea42cfa21e19a3d0ec6b98db3cbd8268d5f24a317f9f6ca602",
  "outer-left.stdout":"f85fb4797b6093dc2360c562ee502ba631ea88ddbfe94dad61957f56b454f8b8"
 }
 cache_hashes={name:sha(cached/name) for name in expected}
 if cache_hashes!=expected:raise ValueError(("immutable 272 mesh pin mismatch",cache_hashes))
 Xraw,T=lib["ns"]["parse_tetgen"](cached/"outer-left")
 d0,_=lib["tet_metrics"](Xraw,T)
 if np.any(d0==0):raise ValueError("cached outer mesh has zero signed tetra")
 T=T.copy();neg=int(np.sum(d0<0))
 if neg:T[d0<0]=T[d0<0][:,[0,1,3,2]]
 X32=Xraw.astype(np.float32)
 coord={tuple(float(x) for x in p):i for i,p in enumerate(Xraw)}
 bmap=np.empty(len(OV),np.int64);dist=np.empty(len(OV),float)
 tree=cKDTree(Xraw)
 for i,p in enumerate(OV):
  key=tuple(float(x) for x in p)
  j=coord.get(key)
  if j is None:
   dd,ix=tree.query(p,k=1);j=int(ix);dist[i]=float(dd)
  else:dist[i]=float(np.linalg.norm(Xraw[j]-p))
  bmap[i]=j
 if np.max(dist)>1e-7 or len(np.unique(bmap))!=len(OV):
  raise ValueError(("cached mesh boundary mapping failed",float(np.max(dist)),len(np.unique(bmap)),len(OV)))
 X32[bmap]=OV.astype(np.float32)
 d64,q64=lib["tet_metrics"](X32.astype(np.float64),T)
 d32,q32=lib["tet_metrics"](X32,T)
 if np.any(d64<=0) or np.any(d32<=0) or float(q32.min())<1e-5:
  raise ValueError(("cached mesh fails actual007 Float32 determinant/quality",int(np.sum(d32<=0)),float(d32.min()),float(q32.min())))
 inc=face_inc(T);boundary=[]
 for uses in inc.values():
  if len(uses)==1:boundary.append(tuple(sorted(uses[0][0])))
 expected_boundary=[tuple(sorted(map(int,bmap[t]))) for t in F]
 if Counter(boundary)!=Counter(expected_boundary):
  raise ValueError(("cached TetGen connectivity does not preserve exact shell",len(boundary),len(expected_boundary)))
 def mapped(key):
  ids=np.asarray(shell[key],np.int64)
  if np.any(ids<0) or np.any(ids>=len(remap)) or np.any(remap[ids]<0):
   raise ValueError(("shell role map includes unused source row",key))
  return bmap[remap[ids]]
 info={"mesh_nodes":int(len(X32)),"mesh_tets":int(len(T)),"fixed_boundary_vertices":int(len(OV)),
  "boundary_max_rebind_m":float(dist.max()),"boundary_faces_expected":int(len(F)),"boundary_faces_actual":int(len(boundary)),
  "input_closed_shell_volume_m3":float(vol),"tet_signed_volume_sum_m3":float(d64.sum()),
  "negative_f64_before_reorientation":neg,"min_f64_signed_det_m3":float(d64.min()),
  "min_f32_signed_det_m3":float(d32.min()),"min_f32_normalized_abs_det":float(q32.min()),
  "f32_nonpositive_tets":int(np.sum(d32<=0)),"tetgen_elapsed_s":0.0,"tetgen_reused":True,
  "cached_mesh_directory":str(cached),"cached_mesh_sha256":cache_hashes,
  "outer_shell_volume_m3":float(vol)}
 meta={"V":X32,"T":T,"OV":X32,"OF":bmap[OF],"capF":bmap[Fc],
  "outer_nodes":np.unique(bmap[OF]),"o_ring":mapped("outer_ring"),"i_ring":mapped("skin_inner_cut_ring"),
  "d_ring":mapped("fat_deep_cut_ring"),"center":int(mapped("cut_disk_center")[0]),
  "source_face_ids":shell["physical_source_face_ids"],"source_face_rows":shell["candidate271_face_rows"],
  "cap_role":shell["cap_face_role"],"shell":shell}
 return meta,info

def select_banks(side,outer_points):
 inv=json.loads(INV.read_text());idoc=json.loads((INV.parent/"inventory.json").read_text())
 if idoc["pose_sha256"]!=sha(POSE):raise ValueError("registered tissue bank pose differs from native007")
 lo=np.asarray(outer_points).min(0)-1e-6;hi=np.asarray(outer_points).max(0)+1e-6
 z=np.load(BANK,allow_pickle=False);out=[]
 for i,m in enumerate(inv["banks"]):
  if m.get("kind") not in ("anatomy","soft_tissue"):continue
  mn=np.asarray(m["aabb_min_m"],float);mx=np.asarray(m["aabb_max_m"],float)
  if np.all(mx>=lo)&np.all(mn<=hi):
   V=z[f"bank_{i:04d}_vertices"].astype(np.float32);F=z[f"bank_{i:04d}_faces"].astype(np.int64)
   closed,top=lib["consistent_closed"](F);top["closed_for_winding"]=bool(closed)
   out.append((m,V,F,top))
 print(json.dumps({"stage":"nearby_banks","side":side,"banks":len(out),"counts_by_kind":dict(Counter(m["kind"] for m,V,F,top in out)),
  "triangles_by_kind":{k:int(sum(len(F) for m,V,F,top in out if m["kind"]==k)) for k in ("anatomy","soft_tissue")},
  "open_soft_groups":[list(g) for g,items in group_banks(out).items() if items[0][3]["closed_for_winding"] is False]},sort_keys=True),flush=True)
 return out
def group_banks(banks):
 g=defaultdict(list)
 for it in banks:g[tuple(sorted(map(int,it[0]["body_indices"])))].append(it)
 return g
def obstacle_field(X,outerV,outerF,banks,outer_ids):
 # The nearest bank set contains both anatomy and soft-tissue/fascia surfaces.
 VV=[];FF=[];face_kind=[];face_owner=[];off=0
 for m,V,F,top in banks:
  VV.append(V);FF.append(F+off);face_kind.extend([m["kind"]]*len(F));face_owner.extend([m["key"]]*len(F));off+=len(V)
 BV=np.vstack(VV);BF=np.vstack(FF);face_kind=np.asarray(face_kind,dtype=object);face_owner=np.asarray(face_owner,dtype=object)
 db,bfi,bbary,bq=lib["closest_surface"](X,BV,BF,k=64);douter,ofi,obary,oq=lib["closest_surface"](X,outerV,outerF,k=48)
 inside=np.zeros(len(X),bool);wind=[]
 for body_ids,items in group_banks(banks).items():
  Vparts=[];Fparts=[];off=0
  for m,V,F,top in items:
   Vparts.append(np.asarray(V,np.float32));Fparts.append(np.asarray(F,np.int64)+off);off+=len(V)
  Vcat=np.vstack(Vparts);Fcat=np.vstack(Fparts);Vg,invx=np.unique(Vcat,axis=0,return_inverse=True);Fg=invx[Fcat]
  closed,top=lib["consistent_closed"](Fg)
  if not closed:
   wind.append({"body_indices":list(body_ids),"bank_keys":[x[0]["key"] for x in items],"kind_counts":dict(Counter(x[0]["kind"] for x in items)),"closed_union":False,"topology":top,"inside_points":0})
   continue
  mn=Vg.min(0).astype(float)-1e-6;mx=Vg.max(0).astype(float)+1e-6
  ids=np.flatnonzero(np.all(np.asarray(X)>=mn,axis=1)&np.all(np.asarray(X)<=mx,axis=1))
  if len(ids):
   w=lib["solid_angle_winding"](X[ids],Vg,Fg);hit=np.abs(w)>0.5;inside[ids[hit]]=True
   maxw=float(np.max(np.abs(w)))
  else:hit=np.zeros(0,bool);maxw=0.0
  wind.append({"body_indices":list(body_ids),"bank_keys":[x[0]["key"] for x in items],"kind_counts":dict(Counter(x[0]["kind"] for x in items)),"closed_union":True,"topology":top,"tested_points":int(len(ids)),"inside_points":int(hit.sum()),"max_abs_winding":maxw})
 # Closed muscle/tendon unions are interior obstacles; every bank surface is also a distance/exact constraint.
 kinds=face_kind[bfi];margin=np.where(kinds=="anatomy",OBSTACLE_MARGIN_BONE_M,OBSTACLE_MARGIN_SOFT_M)
 near=db<=margin
 boundary_inside=int(inside[np.asarray(outer_ids,np.int64)].sum())
 phi=np.where(inside|near,1.0,douter/np.maximum(douter+np.maximum(db,1e-9),1e-9))
 phi=np.clip(phi,0,1)
 # Preserve the actual outer boundary value; a source O touching/inside an obstacle is reported as a conflict.
 phi[np.asarray(outer_ids,np.int64)]=0.0
 report={"classifier":"nearest distance to all overlapping registered anatomy and soft-tissue banks; closed per-body unions use exact-coordinate-welded solid-angle winding; open tendon groups have unsigned distance plus exact triangle gates",
  "obstacle_margin_bone_m":OBSTACLE_MARGIN_BONE_M,"obstacle_margin_soft_m":OBSTACLE_MARGIN_SOFT_M,
  "inside_closed_group_nodes":int(inside.sum()),"inside_group_outer_boundary_nodes":boundary_inside,
  "near_any_bank_nodes":int(near.sum()),"nearest_bank_kind_counts":dict(Counter(map(str,kinds))),
  "nearest_bank_key_counts":dict(Counter(map(str,face_owner[bfi]))),
  "nearest_bank_distance_mm_quantiles":(db*1000).astype(float).tolist() if len(db)<0 else np.quantile(db*1000,[0,.01,.05,.5,.95,.99,1]).tolist(),
  "outer_distance_mm_quantiles":np.quantile(douter*1000,[0,.01,.5,.99,1]).tolist(),"winding_by_group":wind,
  "nearest_bank_face_ids":bfi,"nearest_bank_points":bq,"nearest_bank_distances_m":db,
  "outer_nearest_faces":ofi,"outer_nearest_points":oq,"outer_nearest_distances_m":douter,
  "all_bank_vertices":BV,"all_bank_faces":BF,"all_bank_face_kind":face_kind,"all_bank_face_owner":face_owner}
 return phi,report
def flux(points,F):
 P=np.asarray(points,np.float64);F=np.asarray(F,np.int64)
 if not len(F):return 0.0
 a=P[F[:,0]];b=P[F[:,1]];c=P[F[:,2]]
 return float(np.einsum("ij,ij->i",a,np.cross(b,c)).sum()/6.0)
def analytic_nested_cube_preflight():
 def cube(side):
  h=float(side)*0.5
  V=np.asarray([[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],[-h,-h,h],[h,-h,h],[h,h,h],[-h,h,h]],np.float64)
  F=np.asarray([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]],np.int64)
  return V,F
 sides=(0.200,0.180,0.160)
 meshes=[cube(x) for x in sides]
 volumes=[flux(V,F) for V,F in meshes]
 R=np.asarray([[0,-1,0],[1,0,0],[0,0,1]],np.float64);shift=np.asarray([0.37,-0.12,0.80])
 moved=meshes[0][0]@R.T+shift
 moved_volume=flux(moved,meshes[0][1])
 outer,inner,deep=volumes
 skin=outer-inner;fat=inner-deep;core=deep
 # Synthetic test densities exercise arithmetic only; they are not donor/material inputs.
 rho=np.asarray([1.1,0.9,1.3],np.float64)
 mass=float(rho[0]*skin+rho[1]*fat+rho[2]*core)
 checks={"cube_volumes_match_analytic":all(abs(v-s**3)<=1e-14 for v,s in zip(volumes,sides)),
  "rigid_world_transform_preserves_volume":abs(moved_volume-outer)<=1e-14,
  "nested_partition_closes":abs((skin+fat+core)-outer)<=1e-14,
  "toy_mass_reconstructs":abs(mass-float(np.dot(rho,np.asarray([skin,fat,core]))))<=1e-16}
 if not all(checks.values()):raise AssertionError(("analytic nested cube preflight failed",checks,volumes,moved_volume))
 return {"checks":checks,"side_lengths_m":list(sides),"volumes_m3":volumes,
  "rigid_transform":{"rotation":R.tolist(),"translation_m":shift.tolist(),"transformed_outer_volume_m3":moved_volume},
  "partition_volumes_m3":{"skin":skin,"fat":fat,"core":core,"sum":skin+fat+core},
  "synthetic_test_densities_kg_m3":rho.tolist(),"synthetic_test_mass_kg":mass,
  "not_donor_density_or_material_mass_evidence":True}

def clip_poly(poly,level,keep_less):
 if not poly:return []
 out=[]
 def inside(v):return v<=level if keep_less else v>=level
 for i,p in enumerate(poly):
  q=poly[(i+1)%len(poly)];ip=inside(p[1]);iq=inside(q[1])
  if ip and iq:out.append(q)
  elif ip and not iq:
   t=(level-p[1])/(q[1]-p[1]);out.append(((1-t)*p[0]+t*q[0],float(level)))
  elif not ip and iq:
   t=(level-p[1])/(q[1]-p[1]);out.append(((1-t)*p[0]+t*q[0],float(level)));out.append(q)
 # remove consecutive coincident vertices introduced by clipping
 clean=[]
 for x in out:
  if not clean or np.linalg.norm(np.asarray(x[0])-np.asarray(clean[-1][0]))>1e-12:clean.append(x)
 if len(clean)>1 and np.linalg.norm(np.asarray(clean[0][0])-np.asarray(clean[-1][0]))<=1e-12:clean.pop()
 return clean
def cap_region(capF,X,phi,low=None,high=None):
 tris=[]
 for t in capF:
  poly=[(np.asarray(X[int(v)],np.float64),float(phi[int(v)])) for v in t]
  if low is not None:poly=clip_poly(poly,float(low),False)
  if high is not None:poly=clip_poly(poly,float(high),True)
  if len(poly)<3:continue
  for j in range(1,len(poly)-1):
   tri=np.asarray([poly[0][0],poly[j][0],poly[j+1][0]],np.float32)
   if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))>0:tris.append(tri)
 return np.asarray(tris,np.float32).reshape(-1,3,3)
def flux_triangles(Tri):
 if not len(Tri):return 0.0
 a=Tri[:,0].astype(float);b=Tri[:,1].astype(float);c=Tri[:,2].astype(float)
 return float(np.einsum("ij,ij->i",a,np.cross(b,c)).sum()/6.0)
def cap_vol(P):
 if len(P)==0:return 0.0
 return flux_triangles(P)
def safe_levels(phi,excluded,lo,hi):
 u=np.unique(np.asarray(phi,float)[~np.isin(np.arange(len(phi)),np.asarray(excluded,np.int64))])
 out=[]
 for a,b in zip(u[:-1],u[1:]):
  if b-a>1e-8:
   m=float((a+b)*.5)
   if lo<m<hi:out.append((m,float(b-a)))
 return out
def choose_boundary_level(proposed,intervals,lo,hi,used):
 cand=[(abs(m-proposed),-w,m) for m,w in intervals if lo<m<hi and all(abs(m-x)>1e-14 for x in used)]
 if not cand:raise ValueError(("no safe scalar interval in level bracket",proposed,lo,hi))
 cand.sort();return cand[0][2]
def boundary_report(S,ring,V,n,d):
 F=S["faces"];P=S["points"];inc=defaultdict(int)
 for a,b,c in F:
  for u,v in ((a,b),(b,c),(c,a)):inc[tuple(sorted((int(u),int(v))))]+=1
 be=[e for e,k in inc.items() if k==1];adj=defaultdict(set)
 for a,b in be:adj[a].add(b);adj[b].add(a)
 deg=Counter(map(len,adj.values()));comp=0;seen=set()
 for v in adj:
  if v in seen:continue
  comp+=1;st=[v];seen.add(v)
  while st:
   u=st.pop()
   for w in adj[u]:
    if w not in seen:seen.add(w);st.append(w)
 ids=np.unique(np.asarray(be,np.int64)) if be else np.zeros(0,np.int64)
 p=P[ids].astype(float)
 plane=float(np.max(np.abs(p@np.asarray(n,float)-float(d)))) if len(p) else None
 # Exact nearest distance to the retained cut curve segments.
 rv=np.asarray(V[np.asarray(ring,np.int64)],float);ds=[]
 for x in p:
  best=float("inf")
  for a,b in zip(rv,np.roll(rv,-1,axis=0)):
   ab=b-a;den=float(ab@ab);t=0 if den<=1e-30 else float(np.clip((x-a)@ab/den,0,1))
   best=min(best,float(np.linalg.norm(x-(a+t*ab))))
  ds.append(best)
 return {"boundary_edges":len(be),"boundary_vertices":int(len(ids)),"boundary_components":int(comp),"boundary_degree_histogram":dict(deg),
  "boundary_plane_residual_max_m":plane,"max_distance_to_inherited_cut_ring_m":float(max(ds)) if ds else None,
  "all_boundary_vertices_on_cap":bool(len(be)>0 and max(ds)<2e-5 and (plane or 0)<2e-5)}
def choose_extractable_level(V,T,proposed,intervals,lo,hi,used,field_factory,max_skips=24):
 failures=[];failure_count=0
 for _ in range(max_skips):
  level=choose_boundary_level(proposed,intervals,lo,hi,used)
  used.append(level)
  phi=field_factory(level)
  try:
   surface=lib["ns"]["extract_level"](V,T,phi,level)
  except ValueError as exc:
   msg=str(exc)
   if "degenerate" in msg or "empty level" in msg:
    failure_count+=1
    if len(failures)<2:failures.append({"level":float(level),"error":msg[:6000]})
    witness=lib["ns"].get("last_f32_degenerate_level")
    witness_path=OUT/"first-degenerate-level-witness.json"
    if witness is not None and not witness_path.exists():
     witness_path.write_text(json.dumps(witness,indent=2,sort_keys=True)+"\n")
    continue
   raise
  return float(level),surface,phi,{"rejected_level_count":failure_count,"witness_examples":failures}
 raise ValueError(("no Float32-admissible level in bounded bracket",proposed,lo,hi,{"rejected_level_count":failure_count,"witness_examples":failures}))
def helper_api_preflight():
 required_outer=["records","_audit_self_pairs_touching_faces","_prepare_surface_aabb","_query_surface_aabb","closest_surface","consistent_closed","exact_surface_gates","face_edge_hist","signed_volume","solid_angle_winding","tet_metrics"]
 required_inner=["parse_tetgen","extract_level","volume_centroid"]
 signatures={}
 for name in required_outer:
  fn=lib.get(name)
  if not callable(fn):raise TypeError(f"required helper is unavailable: lib.{name}")
  signatures[f"lib.{name}"]=str(inspect.signature(fn))
 for name in required_inner:
  fn=lib.get("ns",{}).get(name)
  if not callable(fn):raise TypeError(f"required helper is unavailable: lib.ns.{name}")
  signatures[f"lib.ns.{name}"]=str(inspect.signature(fn))
 signatures["exact_level_p1_280.extract_level_exact"]=str(inspect.signature(extract_level_exact))
 return {"all_required_callables":True,"signature_count":len(signatures),"signatures":signatures,"exact_level_policy":"strict sublevel; exact global nodes and strict global edges; no ring epsilon",
         "toy_pipeline_receipt_sha256":sha(Path("/tmp/field-helper-preflight-275.json"))}


def select_banks_cached(side, outer_points):
 if side != "left": raise ValueError("bounded cached followup is left-first")
 z=np.load(CACHE_PATH,allow_pickle=False);recs=json.loads(str(z["bank_metadata_json"][0]))
 vo=np.asarray(z["bank_vertex_offsets"],np.int64);fo=np.asarray(z["bank_face_offsets"],np.int64)
 allv=np.asarray(z["all_bank_vertices_world_m"],np.float32);allf=np.asarray(z["all_bank_faces"],np.int64);ans=[]
 for i,r in enumerate(recs):
  m=r["meta"];top=r["topology"];v0,v1=map(int,vo[i:i+2]);f0,f1=map(int,fo[i:i+2])
  V=allv[v0:v1].copy();F=allf[f0:f1].copy()-v0
  if len(V)!=int(m["vertices"]) or len(F)!=int(m["triangles"]): raise ValueError(("cached bank slice mismatch",i,m["key"]))
  ans.append((m,V,F,top))
 return ans

def obstacle_field_cached(X, T, outerV, outerF, banks, outer_ids):
 z=np.load(CACHE_PATH,allow_pickle=False)
 if not np.array_equal(np.asarray(X,np.float32),z["field_points_world_m"]): raise ValueError("cached field points differ from pinned shell mesh")
 if not np.array_equal(np.asarray(outerF,np.int64),z["physical_outer_faces"]): raise ValueError("cached physical O face rows differ")
 if not np.array_equal(np.asarray(T,np.int64),z["field_tetrahedra"]): raise ValueError("cached tet rows differ from pinned shell mesh")
 if len(banks)!=len(json.loads(str(z["bank_metadata_json"][0]))): raise ValueError("cached bank list mismatch")
 fd=json.loads(str(z["field_report_json"][0]))
 fd.update({"nearest_bank_face_ids":z["nearest_bank_face_ids"],"nearest_bank_points":z["nearest_bank_points_world_m"],"nearest_bank_distances_m":z["nearest_bank_distances_m"],
  "outer_nearest_faces":z["outer_nearest_face_ids"],"outer_nearest_points":z["outer_nearest_points_world_m"],"outer_nearest_distances_m":z["outer_nearest_distances_m"],
  "all_bank_vertices":z["all_bank_vertices_world_m"],"all_bank_faces":z["all_bank_faces"],"all_bank_face_kind":z["all_bank_face_kind"],"all_bank_face_owner":z["all_bank_face_owner"]})
 return np.asarray(z["phi0"],np.float64),fd

def main():
 t0=time.monotonic()
 api=helper_api_preflight()
 toy=analytic_nested_cube_preflight()
 (OUT/"build_field_280.py").write_text(Path(__file__).read_text())
 candidate=np.load(CAND,allow_pickle=False);pose_sha=sha(POSE);bankmeta=json.loads(INV.read_text())
 result={"schema":"numi.human.tissue-aware-independent-layer-commonfield-280.v1","status":"unadmitted_geometry_candidate",
  "method":"Actual007 bilateral source-collar outer skin plus independently inherited O/I/D cut traces; one P1 field uses all overlapping anatomy and soft-tissue/fascia banks as exclusion surfaces, with closed-solid winding and exact open-bank triangle gates; cut-cap scalar traces preserve the inherited inner/deep interface rings.",
  "helper_api_preflight":api,"analytic_nested_cube_preflight":toy,"input_sha256":{"cached_outer_mesh_node":sha(E/"tissue-aware-commonfield-272/outer-left.1.node"),"cached_outer_mesh_ele":sha(E/"tissue-aware-commonfield-272/outer-left.1.ele"),"candidate271":sha(CAND),"shell_preflight":sha(SHELLDIR/"shell-preflight.json"),"registered_banks":sha(BANK),"bank_inventory":sha(INV),"native007_pose":pose_sha,"native007_node_dump":sha(E/"native-app-main-header-integration-151/source-preflight-007/initial-skin-fem-nodes.csv"),"tetgen":sha(TETGEN),"cached_field_279":sha(CACHE_PATH),"exact_level_extractor":sha(Path(__file__).resolve().with_name("exact_level_p1_280.py"))},
  "limitations":["outer-shell TetGen and level surfaces are construction evidence only; no app/native admission","regional geometric volumes are compared with baseline targets, but conservative parent-tet overlap CSR and donor mass mapping are not yet computed","the field margin is explicitly 0.25 mm to bone and 0.10 mm to soft-tissue surfaces; exact triangle gates remain authoritative","open 32-edge tendon union is included as an unsigned-distance/exact surface constraint, never assigned a winding side"], "sides":{}}
 for side in ("left",):
  print(json.dumps({"stage":"outer_mesh","side":side},sort_keys=True),flush=True)
  m,mesh=shell_mesh(side);banks=select_banks_cached(side,m["V"][np.unique(m["OF"])])
  field_t0=time.monotonic()
  phi0,fd=obstacle_field_cached(m["V"],m["T"],m["V"],m["OF"],banks,m["outer_nodes"])
  bank_vo=[0];bank_fo=[0]
  for bm,bv,bf,bt in banks:
   bank_vo.append(bank_vo[-1]+len(bv));bank_fo.append(bank_fo[-1]+len(bf))
  cache=OUT/f"field-cache-{side}.npz"
  bankmeta=[{"meta":bm,"topology":bt} for bm,bv,bf,bt in banks]
  fdscalars={k:v for k,v in fd.items() if not isinstance(v,np.ndarray)}
  def json_default(v):
   if isinstance(v,np.ndarray):return v.tolist()
   if isinstance(v,np.integer):return int(v)
   if isinstance(v,np.floating):return float(v)
   if isinstance(v,np.bool_):return bool(v)
   return str(v)
  np.savez_compressed(cache,field_points_world_m=m["V"],field_tetrahedra=m["T"],phi0=phi0,
   physical_outer_faces=m["OF"],temporary_cap_faces=m["capF"],outer_boundary_node_ids=m["outer_nodes"],
   outer_ring_node_ids=m["o_ring"],skin_inner_cut_ring_node_ids=m["i_ring"],fat_deep_cut_ring_node_ids=m["d_ring"],cut_disk_center_node_id=np.asarray([m["center"]]),
   physical_source_face_ids=m["source_face_ids"],candidate271_face_rows=m["source_face_rows"],cap_face_role=m["cap_role"],
   nearest_bank_face_ids=fd["nearest_bank_face_ids"],nearest_bank_points_world_m=fd["nearest_bank_points"],nearest_bank_distances_m=fd["nearest_bank_distances_m"],
   outer_nearest_face_ids=fd["outer_nearest_faces"],outer_nearest_points_world_m=fd["outer_nearest_points"],outer_nearest_distances_m=fd["outer_nearest_distances_m"],
   all_bank_vertices_world_m=fd["all_bank_vertices"],all_bank_faces=fd["all_bank_faces"],all_bank_face_kind=np.asarray(fd["all_bank_face_kind"],dtype=str),all_bank_face_owner=np.asarray(fd["all_bank_face_owner"],dtype=str),
   bank_vertex_offsets=np.asarray(bank_vo,np.int64),bank_face_offsets=np.asarray(bank_fo,np.int64),bank_metadata_json=np.asarray([json.dumps(bankmeta,default=json_default)]),field_report_json=np.asarray([json.dumps(fdscalars,default=json_default)]))
  cache_receipt={"path":str(cache),"sha256":sha(cache),"bytes":cache.stat().st_size,"field_elapsed_s":time.monotonic()-field_t0,
   "source_pose_sha256":pose_sha,"shell_mesh_node_sha256":sha(E/"tissue-aware-commonfield-272/outer-left.1.node"),
   "shell_mesh_ele_sha256":sha(E/"tissue-aware-commonfield-272/outer-left.1.ele"),"bank_sha256":sha(BANK),"inventory_sha256":sha(INV),
   "field_points":int(len(m["V"])),"field_tets":int(len(m["T"])),"physical_outer_faces":int(len(m["OF"])),"temporary_cap_faces":int(len(m["capF"])),
   "bank_count":len(banks),"bank_triangles":int(sum(len(x[2]) for x in banks)),"role_arrays_included":True,
   "fd_array_names":sorted(k for k,v in fd.items() if isinstance(v,np.ndarray))}
  (OUT/f"field-cache-{side}.json").write_text(json.dumps(cache_receipt,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"stage":"field_cache_checkpoint","side":side,**cache_receipt},sort_keys=True),flush=True)
  if fd["inside_group_outer_boundary_nodes"]:
   raise ValueError((side,"physical O boundary is inside a closed anatomy/soft-tissue bank",fd["inside_group_outer_boundary_nodes"]))
  T=m["T"];X=m["V"];Fphys=m["OF"];Fc=m["capF"];outervol=mesh["input_closed_shell_volume_m3"]
  caps=np.unique(np.r_[m["o_ring"],m["i_ring"],m["d_ring"],[m["center"]],m["outer_nodes"]])
  # Cap/O boundary ring values are explicitly tied to the inherited layer traces.
  intervals=safe_levels(phi0,caps,1e-6,.95)
  if not intervals:raise ValueError((side,"no nondegenerate P1 scalar intervals"))
  Ovol=flux(X,Fphys)
  targetSkin=TARGET[side]["skin"];targetFat=TARGET[side]["fat"];targetCore=outervol-targetSkin-targetFat
  def field_at(lI,lD):
   ph=phi0.copy();ph[m["outer_nodes"]]=0.0
   ph[m["o_ring"]]=0.0;ph[m["i_ring"]]=float(lI);ph[m["d_ring"]]=float(lD);ph[m["center"]]=1.0
   return ph
  # Solve skin volume first; its inherited cut ring is exact-level valued.
  used=[];iterationsI=[];bestI=None;lo=1e-5;hi=.8;fixedD=.95
  for it in range(12):
   proposed=(lo+hi)*.5
   level,surf,ph,skipped=choose_extractable_level(X,T,proposed,intervals,lo,hi,used,lambda value:field_at(value,fixedD))
   lowcap=cap_region(Fc,X,ph,high=level)
   vskin=Ovol+flux(surf["points"],surf["faces"])+cap_vol(lowcap)
   rec={"iteration":it,"proposed":proposed,"level":level,"skin_volume_m3":vskin,"residual_m3":vskin-targetSkin,"cap_triangles":len(lowcap),"skipped_F32_degenerate_levels":skipped}
   iterationsI.append(rec)
   if bestI is None or abs(vskin-targetSkin)<abs(bestI[0]-targetSkin):bestI=(vskin,level,surf,ph,lowcap)
   if vskin<targetSkin:lo=level
   else:hi=level
  vskin,lI,SI,phiI,capI=bestI
  # D is a separate interface of the same field; the muscle/fascia banks are
  # high-valued obstacles, so this contour stays on the skin-side boundary.
  usedD=[];iterationsD=[];bestD=None;lo=max(lI+1e-5,1e-5);hi=.95
  for it in range(12):
   proposed=(lo+hi)*.5
   level,surf,ph,skipped=choose_extractable_level(X,T,proposed,intervals,lo,hi,usedD,lambda value:field_at(lI,value))
   highcap=cap_region(Fc,X,ph,low=level)
   vcore=-flux(surf["points"],surf["faces"])+cap_vol(highcap)
   rec={"iteration":it,"proposed":proposed,"level":level,"core_volume_m3":vcore,"residual_m3":vcore-targetCore,"cap_triangles":len(highcap),"skipped_F32_degenerate_levels":skipped}
   iterationsD.append(rec)
   if bestD is None or abs(vcore-targetCore)<abs(bestD[0]-targetCore):bestD=(vcore,level,surf,ph,highcap)
   if vcore>targetCore:lo=level
   else:hi=level
  vcore,lD,SD,phiD,capD=bestD
  # Re-evaluate I with the final pair, then compute fat from the regional partition.
  phi=field_at(lI,lD)
  try:
   SI=lib["ns"]["extract_level"](X,T,phi,lI);SD=lib["ns"]["extract_level"](X,T,phi,lD)
  except ValueError as exc:
   raise ValueError((side,"final shared-field re-extraction failed after level search",repr(exc),"retain chosen-level surfaces; do not proceed to exact audit/second TetGen"))
  capI=cap_region(Fc,X,phi,high=lI);capD=cap_region(Fc,X,phi,low=lD)
  vskin=Ovol+flux(SI["points"],SI["faces"])+cap_vol(capI);vcore=-flux(SD["points"],SD["faces"])+cap_vol(capD);vfat=outervol-vskin-vcore
  # Source-face/bary ancestry for each independent extracted interface.
  for name,S in (("skin_inner",SI),("fat_deep",SD)):
   dist,fi,bary,qpt=lib["closest_surface"](S["points"],X,Fphys,k=64)
   S["source_shell_face_index"]=fi;S["source_face_id"]=m["source_face_ids"][fi]
   S["source_face_barycentric"]=bary;S["source_projection_world_m"]=qpt;S["source_projection_distance_m"]=dist
  gate=lib["exact_surface_gates"](X,Fphys,SI,SD,banks)
  bI=boundary_report(SI,m["i_ring"],X,m["shell"]["plane_normal_actual007"],float(m["shell"]["plane_origin_actual007_m"]@m["shell"]["plane_normal_actual007"]+m["shell"]["plane_offset_m"]))
  bD=boundary_report(SD,m["d_ring"],X,m["shell"]["plane_normal_actual007"],float(m["shell"]["plane_origin_actual007_m"]@m["shell"]["plane_normal_actual007"]+m["shell"]["plane_offset_m"]))
  open_counts={"skin_inner":dict(lib["face_edge_hist"](SI["faces"])),"fat_deep":dict(lib["face_edge_hist"](SD["faces"]))}
  geometry_closed=(bI["boundary_components"]==1 and bD["boundary_components"]==1 and bI["all_boundary_vertices_on_cap"] and bD["all_boundary_vertices_on_cap"] and all(all(int(k) in (1,2) for k in hist) and hist.get(1,0)>0 for hist in open_counts.values()))
  exact_pairs=sum(int(v.get("intersection_pairs",v.get("count",0))) for k,v in gate.items() if "_vs_" in k or k.endswith("_self"))
  deep_bank_d,deep_bank_f,_,deep_bank_q=lib["closest_surface"](SD["points"],fd["all_bank_vertices"],fd["all_bank_faces"],k=64)
  report={"side":side,"status":"finite-tissue-aware-open-interface-candidate-unadmitted","frame":"actual007 world; no rigid remapping after cap construction",
   "outer_shell":mesh,"nearby_banks":[{"key":m0["key"],"stable_id":m0["stable_id"],"kind":m0["kind"],"body_indices":m0["body_indices"],"triangles":len(F)} for m0,V0,F,_ in banks],
   "soft_tissue_obstacle_counts":{"banks":sum(m0["kind"]=="soft_tissue" for m0,V0,F,_ in banks),"closed_body_index_unions":sum(any(x[0]["kind"]=="soft_tissue" for x in xs) and xs[0][3]["closed_for_winding"] for xs in group_banks(banks).values()),
     "open_body_index_unions":[list(k) for k,xs in group_banks(banks).items() if any(x[0]["kind"]=="soft_tissue" for x in xs) and not xs[0][3]["closed_for_winding"]]},
   "obstacle_field":{k:v for k,v in fd.items() if not isinstance(v,np.ndarray)},
   "levels":{"skin_inner":{"phi":float(lI),"target_volume_m3":targetSkin,"realized_volume_m3":float(vskin),"delta_percent":100*(vskin/targetSkin-1),"search":iterationsI},
    "fat_deep":{"phi":float(lD),"target_volume_m3":targetFat,"realized_volume_m3":float(vfat),"delta_percent":100*(vfat/targetFat-1),"target_core_volume_m3":targetCore,"realized_core_volume_m3":float(vcore),"search":iterationsD}},
   "interfaces":{"skin_inner":{"vertices":len(SI["points"]),"faces":len(SI["faces"]),"edge_histogram":open_counts["skin_inner"],"boundary":bI},
    "fat_deep":{"vertices":len(SD["points"]),"faces":len(SD["faces"]),"edge_histogram":open_counts["fat_deep"],"boundary":bD},
    "O_to_I_distance_mm_quantiles":np.quantile(lib["closest_surface"](SI["points"],X,Fphys,k=64)[0]*1000,[0,.01,.5,.99,1]).tolist(),
    "I_to_D_distance_mm_quantiles":np.quantile(lib["closest_surface"](SD["points"],SI["points"],np.array([[i,i+1,i+2] for i in []],np.int64) if False else Fphys,k=1)[0]*1000,[0,.01,.5,.99,1]).tolist() if False else None,
    "D_to_nearest_registered_bank_mm_quantiles":np.quantile(deep_bank_d*1000,[0,.01,.5,.99,1]).tolist()},
   "exact_surface_gates":gate,"cut_trace_gate":bool(geometry_closed),"exact_no_crossing_gate":bool(exact_pairs==0),
   "unadmitted_reasons":["no conservative parent-tet overlap CSR or material mass ledger","current native owner integration for independent open I/D interfaces remains a separate gate"]}
  for name,S in (("skin_inner",SI),("fat_deep",SD)):
   p=OUT/f"{side}-{name}.npz"
   np.savez_compressed(p,points_world_m=S["points"],faces=S["faces"],parent_tet_edge_node_ids=S["parent_edge"],edge_parameter=S["edge_t"],parent_tet_ids=S["parent_tet"],parent_node_id=S["parent_node_id"],point_kind=S["point_kind"],parent_face_tet=S["parent_face_tet"],cancelled_internal_exact_faces=np.asarray([S["cancelled_internal_exact_faces"]],np.int64),
    source_shell_face_index=S["source_shell_face_index"],source_face_id=S["source_face_id"],source_face_barycentric=S["source_face_barycentric"],source_projection_world_m=S["source_projection_world_m"],source_projection_distance_m=S["source_projection_distance_m"],
    level_phi=np.asarray([lI if name=="skin_inner" else lD]),frame=np.asarray(["actual007-world"]),cut_trace_boundary_report=np.asarray([json.dumps(bI if name=="skin_inner" else bD)]))
   report.setdefault("outputs",{})[name]={"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
  (OUT/f"{side}-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  result["sides"][side]=report
  if not geometry_closed or exact_pairs!=0:
   result["status"]="stopped-at-first-side-exact-interface-or-obstacle-failure"
   result["failed_side"]=side
   result["elapsed_s"]=time.monotonic()-t0
   result["outputs"]={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
   (OUT/"candidate-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\\n")
   raise SystemExit(f"stop-first hard gate failure on {side}: exact_pairs={exact_pairs}, geometry_closed={geometry_closed}")
  print(json.dumps({"stage":"side_complete","side":side,"surface_gates":{k:v.get("intersection_pairs",v.get("count")) for k,v in gate.items()},"layer_volumes_m3":{"skin":vskin,"fat":vfat,"core":vcore},"interface_boundaries":{"I":bI,"D":bD},"status":report["status"]},sort_keys=True),flush=True)
 result["elapsed_s"]=time.monotonic()-t0;result["outputs"]={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
 result["status"]="left-tissue-aware-field-emitted-unadmitted"
 (OUT/"candidate-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"complete":True,"path":str(OUT/"candidate-report.json"),"elapsed_s":result["elapsed_s"]},sort_keys=True),flush=True)
if __name__=="__main__":main()

