#!/usr/bin/env python3
import argparse, csv, hashlib, json, math
from pathlib import Path
root=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
run=root/"P/native-detail-bridge-426/canonical-face-strict-reload-007"
source=root/"full-current-surface-registration-399/sidecars/registered-O.csv"
map_path=root/"P/native-detail-bridge-426/canonical-face-export-005/map-output/fine-O-to-coarse-boundary-native-canonical.csv"
node_path=root/"corrected-coarse-native-sidecars-410/result-002/volume/mesh.node"
ele_path=root/"corrected-coarse-native-sidecars-410/result-002/volume/mesh.ele"
labels_path=root/"corrected-coarse-native-sidecars-410/result-002/volume/child.labels.csv"
parser=argparse.ArgumentParser(description="Recompute pinned native effective-tet weights into an explicit evidence directory.")
parser.add_argument("--output-dir", required=True, help="external evidence directory; the source tree is never used for CSV output")
args=parser.parse_args()
outdir=Path(args.output_dir).resolve()
outdir.mkdir(parents=True,exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def read_nodes(path):
    with open(path) as f:
        lines=(line for line in f if line.strip() and not line.lstrip().startswith("#"))
        n=int(next(lines).split()[0]); pts=[None]*n
        for line in lines:
            t=line.split()
            if len(t)>=4: pts[int(t[0])-1]=tuple(map(float,t[1:4]))
    if any(p is None for p in pts): raise RuntimeError("incomplete TetGen node rows")
    return pts
def read_tets(path):
    with open(path) as f:
        lines=(line for line in f if line.strip() and not line.lstrip().startswith("#"))
        n=int(next(lines).split()[0]); tets=[None]*n
        for line in lines:
            t=line.split()
            if len(t)>=5: tets[int(t[0])-1]=tuple(int(x)-1 for x in t[1:5])
    if any(t is None for t in tets): raise RuntimeError("incomplete TetGen elements")
    return tets
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def sub(a,b): return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def accum(out,p,s): return (out[0]+p[0]*s,out[1]+p[1]*s,out[2]+p[2]*s)
def norm(a): return math.sqrt(dot(a,a))
pts=read_nodes(node_path); tets=read_tets(ele_path)
with open(source,newline="") as f: fine=list(csv.DictReader(f))
with open(map_path,newline="") as f: maps=list(csv.DictReader(f))
with open(labels_path,newline="") as f: labels=list(csv.DictReader(f))
if len(fine)!=56054 or len(maps)!=len(fine) or len(labels)!=len(tets): raise RuntimeError("pinned input row counts differ")
fields=["fine_vertex_row","fine_source_vertex_row","detail_world_x_m","detail_world_y_m","detail_world_z_m","source_frame_owner_body","fine_parent_material_index","fine_parent_geometric_donor_body","fine_parent_region_id","fine_parent_stable_source_face_id","canonical_selected_boundary_face_row","canonical_selected_parent_tet_row","tet_node0","tet_node1","tet_node2","tet_node3","tet_material_index","tet_geometric_donor_body","tet_region_id","effective_weight0","effective_weight1","effective_weight2","effective_weight3","max_abs_effective_weight","exceeds_declared_1p5_bound","tet_condition_estimate","tet_extrapolation_residual_x_m","tet_extrapolation_residual_y_m","tet_extrapolation_residual_z_m","tet_extrapolation_distance_m","canonical_surface_residual_x_m","canonical_surface_residual_y_m","canonical_surface_residual_z_m","canonical_surface_projection_distance_m","surface_residual_recompute_delta_m"]
output=outdir/"all-O-effective-tet-weights.csv"
count=violations=0; maximum=-1.; maxrow=-1; maxdistance=0.; maxcondition=0.; maxsurface=0.; maxSurfaceDelta=0.; first=[]
with open(output,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for i,(fr,mr) in enumerate(zip(fine,maps)):
        if int(mr["fine_vertex_row"])!=i: raise RuntimeError(f"map row mismatch at {i}")
        cell=int(mr["canonical_selected_parent_tet_row"])
        if not 0<=cell<len(tets): raise RuntimeError(f"invalid selected tet row {cell}")
        tet=tets[cell]; a,b,c,d=(pts[n] for n in tet)
        e1=sub(b,a);e2=sub(c,a);e3=sub(d,a);den=dot(e1,cross(e2,e3))
        if not math.isfinite(den) or den<=1.0e-24: raise RuntimeError(f"inverted/singular tet {cell}")
        p=(float(fr["wx"]),float(fr["wy"]),float(fr["wz"])); r=sub(p,a)
        inv0=tuple(x/den for x in cross(e2,e3));inv1=tuple(x/den for x in cross(e3,e1));inv2=tuple(x/den for x in cross(e1,e2))
        c0=dot(r,inv0);c1=dot(r,inv1);c2=dot(r,inv2)
        ww=(1.-c0-c1-c2,c0,c1,c2)
        mx=max(abs(x) for x in ww)
        if mx>maximum: maximum=mx;maxrow=i
        extrap=[min(1.,max(0.,x)) for x in ww]; es=sum(extrap)
        if not math.isfinite(es) or es<=0.: raise RuntimeError(f"invalid clamped weights at row {i}")
        extrap=[x/es for x in extrap]; q=(0.,0.,0.)
        for x,node in zip(extrap,tet): q=accum(q,pts[node],x)
        er=sub(p,q); ed=norm(er); maxdistance=max(maxdistance,ed)
        matrixF2=dot(e1,e1)+dot(e2,e2)+dot(e3,e3)
        inverseF2=dot(inv0,inv0)+dot(inv1,inv1)+dot(inv2,inv2)
        cond=math.sqrt(matrixF2*inverseF2);maxcondition=max(maxcondition,cond)
        lab=labels[cell]
        if int(lab["child_tet_row"])!=cell: raise RuntimeError(f"cell labels out of order at {cell}")
        if (int(lab["material_index"])!=int(mr["canonical_selected_material_index"]) or
            int(lab["geometric_donor_body"])!=int(mr["canonical_selected_geometric_donor_body"]) or
            int(lab["anatomical_region_id"])!=int(mr["canonical_selected_region_id"])):
            raise RuntimeError(f"selected tet labels disagree with pinned canonical map at row {i}")
        faceNodes=[int(mr[f"canonical_selected_volume_node{k}"]) for k in range(3)]
        faceBary=[float(mr[f"canonical_selected_face_bary{k}"]) for k in range(3)]
        fq=(0.,0.,0.)
        for x,node in zip(faceBary,faceNodes): fq=accum(fq,pts[node],x)
        sr=sub(p,fq);sd=norm(sr)
        csvsr=tuple(float(mr[f"canonical_selected_residual_{axis}_m"]) for axis in "xyz")
        sdelta=norm(sub(sr,csvsr));maxSurfaceDelta=max(maxSurfaceDelta,sdelta);maxsurface=max(maxsurface,sd)
        exc=mx>1.5
        if exc:
            violations+=1
            if len(first)<12:first.append({"row":i,"cell":cell,"max_abs_weight":mx,"tet_nodes":tet})
        vals={
          "fine_vertex_row":i,"fine_source_vertex_row":fr["source_vertex_row"],
          "detail_world_x_m":format(p[0],".17g"),"detail_world_y_m":format(p[1],".17g"),"detail_world_z_m":format(p[2],".17g"),
          "source_frame_owner_body":mr["fine_frame_owner_body"],"fine_parent_material_index":mr["fine_parent_material_index"],
          "fine_parent_geometric_donor_body":mr["fine_parent_geometric_donor_body"],"fine_parent_region_id":mr["fine_parent_region_id"],
          "fine_parent_stable_source_face_id":mr["fine_parent_stable_source_face_id"],
          "canonical_selected_boundary_face_row":mr["canonical_selected_boundary_face_row"],"canonical_selected_parent_tet_row":cell,
          "tet_node0":tet[0],"tet_node1":tet[1],"tet_node2":tet[2],"tet_node3":tet[3],
          "tet_material_index":lab["material_index"],"tet_geometric_donor_body":lab["geometric_donor_body"],"tet_region_id":lab["anatomical_region_id"],
          "effective_weight0":format(ww[0],".17g"),"effective_weight1":format(ww[1],".17g"),
          "effective_weight2":format(ww[2],".17g"),"effective_weight3":format(ww[3],".17g"),
          "max_abs_effective_weight":format(mx,".17g"),"exceeds_declared_1p5_bound":int(exc),
          "tet_condition_estimate":format(cond,".17g"),
          "tet_extrapolation_residual_x_m":format(er[0],".17g"),"tet_extrapolation_residual_y_m":format(er[1],".17g"),
          "tet_extrapolation_residual_z_m":format(er[2],".17g"),"tet_extrapolation_distance_m":format(ed,".17g"),
          "canonical_surface_residual_x_m":mr["canonical_selected_residual_x_m"],
          "canonical_surface_residual_y_m":mr["canonical_selected_residual_y_m"],
          "canonical_surface_residual_z_m":mr["canonical_selected_residual_z_m"],
          "canonical_surface_projection_distance_m":mr["canonical_selected_projection_distance_m"],
          "surface_residual_recompute_delta_m":format(sdelta,".17g")}
        w.writerow(vals);count+=1
def fileinfo(path): return {"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}
launch=json.load(open(run/"launch.json"))
receipt={
 "schema":"numi-human.detail-effective-tet-weight-witness.v1",
 "classification":"source-only mechanical-refinement diagnostic; not geometry admission, GPU validation, Runtime, or physical qualification",
 "native_failure":{"run":str(run),"reason":"detail-to-volume effective tet weight exceeds its declared bound","declared_max_absolute_effective_weight":1.5,"binary_sha256":launch["binary_sha256"],"launch_sha256":sha(run/"launch.json"),"stderr_sha256":sha(run/"stderr.log"),"stdout_sha256":sha(run/"stdout.log")},
 "coordinate_domains":{"fine_points":"399 registered-O.csv wx/wy/wz in detail source row order","tet_points":"410 canonical imported TetGen mesh.node coordinates","selected_tets":"canonical_selected_parent_tet_row from frozen 005 native selected-face map; connectivity from 410 mesh.ele","surface_projection_residual":"canonical selected surface residual from the 005 map, independently recomputed and compared","tet_extrapolation_residual":"detail point minus clipped-and-renormalized barycentric reconstruction in selected tetrahedron"},
 "inputs":{k:fileinfo(v) for k,v in {"fine_registered_O":source,"canonical_native_face_map":map_path,"canonical_imported_tetgen_nodes":node_path,"canonical_imported_tetgen_elements":ele_path,"canonical_cell_labels":labels_path}.items()},
 "method":"Replayed the C++ tetBarycentric formula in double precision: e1/e2/e3, determinant, inverse cofactors, c0/c1/c2, weights [1-c0-c1-c2,c0,c1,c2]. Tet extrapolation uses clamp(weights,0,1), normalize, reconstruct, and subtract from source point, matching conservativeTetExtrapolationDistance.",
 "declared_gates":{"maximum_absolute_effective_weight":1.5,"maximum_tet_extrapolation_distance_m":0.05,"maximum_surface_projection_distance_m":0.02},
 "result":{"all_O_rows":count,"over_effective_weight_bound":violations,"max_abs_effective_weight":maximum,"max_effective_weight_row":maxrow,"max_tet_extrapolation_distance_m":maxdistance,"max_tet_condition_estimate":maxcondition,"max_surface_projection_distance_m":maxsurface,"max_surface_residual_recompute_delta_m":maxSurfaceDelta,"first_twelve_over_bound":first},
 "generator":{"source_path":str(Path(__file__).resolve()),"source_sha256":sha(Path(__file__).resolve())},
 "output":fileinfo(output)}
rp=outdir/"receipt.json";rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(output),"receipt":str(rp),"rows":count,"over_1p5":violations,"max_abs_weight":maximum,"max_weight_row":maxrow,"max_tet_extrapolation_m":maxdistance,"max_surface_projection_m":maxsurface,"surface_residual_recompute_delta_max_m":maxSurfaceDelta,"output_sha256":sha(output),"receipt_sha256":sha(rp)},indent=2))
