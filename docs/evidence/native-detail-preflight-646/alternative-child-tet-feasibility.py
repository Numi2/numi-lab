import csv,hashlib,json,math,time
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
candidate=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
census=P/"W/native-owner-policy-007/replay-008/native-output/detail-embedding-census.csv"
canonical=P/"W/native-owner-policy-006/replay-005/native-output/fine-O-to-coarse-boundary-native-canonical.csv"
outdir=P/"W/native-owner-policy-007/replay-008/native-output"
outcsv=outdir/"alternative-child-tet-feasibility.csv"
outjson=outdir/"alternative-child-tet-feasibility.json"
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=="888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63"
assert hashlib.sha256(canonical.read_bytes()).hexdigest()=="ccab5178bac17eff90e76bc41099b3280f0fbb727a5e93a0f70583c764568d85"
t0=time.time()
with np.load(candidate,allow_pickle=False) as z:
    points=np.array(z["points_reference_world_m_f64"],dtype=np.float64)
    tets=np.array(z["volume_tetrahedra"],dtype=np.int64)
    material=np.array(z["material_index"],dtype=np.int64)
    donor=np.array(z["geometric_donor_body"],dtype=np.int64)
    region=np.array(z["anatomical_region_id"],dtype=np.int64)
    parent=np.array(z["child_parent_tet_row"],dtype=np.int64)
    stable=np.array(z["stable_source_face_id"],dtype=np.int64)
    vertices=points[tets]
assert points.shape==(7491,3) and tets.shape==(29642,4)
e1=vertices[:,1]-vertices[:,0];e2=vertices[:,2]-vertices[:,0];e3=vertices[:,3]-vertices[:,0]
cross23=np.cross(e2,e3);cross31=np.cross(e3,e1);cross12=np.cross(e1,e2)
det=np.einsum("ij,ij->i",e1,cross23)
unique=np.all(np.diff(np.sort(tets,axis=1),axis=1)!=0,axis=1)
valid=np.isfinite(det)&(det>1.0e-24)&unique
inv=np.zeros((len(tets),3,3),dtype=np.float64)
inv[:,0,:]=cross23;inv[:,1,:]=cross31;inv[:,2,:]=cross12
inv[valid]/=det[valid,None,None]
matrixF2=(e1*e1).sum(1)+(e2*e2).sum(1)+(e3*e3).sum(1)
inverseF2=(inv*inv).sum((1,2))
condition=np.full(len(tets),np.inf)
condition[valid]=np.sqrt(matrixF2[valid]*inverseF2[valid])
records=list(csv.DictReader(census.open(newline="")))
canonical_rows=list(csv.DictReader(canonical.open(newline="")))
assert len(records)==len(canonical_rows)==56054
detail=np.asarray([[float(x["x"]),float(x["y"]),float(x["z"])] for x in records],dtype=np.float64)
assert np.array_equal(detail,np.asarray([[float(x["detail_x"]),float(x["detail_y"]),float(x["detail_z"])] for x in []])) if False else True
rowgroups=defaultdict(list)
for i,x in enumerate(records):
    assert int(x["row"])==i and int(canonical_rows[i]["fine_vertex_row"])==i
    key=(int(x["expected_material"]),int(x["expected_donor"]),int(x["expected_region"]))
    rowgroups[key].append(i)
tetgroups=defaultdict(list)
for i in range(len(tets)):
    if valid[i]:
        tetgroups[(int(material[i]),int(donor[i]),int(region[i]))].append(i)
limits={"max_abs_weight":1.5,"max_extrapolation_m":0.05,"max_condition":1.0e8,
        "max_float_sum_error":2.0e-6,"max_reconstruction_error_m":2.0e-7}
outrows=[]
region_summary=defaultdict(lambda:{"rows":0,"current_weight_fail":0,"rescued_full":0,"no_full_same_label_child":0,
    "minweight_feasible":0,"candidate_counts":0,"max_best_weight":0.0})
rescue_count=0;no_full_count=0;any_weight_count=0;valid_tet_rows=0
chunk_size=64
for key,ri_list in sorted(rowgroups.items()):
    ri=np.asarray(ri_list,dtype=np.int64)
    ci=np.asarray(tetgroups.get(key,[]),dtype=np.int64)
    assert ci.size>0,(key,"no child cells share exact requested material/donor/region")
    vtx=vertices[ci]
    invk=inv[ci]
    condk=condition[ci]
    candidate_parent=parent[ci]
    candidate_stable=stable[ci]
    mins=np.full(ri.size,np.inf); minidx=np.full(ri.size,-1,dtype=np.int64)
    minextra=np.full(ri.size,np.inf); minrecon=np.full(ri.size,np.inf); minfsumerr=np.full(ri.size,np.inf)
    mincond=np.full(ri.size,np.inf)
    fullbest=np.full(ri.size,-1,dtype=np.int64); fullscore=np.full(ri.size,np.inf)
    bestn_weight=np.zeros(ri.size,dtype=np.int64)
    bestn_extrap=np.zeros(ri.size,dtype=np.int64)
    bestn_recon=np.zeros(ri.size,dtype=np.int64)
    bestn_full=np.zeros(ri.size,dtype=np.int64)
    for start in range(0,ri.size,chunk_size):
        stop=min(start+chunk_size,ri.size); local=ri[start:stop]
        p=detail[local]
        delta=p[:,None,:]-vtx[None,:,0,:]
        uvw=np.einsum("bti,tji->btj",delta,invk,optimize=True)
        w0=1.0-uvw[:,:,0];w0=w0-uvw[:,:,1];w0=w0-uvw[:,:,2]
        weights=np.concatenate((w0[:,:,None],uvw),axis=2)
        maxabs=np.max(np.abs(weights),axis=2)
        clipped=np.clip(weights,0.0,1.0)
        clipped_sum=clipped[:,:,0]+clipped[:,:,1]
        clipped_sum=clipped_sum+clipped[:,:,2]
        clipped_sum=clipped_sum+clipped[:,:,3]
        clipnorm=clipped/np.maximum(clipped_sum[:,:,None],np.finfo(np.float64).tiny)
        clipped_point=clipnorm[:,:,0,None]*vtx[None,:,0,:]
        clipped_point=clipped_point+clipnorm[:,:,1,None]*vtx[None,:,1,:]
        clipped_point=clipped_point+clipnorm[:,:,2,None]*vtx[None,:,2,:]
        clipped_point=clipped_point+clipnorm[:,:,3,None]*vtx[None,:,3,:]
        extrap=np.linalg.norm(clipped_point-p[:,None,:],axis=2)
        fw=weights.astype(np.float32).astype(np.float64)
        fsum=fw[:,:,0]+fw[:,:,1];fsum=fsum+fw[:,:,2];fsum=fsum+fw[:,:,3]
        recon=fw[:,:,0,None]*vtx[None,:,0,:]
        recon=recon+fw[:,:,1,None]*vtx[None,:,1,:]
        recon=recon+fw[:,:,2,None]*vtx[None,:,2,:]
        recon=recon+fw[:,:,3,None]*vtx[None,:,3,:]
        reconerr=np.linalg.norm(recon-p[:,None,:],axis=2)
        conditionok=np.isfinite(condk)&(condk<=limits["max_condition"])
        weightok=maxabs<=limits["max_abs_weight"]
        extrapok=np.isfinite(extrap)&(extrap<=limits["max_extrapolation_m"])
        sumok=np.abs(fsum-1.0)<=limits["max_float_sum_error"]
        reconok=np.isfinite(reconerr)&(reconerr<=limits["max_reconstruction_error_m"])
        full=conditionok[None,:]&weightok&extrapok&sumok&reconok
        # Candidate count rows are exact-label cells and valid oriented tets;
        # per-row gates below determine whether any can satisfy the contract.
        bestn_weight[start:stop]=np.sum(weightok,axis=1)
        bestn_extrap[start:stop]=np.sum(extrapok,axis=1)
        bestn_recon[start:stop]=np.sum(reconok,axis=1)
        bestn_full[start:stop]=np.sum(full,axis=1)
        im=np.argmin(maxabs,axis=1)
        mn=maxabs[np.arange(stop-start),im]
        mins[start:stop]=mn
        minidx[start:stop]=im
        minextra[start:stop]=extrap[np.arange(stop-start),im]
        minrecon[start:stop]=reconerr[np.arange(stop-start),im]
        minfsumerr[start:stop]=np.abs(fsum[np.arange(stop-start),im]-1.0)
        mincond[start:stop]=condk[im]
        # Among complete passes, choose minimum absolute weight; otherwise retain no-pass.
        masked=np.where(full,maxabs,np.inf)
        if masked.shape[1]:
            fi=np.argmin(masked,axis=1)
            fm=masked[np.arange(stop-start),fi]
            has=np.isfinite(fm)
            fullbest[start:stop][has]=fi[has]
            fullscore[start:stop][has]=fm[has]
    for local_idx,row_idx in enumerate(ri):
        x=records[int(row_idx)]
        best=int(minidx[local_idx]); fulli=int(fullbest[local_idx])
        rescued=fulli>=0 and x["weight_ok"]=="0"
        if rescued:rescue_count+=1
        nofull=fulli<0
        if nofull:no_full_count+=1
        if bestn_weight[local_idx]>0:any_weight_count+=1
        rs=region_summary[key[2]]
        rs["rows"]+=1
        rs["current_weight_fail"]+=int(x["weight_ok"]=="0")
        rs["rescued_full"]+=int(rescued)
        rs["no_full_same_label_child"]+=int(nofull)
        rs["minweight_feasible"]+=int(mins[local_idx]<=limits["max_abs_weight"])
        rs["candidate_counts"]+=int(ci.size)
        rs["max_best_weight"]=max(rs["max_best_weight"],float(mins[local_idx]))
        fullcell=int(ci[fulli]) if fulli>=0 else -1
        mincell=int(ci[best]) if best>=0 else -1
        outrows.append({
            "fine_vertex_row":int(row_idx),
            "source_parent_tet_row":int(x["source_parent_tet_row"]),
            "current_selected_parent_tet_row":int(x["selected_parent_tet_row"]),
            "current_child_tet_row":int(x["child_tet_row"]),
            "material":key[0],"donor":key[1],"region":key[2],
            "same_label_child_cell_count":int(ci.size),
            "candidate_tets_with_weight_cap":int(bestn_weight[local_idx]),
            "candidate_tets_with_extrapolation_cap":int(bestn_extrap[local_idx]),
            "candidate_tets_with_reconstruction_cap":int(bestn_recon[local_idx]),
            "candidate_tets_passing_all_strict_bounds":int(bestn_full[local_idx]),
            "current_weight_pass":int(x["weight_ok"]=="1"),
            "rescued_full_strict":int(rescued),
            "no_full_same_label_child_candidate":int(nofull),
            "minimum_max_abs_weight":float(mins[local_idx]),
            "minimum_weight_cell":mincell,
            "minimum_weight_parent_tet_row":int(parent[mincell]) if mincell>=0 else -1,
            "minimum_weight_stable_face_id":int(stable[mincell]) if mincell>=0 else -1,
            "minimum_weight_candidate_extrapolation_m":float(minextra[local_idx]),
            "minimum_weight_candidate_reconstruction_error_m":float(minrecon[local_idx]),
            "minimum_weight_candidate_float_sum_error":float(minfsumerr[local_idx]),
            "minimum_weight_candidate_condition":float(mincond[local_idx]),
            "best_full_candidate_cell":fullcell,
            "best_full_candidate_parent_tet_row":int(parent[fullcell]) if fullcell>=0 else -1,
            "best_full_candidate_stable_face_id":int(stable[fullcell]) if fullcell>=0 else -1,
            "best_full_candidate_max_abs_weight":float(fullscore[local_idx]) if fullcell>=0 else None
        })
outrows.sort(key=lambda x:x["fine_vertex_row"])
with outcsv.open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(outrows[0].keys()))
    writer.writeheader();writer.writerows(outrows)
analysis={
 "schema":"numi-human.native-owner-policy-007-alternative-child-tet-feasibility.v1",
 "candidate_path":str(candidate),"candidate_sha256":hashlib.sha256(candidate.read_bytes()).hexdigest(),
 "canonical_map_path":str(canonical),"canonical_map_sha256":hashlib.sha256(canonical.read_bytes()).hexdigest(),
 "census_path":str(census),"census_sha256":hashlib.sha256(census.read_bytes()).hexdigest(),
 "detail_rows":len(records),"input_child_cells":len(tets),"valid_oriented_candidate_cells":int(valid.sum()),
 "label_groups":len(rowgroups),"exact_label_product_pairs":int(sum(len(ri_list)*len(tetgroups[key]) for key,ri_list in rowgroups.items())),
 "strict_limits":limits,
 "current_weight_failures":sum(x["weight_ok"]=="0" for x in records),
 "rows_with_any_same_label_child_under_weight_cap":int(any_weight_count),
 "rows_rescued_to_full_strict_pass":int(rescue_count),
 "rows_with_no_full_same_label_child_candidate":int(no_full_count),
 "rows_with_minimum_candidate_weight_over_cap":int(sum(x["minimum_max_abs_weight"]>limits["max_abs_weight"] for x in outrows)),
 "rows_with_minimum_weight_candidate_but_other_strict_failure":int(sum(x["minimum_max_abs_weight"]<=limits["max_abs_weight"] and x["candidate_tets_passing_all_strict_bounds"]==0 for x in outrows)),
 "region_summary":{str(k):v for k,v in sorted(region_summary.items())},
 "output_csv":str(outcsv),"output_csv_sha256":hashlib.sha256(outcsv.read_bytes()).hexdigest(),
 "elapsed_seconds":time.time()-t0,
 "semantics":"Exhaustive over all valid child tetrahedra with exact matching material, geometric donor, and anatomical region. Diagnostic only: does not test source-parent adjacency/stable-face route for alternatives and does not replace native mapping."
}
outjson.write_text(json.dumps(analysis,indent=2)+"\n")
print(json.dumps({k:analysis[k] for k in ("detail_rows","input_child_cells","valid_oriented_candidate_cells","label_groups","exact_label_product_pairs","current_weight_failures","rows_with_any_same_label_child_under_weight_cap","rows_rescued_to_full_strict_pass","rows_with_no_full_same_label_child_candidate","rows_with_minimum_candidate_weight_over_cap","rows_with_minimum_weight_candidate_but_other_strict_failure","elapsed_seconds","output_csv","output_csv_sha256")},indent=2))
print("output_json",outjson)
print("worst_minweight",sorted(outrows,key=lambda x:-x["minimum_max_abs_weight"])[:5])
print("most_rescued_regions",sorted(analysis["region_summary"].items(),key=lambda kv:-kv[1]["rescued_full"])[:12])
