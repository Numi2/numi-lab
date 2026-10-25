import hashlib, json
from pathlib import Path
import numpy as np

E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
RUN=E/"E/connected-volume-refinement-037/run-006"
BASE=E/"route-audit-counter-fix-030/baseline-035.npz"
REF=E/"connected-volume-repair-032/repair-input.npz"
PINS={"candidate.npz":"52c81dbe323eab2240bc315f7b7bde605a201fa8e928206f27000de7e5d7a492",
      "baseline_035.npz":"452602aea036c15faaafe6c154e6c4676585e1b2cb57c9a1c9128de82050b570",
      "repair-input.npz":"3df28e930e5b27229633bf2e3a91f0d40fd6e57d94489c3ba7de6ae2828a0e1d"}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def volq(x,t):
    p=np.asarray(x,dtype=np.float64)[np.asarray(t,dtype=np.int64)]
    d=np.einsum("ij,ij->i",p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]))
    me=np.zeros(len(p))
    for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
        me=np.maximum(me,np.linalg.norm(p[:,i]-p[:,j],axis=1))
    return d/6,d/np.maximum(me,np.finfo(float).tiny)**3
def main():
    for name,path in (("candidate.npz",RUN/"candidate.npz"),("baseline_035.npz",BASE),("repair-input.npz",REF)):
        if sha(path)!=PINS[name]: raise RuntimeError("pinned hash mismatch: "+str(path))
    b=np.load(BASE,allow_pickle=False); r=np.load(REF,allow_pickle=False); c=np.load(RUN/"candidate.npz",allow_pickle=False)
    n=len(b["points_world_f32"]); m=len(b["tetrahedra"])
    if not (np.array_equal(b["points_world_f32"],r["points_world_f32"]) and np.array_equal(b["tetrahedra"],r["tetrahedra"]) and np.array_equal(b["material_ids"],r["material_ids"]) and np.array_equal(b["global_node_ids"],r["global_nodes"])):
        raise RuntimeError("baseline/repair source domain mismatch")
    ref=r["reference_points_m"]; cur=b["points_world_f32"]; oldt=b["tetrahedra"]; outt=c["tetrahedra"]; parent=c["child_parent_rows"]
    if not (np.array_equal(c["current_points_f32"][:n],cur) and np.array_equal(c["canonical_reference_points_f64"][:n],ref) and np.array_equal(c["reference_points_f32"][:n],ref.astype(np.float32))):
        raise RuntimeError("source prefix exactness failed")
    if not np.array_equal(c["material_ids"],b["material_ids"][parent]) or not np.array_equal(c["child_parent_cell_ids"],parent):
        raise RuntimeError("parent/material lineage failed")
    if not np.array_equal(c["global_node_ids_original_prefix"],b["global_node_ids"]):
        raise RuntimeError("global source ID prefix failed")
    edges=c["new_node_parent_edges_local_rows"]; newrows=c["new_node_rows"]
    if len(edges)!=2087 or len(newrows)!=2087 or not np.array_equal(newrows,np.arange(n,n+len(newrows))):
        raise RuntimeError("midpoint row/edge inventory mismatch")
    ptr,ids,w=c["source_vertex_ancestry_indptr"],c["source_vertex_ancestry_global_ids"],c["source_vertex_ancestry_weights"]
    if len(ptr)!=len(c["current_points_f32"])+1 or not np.array_equal(ptr[:n+1],np.arange(n+1)) or not np.array_equal(ids[:n],b["global_node_ids"]) or not np.array_equal(w[:n],np.ones(n)):
        raise RuntimeError("source ancestry prefix failed")
    sourceid_to_row={int(g):i for i,g in enumerate(b["global_node_ids"])}
    for row,(a,bb) in zip(newrows,edges):
        if not np.array_equal(c["canonical_reference_points_f64"][row],(ref[a]+ref[bb])*0.5):
            raise RuntimeError("F64 midpoint not derived from exact source endpoints")
        lo,hi=int(ptr[row]),int(ptr[row+1])
        exp=np.sort(b["global_node_ids"][[a,bb]])
        if hi-lo!=2 or not np.array_equal(np.sort(ids[lo:hi]),exp) or not np.array_equal(np.sort(w[lo:hi]),np.asarray([.5,.5])):
            raise RuntimeError("midpoint source CSR does not map to its two edge endpoints")
    rv,rq=volq(c["canonical_reference_points_f64"],outt); r32v,r32q=volq(c["reference_points_f32"],outt); cv,cq=volq(c["current_points_f32"],outt)
    oldrv,_=volq(ref,oldt); old32v,_=volq(ref.astype(np.float32),oldt)
    pafter=np.bincount(parent,weights=rv,minlength=m); drift=np.abs(pafter-oldrv)/oldrv
    f32after=np.bincount(parent,weights=r32v,minlength=m); f32drift=np.abs(f32after-old32v)/old32v
    if float(drift.max())>1e-4 or min(float(rq.min()),float(r32q.min()),float(cq.min()))<1e-5:
        raise RuntimeError("F64 closure or F32 quality gate failed")
    if not np.allclose(rv/oldrv[parent],c["child_reference_volume_fraction_of_input_parent_f64"],rtol=0,atol=1e-15):
        raise RuntimeError("saved canonical F64 fractions mismatch")
    faces=np.concatenate((outt[:,[1,2,3]],outt[:,[0,3,2]],outt[:,[0,1,3]],outt[:,[0,2,1]]),axis=0)
    keys=np.sort(faces,axis=1)
    unique,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    if np.any((counts!=1)&(counts!=2)): raise RuntimeError("nonmanifold face incidence")
    ordered=np.argsort(faces,axis=1)
    parity=((ordered[:,0]>ordered[:,1]).astype(np.int8)+(ordered[:,0]>ordered[:,2]).astype(np.int8)+(ordered[:,1]>ordered[:,2]).astype(np.int8))%2
    orient=np.where(parity==0,1,-1)
    balance=np.bincount(inv,weights=orient,minlength=len(unique))
    if np.any((counts==2)&(balance!=0)): raise RuntimeError("internal face orientation mismatch")
    report={"status":"PASS_INDEPENDENT_CONSTRUCTION_AND_ANCESTRY_CHECK","candidate_sha256":sha(RUN/"candidate.npz"),
        "baseline_035_sha256":sha(BASE),"repair_input_sha256":sha(REF),"nodes_input":n,"nodes_output":len(c["current_points_f32"]),
        "tets_input":m,"tets_output":len(outt),"marked_edges":len(edges),"source_prefix_exact":True,
        "midpoint_F64_endpoint_and_CSR_check":"PASS","material_parent_inheritance":"PASS","internal_face_orientation":"PASS",
        "boundary_face_count":int(np.count_nonzero(counts==1)),"canonical_f64_max_parent_volume_drift":float(drift.max()),
        "canonical_f64_total_volume_drift":float(abs(rv.sum()-oldrv.sum())/oldrv.sum()),
        "emitted_f32_reference_parent_volume_max_drift_diagnostic_only":float(f32drift.max()),
        "reference_f32_min_q":float(r32q.min()),"current_f32_min_q":float(cq.min()),"canonical_f64_min_q":float(rq.min()),
        "physical_mass_transfer":False,"bone_clearance":False,"native_physics":False}
    (RUN/"independent-verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
