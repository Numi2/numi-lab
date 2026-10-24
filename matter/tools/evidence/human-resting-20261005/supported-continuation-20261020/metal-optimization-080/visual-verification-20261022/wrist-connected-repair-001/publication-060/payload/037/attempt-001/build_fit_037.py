#!/usr/bin/env python3
"""037 component-separable all-bank bone clearance fit, using frozen 035/036 inputs."""
from pathlib import Path
import csv, json, hashlib, time, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-037-preparation-035")
from fit_support_037 import layer_floor_mm, normalized_tet_quality_jacobian

ROOT=Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
PREP=ROOT/"bone-clearance-fit-037-preparation-035"
OUT=ROOT/"bone-clearance-fit-037-componentwise-035affine"
OUT.mkdir(exist_ok=True)
AP=ROOT/"bone-clearance-feasibility-033/world-bone-clearance-constraints.npz"
PP=ROOT/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
DONOR=ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/generated/donor-frames-fullweights.csv"
POSE=ROOT/"native-candidate020-bone-witness-029/source-preflight-002/registered-skin-poses.json"
AFF=ROOT/"E/native-candidate020-all-bank-affine-roundtrip-036/all-bank-local-to-world-rows.npz"
AFF_REPORT=ROOT/"E/native-candidate020-all-bank-affine-roundtrip-036/report.json"
COV=ROOT/"E/native-candidate020-all-bank-affine-roundtrip-036/covector-plane-037/inside-row-covector-planes.npz"
PINFILES=[AP,PP,DONOR,POSE,AFF,COV,PREP/"per-bone-constraints-037.npz",PREP/"restart-state-037.npz"]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def q_and_grad(V):
    return normalized_tet_quality_jacobian(V)

def quat_rotate_matrix(q):
    """Native rotate(raw q, basis) matrix, with no quaternion normalization."""
    q=np.asarray(q,dtype=np.float64).reshape(4)
    xyz=q[:3]; w=q[3]
    M=np.empty((3,3),dtype=np.float64)
    for j in range(3):
        v=np.eye(3,dtype=np.float64)[j]
        t=2.0*np.cross(xyz,v)
        M[:,j]=v+w*t+np.cross(xyz,t)
    return M

def affine_planes(hnode,hbody,hbone,aff_path,pose_path,X0):
    z=np.load(aff_path,allow_pickle=False)
    required={"layer","row","body_index","bone_stable_id","classification_code","query_world_m","query_local_m","closest_local_m","closest_world_m","affine_roundtrip_residual_m"}
    if not required.issubset(z.files): raise ValueError("036 affine rows missing required arrays")
    poses={int(r["body_index"]):r["current"] for r in json.loads(Path(pose_path).read_text())["all_bodies"]}
    cv=np.load(COV,allow_pickle=False)
    covlookup={}
    for ci in range(len(cv["row"])):
        covkey=(int(cv["layer"][ci])*2411+int(cv["row"][ci]),int(cv["body_index"][ci]),int(cv["bone_stable_id"][ci]))
        if covkey in covlookup: raise ValueError("duplicate covector row key")
        covlookup[covkey]=ci
    lookup={}
    for i in range(len(z["row"])):
        layer=int(z["layer"][i]); row=int(z["row"][i]); node=layer*2411+row
        key=(node,int(z["body_index"][i]),int(z["bone_stable_id"][i]))
        if key in lookup: raise ValueError("duplicate 036 affine node/body/bone key")
        lookup[key]=i
        if int(z["classification_code"][i]) not in (0,1): raise ValueError("unknown 036 classification code")
    nodes=[]; bodies=[]; bones=[]; tris=[]; cp=[]; normal=[]; basegap=[]; residual=[]; nlocal_out=[]
    for node,body,bone in zip(hnode,hbody,hbone):
        key=(int(node),int(body),int(bone))
        if key not in lookup: raise ValueError(f"missing 036 affine row {key}")
        i=lookup[key]
        if int(z["classification_code"][i])!=1: raise ValueError(f"035 inside row is not classified inside by 036 {key}")
        if body not in poses: raise ValueError(f"missing body pose {body}")
        cur=poses[body]
        qq=np.asarray(cur["orientation_world_xyzw"],dtype=np.float32).astype(np.float64).copy()
        if qq.shape!=(4,) or not np.all(np.isfinite(qq)): raise ValueError("bad raw current quaternion")
        qq[:3]*=-1.0
        M=quat_rotate_matrix(qq)
        ql=np.asarray(z["query_local_m"][i],dtype=np.float64)
        cl=np.asarray(z["closest_local_m"][i],dtype=np.float64)
        delta=cl-ql; dn=float(np.linalg.norm(delta))
        if dn<=0 or not np.isfinite(dn): raise ValueError("inside row lacks positive local escape vector")
        nl=delta/dn
        nw=M.T@nl; nn=float(np.linalg.norm(nw))
        if nn<=0 or not np.isfinite(nn): raise ValueError("invalid covector-transformed plane normal")
        nw/=nn
        if key not in covlookup: raise ValueError(f"missing covector receipt row {key}")
        ci=covlookup[key]
        if int(cv["classification_code"][ci])!=1: raise ValueError(f"covector receipt row not inside {key}")
        nrc=np.asarray(cv["unit_plane_normal_world"][ci],dtype=np.float64)
        if not np.all(np.isfinite(nrc)) or np.linalg.norm(nrc-nw)>2e-12: raise ValueError(f"covector receipt mismatch {key}")
        nw=nrc
        qworld=np.asarray(z["query_world_m"][i],dtype=np.float64)
        cworld=np.asarray(z["closest_world_m"][i],dtype=np.float64)
        # Verify the affine difference cancels its COM-offset translation.
        local_delta=M@(cworld-qworld)
        err=float(np.linalg.norm(local_delta-delta))
        if err>max(3e-10,1e-8*dn): raise ValueError(f"036 affine local/world difference mismatch {key}: {err}")
        x=np.asarray(X0[int(node)],dtype=np.float64)
        if np.linalg.norm(x-qworld)>1e-12: raise ValueError(f"036 query point not current physical row {key}")
        gap=float(np.dot(x-cworld,nw))
        if gap>5e-8: raise ValueError(f"inside row has nonnegative covector gap {key}: {gap}")
        nodes.append(int(node));bodies.append(int(body));bones.append(int(bone));tris.append(int(z["nearest_triangle"][i]))
        cp.append(cworld);normal.append(nw);basegap.append(gap);residual.append(err);nlocal_out.append(nl)
    return {"node":np.asarray(nodes,np.int64),"body":np.asarray(bodies,np.int64),"bone":np.asarray(bones,np.int64),
            "triangle":np.asarray(tris,np.int64),"closest_world_m":np.asarray(cp,float),
            "normal_world":np.asarray(normal,float),"baseline_gap_m":np.asarray(basegap,float),
            "difference_roundtrip_m":np.asarray(residual,float),"normal_local":np.asarray(nlocal_out,float)}

class DSU:
    def __init__(self,n): self.p=np.arange(n,dtype=np.int64);self.r=np.zeros(n,np.int8)
    def find(self,x):
        x=int(x)
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]];x=int(self.p[x])
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.r[a]<self.r[b]:a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]:self.r[a]+=1

def quality_mm(Xmm,T,Trows):
    V=Xmm[T[Trows]]
    q,_=q_and_grad(V)
    return q

def full_metrics(Xm,T,materials):
    X32=np.asarray(Xm,dtype=np.float32).astype(np.float64)
    V64=np.asarray(Xm,dtype=np.float64)[T]
    V32=X32[T]
    q64,_=q_and_grad(V64);q32,_=q_and_grad(V32)
    det64=np.einsum("ij,ij->i",V64[:,1]-V64[:,0],np.cross(V64[:,2]-V64[:,0],V64[:,3]-V64[:,0]))
    det32=np.einsum("ij,ij->i",V32[:,1]-V32[:,0],np.cross(V32[:,2]-V32[:,0],V32[:,3]-V32[:,0]))
    vols={str(int(k)):float(det64[materials==k].sum()/6.0) for k in np.unique(materials)}
    return {"nonpositive_float64":int(np.sum(det64<=0)),"nonpositive_float32":int(np.sum(det32<=0)),
            "below_q1e5_float64":int(np.sum(q64<1e-5)),"below_q1e5_float32":int(np.sum(q32<1e-5)),
            "min_det_float64_m3":float(det64.min()),"min_det_float32_m3":float(det32.min()),
            "min_q_float64":float(q64.min()),"min_q_float32":float(q32.min()),
            "p01_q_float32":float(np.percentile(q32,1)),"material_volume_m3":vols}

def main():
    ap=np.load(AP); pp=np.load(PP); prep=np.load(PREP/"per-bone-constraints-037.npz"); rs=np.load(PREP/"restart-state-037.npz")
    X0=np.asarray(ap["points"],dtype=np.float32).astype(np.float64)
    T=np.asarray(ap["tetrahedra"],dtype=np.int64); material=np.asarray(pp["material_ids"],dtype=np.int64)
    assert X0.shape==(7233,3) and T.shape==(28632,4) and np.array_equal(T,pp["tetrahedra"]) and np.array_equal(material,pp["material_ids"])
    hnode=np.asarray(prep["constraint_local_node"],np.int64);hbody=np.asarray(prep["constraint_body_index"],np.int64);hbone=np.asarray(prep["constraint_bone_stable_id"],np.int64)
    assert len(hnode)==139 and len(set(zip(hnode.tolist(),hbody.tolist(),hbone.tolist())))==139
    b=affine_planes(hnode,hbody,hbone,AFF,POSE,X0)
    H=len(hnode); margin_m=float(ap["inferred_margin_m"]); margin_mm=margin_m*1000.
    Vrows=np.asarray(rs["variable_local_rows"],np.int64); xrestart=np.asarray(rs["x_mm"],np.float64)
    Ttouch=np.asarray(rs["affected_tet_rows"],np.int64); active=np.asarray(rs["initial_active_tet_rows"],np.int64)
    assert len(Vrows)==858 and len(Ttouch)==7060 and xrestart.shape==(len(Vrows),3)
    QF=1.25e-5; QACTIVE=QF*1.02; BUDGET=float(__import__("os").environ.get("FIT_BUDGET_S","105"))
    OF=np.asarray(pp["O_faces"],np.int64); ring=np.unique(Vrows%2411)
    assert len(ring)==423
    allbase_mm=X0*1000.0
    current=allbase_mm.copy();current[Vrows]=xrestart
    vm=np.full(len(X0),-1,np.int64);vm[Vrows]=np.arange(len(Vrows))
    # Match preparation's full one-ring domain and quality active rows.
    t_touch=T[Ttouch]
    assert np.array_equal(np.unique(Ttouch[np.isin(t_touch,Vrows).any(axis=1)]),Ttouch)
    var_members=np.isin(t_touch,Vrows).sum(axis=1)
    if np.any(var_members==0): raise ValueError("affected cell has no variable node")
    # Exact variable components from quality-cell, O-edge objective, and layer constraints.
    dsu=DSU(len(Vrows))
    for nodes in t_touch:
        ix=vm[nodes];ix=ix[ix>=0]
        if len(ix)>1:
            for j in ix[1:]:dsu.union(ix[0],j)
    col_to_v={int(c):[] for c in ring}
    for j,n in enumerate(Vrows): col_to_v[int(n%2411)].append(j)
    for c,ixs in col_to_v.items():
        for j in ixs[1:]:dsu.union(ixs[0],j)
    edge_set=set()
    for f in OF:
        for u,v in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            for off in (0,2411,4822):
                na=off+int(u);nb=off+int(v);a=int(vm[na]);bb=int(vm[nb])
                if a>=0 or bb>=0:
                    edge_set.add((min(na,nb),max(na,nb)))
                    if a>=0 and bb>=0:
                        dsu.union(a,bb)
    roots=np.asarray([dsu.find(i) for i in range(len(Vrows))],np.int64)
    unique_roots=sorted(set(roots.tolist()),key=lambda r:(-int(np.sum(roots==r)),int(r)))
    groups=[np.flatnonzero(roots==r) for r in unique_roots]
    comp_of_v=np.full(len(Vrows),-1,np.int64)
    for ci,g in enumerate(groups):comp_of_v[g]=ci
    counts=[int(len(g)) for g in groups]
    expected=[364,103,72,65,58,52,46,38,18,14,14,14]
    if sorted(counts,reverse=True)!=expected: raise ValueError(f"component decomposition mismatch {sorted(counts,reverse=True)} != {expected}")
    comp_of_node=np.full(len(X0),-1,np.int64);comp_of_node[Vrows]=comp_of_v
    # Every affected tet belongs to exactly one component by its free nodes.
    row_comp=np.empty(len(Ttouch),np.int64)
    for k,nodes in enumerate(t_touch):
        c=np.unique(comp_of_node[nodes]);c=c[c>=0]
        if len(c)!=1:raise ValueError(f"affected tet spans {len(c)} variable components")
        row_comp[k]=c[0]
    global_to_touch={int(r):i for i,r in enumerate(Ttouch)}
    active=set(map(int,active.tolist()))
    seed_global=np.flatnonzero(np.isin(T,hnode).any(axis=1))
    seed_global=np.intersect1d(seed_global,Ttouch,assume_unique=False)
    active.update(map(int,seed_global.tolist()))
    # Recompute initial full affected quality after F32 emission and add all detected rows.
    def scan_full(Xmm):
        q64=quality_mm(Xmm,T,Ttouch)
        Xf32=(np.asarray(Xmm/1000.,np.float32).astype(np.float64))*1000.
        q32=quality_mm(Xf32,T,Ttouch)
        low=Ttouch[(q64<QACTIVE)|(q32<QACTIVE)]
        return q64,q32,np.asarray(low,np.int64)
    qi64,qi32,low=scan_full(current);active.update(map(int,low))
    np.savez_compressed(OUT/"initial-component-state.npz",x_mm=current[Vrows],variable_local_rows=Vrows,
                        affected_tet_rows=Ttouch,initial_active_tet_rows=np.asarray(sorted(active),np.int64),
                        q64=qi64,q_f32=qi32,component_id_by_variable=comp_of_v)
    poses=json.loads(POSE.read_text())["all_bodies"]
    donor_rows=list(csv.DictReader(open(DONOR,newline="")))
    assert len(donor_rows)==2411
    # Surface objective edge pairs map variable IDs; duplicate undirected edges removed.
    edges=np.asarray(sorted(edge_set),np.int64)
    if len(edges): edges=np.unique(edges,axis=0)
    penalty=np.choose(Vrows//2411,[4.,2.,1.])
    # Build per-component bone and layer row lists.
    node_to_comp={int(n):int(comp_of_node[n]) for n in Vrows}
    bone_comp=np.asarray([node_to_comp[int(n)] for n in b["node"]],np.int64)
    for n in b["node"]:
        if int(n) not in node_to_comp: raise ValueError(f"bone witness row not variable: {n}")
    layer_comp=np.asarray([col_to_v[int(c)] and int(comp_of_v[col_to_v[int(c)][0]]) for c in ring],np.int64)
    if np.any(layer_comp<0): raise ValueError("layer rows missing variable component")
    # Physical layer normals and minimum construction floor in mm.
    O0,I0,D0=X0[:2411],X0[2411:4822],X0[4822:]
    sv=O0[ring]-I0[ring];fv=I0[ring]-D0[ring]
    sl=np.linalg.norm(sv,axis=1);fl=np.linalg.norm(fv,axis=1)
    if np.any(sl<=1e-8)|np.any(fl<=1e-8): raise ValueError("degenerate reference layer column")
    sn=sv/sl[:,None];fn=fv/fl[:,None]
    sfloor=layer_floor_mm(sl);ffloor=layer_floor_mm(fl)
    # At baseline we use 034 candidate iterate, but reject if it violates corrected floor beyond rounding.
    start32=(current/1000.).astype(np.float32).astype(np.float64)*1000.
    sb=np.einsum("ij,ij->i",start32[ring]-start32[2411+ring],sn)
    fb=np.einsum("ij,ij->i",start32[2411+ring]-start32[4822+ring],fn)
    # The 034 restart may violate corrected layer floors or q barriers; retain it as an infeasible warm start.
    initial_skin_slack=float((sb-sfloor).min());initial_fat_slack=float((fb-ffloor).min())
    # Plane row mapping and source checks already done in affine_planes.
    start_quality, start_f32_q, _=scan_full(current)
    baseline_q64=quality_mm(allbase_mm,T,Ttouch); baseline_q32=quality_mm((allbase_mm/1000.).astype(np.float32).astype(np.float64)*1000.,T,Ttouch)
    if np.any(baseline_q64<=0) or np.any(baseline_q32<=0): raise ValueError("baseline geometry has nonpositive affected cell")
    # record immutable setup before optimization
    affine_report=json.loads(AFF_REPORT.read_text())
    report={"schema":"numi.human.d37-componentwise-bone-clearance-fit.v1","status":"running",
            "input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in PINFILES},
            "affine_receipt_sha256":sha(AFF),"affine_report_sha256":sha(AFF_REPORT),"covector_plane_receipt_sha256":sha(COV),
            "affine":affine_report,"constraints":{"inside_rows":int(H),"distinct_nodes":int(len(np.unique(b["node"]))),
            "bodies":int(len(np.unique(b["body"]))),"bones":int(len(np.unique(b["bone"]))),
            "min_baseline_plane_gap_m":float(b["baseline_gap_m"].min()),"margin_m":margin_m,
            "max_difference_roundtrip_m":float(b["difference_roundtrip_m"].max()),
            "plane_normal":"normalized M_conjugate.T @ unit(closest_local-query_local); inverse-linear escape vector explicitly not used"},
            "components":{"node_counts":counts,"dof_counts":[3*x for x in counts],"affected_tet_counts":[int(np.sum(row_comp==i)) for i in range(len(groups))],
            "exact_separability":"quality tets, O-edge objective, and same-column layer rows all lie within one component; no global mass term"},
            "quality":{"floor":QF,"active_threshold":QACTIVE,"affected_cells":len(Ttouch),"initial_active_count":len(active),"direct_witness_seed_tets":int(len(seed_global)),
            "initial_restart_nonpositive_f64":int(np.sum(start_quality<=0)),"initial_restart_nonpositive_f32":int(np.sum(start_f32_q<=0)),"baseline_nonpositive_f64":int(np.sum(baseline_q64<=0)),"baseline_nonpositive_f32":int(np.sum(baseline_q32<=0)),"initial_restart_low_rows":int(np.sum((start_quality<QACTIVE)|(start_f32_q<QACTIVE)))},
            "layer":{"floor":"min(0.2*baseline thickness in mm, 0.10 mm)","construction_bound_only":True,"physiological_limits":"pending",
            "warm_start_skin_min_slack_mm":initial_skin_slack,"warm_start_fat_min_slack_mm":initial_fat_slack},
            "budget_s":BUDGET,"component_solve":[]}
    (OUT/"fit-progress.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    vm_nodes=Vrows
    xbase=allbase_mm[vm_nodes].copy()
    xcur=current[vm_nodes].copy()
    # Efficient global quality scan over every affected row at each accepted callback.
    if __import__("os").environ.get("FIT_PREFLIGHT_ONLY")=="1":
        print("PREFLIGHT_PASS",json.dumps({"inside":H,"variable_nodes":len(Vrows),"components":counts,"affected_tets":len(Ttouch),"initial_active":len(active),"direct_seed_tets":len(seed_global),"max_affine_roundtrip_m":float(b["difference_roundtrip_m"].max()),"min_baseline_gap_m":float(b["baseline_gap_m"].min())}),flush=True)
        return
    global_start=time.monotonic();allscans=0;newrows_by_comp=[set() for _ in groups]
    def make_full_from_local(ci,localx):
        xx=current.copy()
        g=groups[ci]; nodes=Vrows[g]
        xx[nodes]=np.asarray(localx).reshape(-1,3)
        return xx
    class RestartForRows(Exception):pass
    class BoundedStop(Exception):pass
    for ci,g in enumerate(groups):
        if time.monotonic()-global_start>=BUDGET: break
        gv=Vrows[g];nd=3*len(g);loc_of=np.full(len(X0),-1,np.int64);loc_of[gv]=np.arange(len(g))
        # Component's affected tetrahedron rows and active subset.
        local_touch=np.flatnonzero(row_comp==ci);global_tets=Ttouch[local_touch]
        local_active=lambda: np.asarray([r for r in sorted(active) if row_comp[global_to_touch.get(int(r),-1)]==ci],np.int64)
        # Linear bone rows in this component.
        bi=np.flatnonzero(bone_comp==ci);bnode=b["node"][bi];bnorm=b["normal_world"][bi]
        bclosest=b["closest_world_m"][bi]*1000.
        # Layer rows for columns in this component.
        li=np.flatnonzero(layer_comp==ci);cols=ring[li]
        snc=sn[li];fnc=fn[li];sfc=sfloor[li];ffc=ffloor[li]
        # Objective edge list with at least one variable endpoint.
        el=[]
        for na,nb in edges:
            ca=int(comp_of_node[na]);cb=int(comp_of_node[nb])
            if ca==ci or cb==ci:
                if ca not in (-1,ci) or cb not in (-1,ci):raise ValueError(f"objective edge crosses components ci={ci} nodes={(int(na),int(nb))} comps={(ca,cb)}")
                el.append((int(na),int(nb)))
        el=np.asarray(el,np.int64).reshape(-1,2)
        def assemble(localx):
            xx=current.copy();xx[gv]=np.asarray(localx).reshape(-1,3)
            return xx
        def objective(localx):
            x=np.asarray(localx).reshape(-1,3);dx=x-xbase[g]
            val=float(np.sum(penalty[g,None]*dx*dx))
            if len(el):
                da=np.zeros((len(el),3));db=np.zeros_like(da)
                aa,bb=el.T
                ma=loc_of[aa]>=0;mb=loc_of[bb]>=0
                da[ma]=x[loc_of[aa[ma]]]-xbase[vm[aa[ma]]]
                db[mb]=x[loc_of[bb[mb]]]-xbase[vm[bb[mb]]]
                diff=da-db;val+=.08*float(np.sum(diff*diff))
            return val
        def objective_jac(localx):
            x=np.asarray(localx).reshape(-1,3);dx=x-xbase[g];grad=2*penalty[g,None]*dx
            if len(el):
                da=np.zeros((len(el),3));db=np.zeros_like(da);aa,bb=el.T
                ma=loc_of[aa]>=0;mb=loc_of[bb]>=0
                da[ma]=x[loc_of[aa[ma]]]-xbase[vm[aa[ma]]]
                db[mb]=x[loc_of[bb[mb]]]-xbase[vm[bb[mb]]]
                z=.16*(da-db)
                np.add.at(grad,loc_of[aa[ma]],z[ma]);np.add.at(grad,loc_of[bb[mb]],-z[mb])
            return grad.ravel()
        def constraints_fun(localx, need_jac=False, active_rows=None):
            xx=assemble(localx)
            rows=local_active() if active_rows is None else np.asarray(active_rows,np.int64)
            if len(rows):
                V=xx[T[rows]]
                q,G=q_and_grad(V)
                if need_jac:
                    J=np.zeros((len(rows),nd),np.float64)
                    for k in range(4):
                        nodes=T[rows,k];ix=loc_of[nodes];ok=ix>=0
                        if np.any(ok):
                            rr=np.flatnonzero(ok)
                            for axis in range(3): np.add.at(J,(rr,3*ix[ok]+axis),G[ok,k,axis])
            else:
                q=np.empty(0,np.float64)
                J=np.zeros((0,nd),np.float64)
            # Bone-plane margins, in mm.
            pb=np.einsum("ij,ij->i",xx[bnode]-bclosest,bnorm)-margin_mm
            # Layer floors in mm, including fixed O endpoints.
            so=np.einsum("ij,ij->i",xx[cols]-xx[2411+cols],snc)-sfc
            fa=np.einsum("ij,ij->i",xx[2411+cols]-xx[4822+cols],fnc)-ffc
            vals=np.r_[q*1e5-QF*1e5,pb,so,fa]
            if not need_jac:return vals
            JL=np.zeros((len(pb)+len(so)+len(fa),nd),np.float64)
            for k,n in enumerate(bnode):
                ix=loc_of[n]
                JL[k,3*ix:3*ix+3]=bnorm[k]
            skin0=len(pb);fat0=len(pb)+len(so)
            for k,c in enumerate(cols):
                io,ii,di=loc_of[c],loc_of[2411+int(c)],loc_of[4822+int(c)]
                if io>=0:JL[skin0+k,3*io:3*io+3]+=snc[k]
                if ii>=0:JL[skin0+k,3*ii:3*ii+3]-=snc[k]
                if ii>=0:JL[fat0+k,3*ii:3*ii+3]+=fnc[k]
                if di>=0:JL[fat0+k,3*di:3*di+3]-=fnc[k]
            return np.vstack((J*1e5,JL))
        attempt=0;comp_start=time.monotonic();done=False
        while attempt<5 and time.monotonic()-global_start<BUDGET:
            active_rows=local_active()
            x0loc=current[gv].ravel().copy()
            # Verify exact constraint partition and Jacobian dimensions at each restart.
            vals=constraints_fun(x0loc,False,active_rows)
            jac=constraints_fun(x0loc,True,active_rows)
            if jac.shape!=(len(vals),nd):raise ValueError("component constraint Jacobian shape mismatch")
            last={"x":x0loc.copy()};callback_count=0;force_restart={"yes":False}
            def callback(x):
                nonlocal callback_count
                callback_count+=1;last["x"]=x.copy()
                nonlocal allscans
                xx=assemble(x)
                q64=quality_mm(xx,T,Ttouch)
                xf32=(xx/1000.).astype(np.float32).astype(np.float64)*1000.
                q32=quality_mm(xf32,T,Ttouch)
                allscans+=1
                low=Ttouch[(q64<QACTIVE)|(q32<QACTIVE)]
                newly=np.asarray(sorted(set(map(int,low))-active),np.int64)
                if len(newly):
                    active.update(map(int,newly))
                    for rr in newly:
                        jj=global_to_touch[int(rr)];newrows_by_comp[int(row_comp[jj])].add(int(rr))
                    if any(int(row_comp[global_to_touch[int(rr)]])==ci for rr in newly):
                        force_restart["yes"]=True
                        raise RestartForRows()
                if time.monotonic()-global_start>BUDGET:raise BoundedStop()
                if callback_count%2==0:
                    np.savez_compressed(OUT/"solver-checkpoint.npz",points_mm=xx[vm_nodes],variable_local_rows=vm_nodes,
                        active_global_tet_rows=np.asarray(sorted(active),np.int64),component_index=np.asarray(ci),attempt=np.asarray(attempt))
                    report["component_solve"].append({"component":ci,"nodes":int(len(g)),"active_quality_rows":int(len(active_rows)),
                        "callback":callback_count,"elapsed_global_s":time.monotonic()-global_start,"objective":objective(x)})
                    report["status"]="running";(OUT/"fit-progress.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
            # Verify the actual assembled objective, q, and linear-constraint Jacobians once before any fit.
            if ci==0 and attempt==0:
                hfd=1e-3
                oj=objective_jac(x0loc)
                od=int(np.argmax(np.abs(oj)))
                ep=np.zeros(nd);ep[od]=hfd
                ofd=(objective(x0loc+ep)-objective(x0loc-ep))/(2*hfd)
                oerr=abs(ofd-oj[od])/max(1.0,abs(ofd),abs(oj[od]))
                if oerr>2e-6: raise ValueError(f"assembled objective Jacobian FD mismatch {oerr}")
                if len(active_rows):
                    Jq=jac[:len(active_rows)]
                    qi,qd=np.unravel_index(np.argmax(np.abs(Jq)),Jq.shape)
                    ep=np.zeros(nd);ep[qd]=hfd
                    qfd=(constraints_fun(x0loc+ep,False,active_rows)[qi]-constraints_fun(x0loc-ep,False,active_rows)[qi])/(2*hfd)
                    qerr=abs(qfd-Jq[qi,qd])/max(1.0,abs(qfd),abs(Jq[qi,qd]))
                    if qerr>2e-5: raise ValueError(f"assembled q Jacobian FD mismatch {qerr}")
                else:qerr=0.0
                Jlin=jac[len(active_rows):]
                linvals=constraints_fun(x0loc,False,active_rows)[len(active_rows):]
                linear_errors={}
                starts={"bone":(0,len(bnode)),"skin":(len(bnode),len(bnode)+len(cols)),"fat":(len(bnode)+len(cols),len(bnode)+2*len(cols))}
                basevals=constraints_fun(x0loc,False,active_rows)
                for label,(r0,r1) in starts.items():
                    if r1<=r0: continue
                    localJ=Jlin[r0:r1]
                    lr,ld=np.unravel_index(np.argmax(np.abs(localJ)),localJ.shape)
                    ep=np.zeros(nd);ep[ld]=hfd;v0=len(active_rows)+r0+lr
                    lfd=(constraints_fun(x0loc+ep,False,active_rows)[v0]-constraints_fun(x0loc-ep,False,active_rows)[v0])/(2*hfd)
                    le=abs(lfd-localJ[lr,ld])/max(1.0,abs(lfd),abs(localJ[lr,ld]))
                    linear_errors[label]=le
                    if le>2e-7: raise ValueError(f"assembled {label} Jacobian FD mismatch {le}")
                report["jacobian_preflight"]={"objective_fd_relative_error":oerr,"quality_fd_relative_error":qerr,"linear_fd_relative_error_by_class":linear_errors,"quality_scale":1e5,"finite_difference_step_mm":hfd}
                print("JACOBIAN_PREFLIGHT",json.dumps(report["jacobian_preflight"]),flush=True)
                if __import__("os").environ.get("FIT_JACOBIAN_PREFLIGHT_ONLY")=="1": return
            try:
                result=minimize(objective,x0loc,jac=objective_jac,method="SLSQP",
                    bounds=[(float(v-15.),float(v+15.)) for v in xbase[g].ravel()],
                    constraints=[{"type":"ineq","fun":lambda x,ar=active_rows:constraints_fun(x,False,ar),
                                  "jac":lambda x,ar=active_rows:constraints_fun(x,True,ar)}],
                    callback=callback,options={"maxiter":70,"ftol":2e-7,"disp":False})
                accepted=np.asarray(result.x,np.float64)
                result_summary={"success":bool(result.success),"status":int(result.status),"message":str(result.message),"nit":int(result.nit),"fun":float(result.fun)}
            except RestartForRows:
                accepted=last["x"]
                result_summary={"success":False,"status":-3,"message":"new globally scanned quality rows added; restart this component","nit":callback_count,"fun":objective(accepted)}
            except BoundedStop:
                accepted=last["x"]
                result_summary={"success":False,"status":-2,"message":"global wall-clock budget reached; retained last accepted callback iterate","nit":callback_count,"fun":objective(accepted)}
            current[gv]=accepted.reshape(-1,3)
            # All affected cells are scanned after every solver exit as well.
            q64=quality_mm(current,T,Ttouch);xf32=(current/1000.).astype(np.float32).astype(np.float64)*1000.;q32=quality_mm(xf32,T,Ttouch)
            low=Ttouch[(q64<QACTIVE)|(q32<QACTIVE)]
            newly=set(map(int,low))-active
            new_current=any(int(row_comp[global_to_touch[rr]])==ci for rr in newly)
            if newly:
                active.update(newly)
                for rr in newly:newrows_by_comp[int(row_comp[global_to_touch[rr]])].add(rr)
            localbad=np.flatnonzero((q64[local_touch]<QACTIVE)|(q32[local_touch]<QACTIVE))
            entry={"component":ci,"attempt":attempt,"nodes":int(len(g)),"dofs":int(nd),"affected_tets":int(len(local_touch)),
                   "active_constraints":int(len(active_rows)),"solver":result_summary,
                   "global_scan_count":allscans,"global_elapsed_s":time.monotonic()-global_start,
                   "component_low_quality_after":int(len(localbad)),"new_active_rows":sorted(newrows_by_comp[ci]),
                   "min_q64_affected":float(q64[local_touch].min()),"min_q32_affected":float(q32[local_touch].min()),
                   "min_plane_or_layer_slack":float(constraints_fun(current[gv].ravel(),False,local_active())[len(local_active()):].min()) if len(local_active()) else float("nan")}
            report["component_solve"].append(entry)
            print("COMPONENT",json.dumps(entry),flush=True)
            if result_summary["status"]==-2: report["status"]="bounded_partial";done=True;break
            if new_current:
                attempt+=1;continue
            if np.any(np.asarray(constraints_fun(current[gv].ravel(),False,local_active())) < -2e-8):
                if result_summary["status"]==-3:
                    attempt+=1;continue
                entry["component_candidate_constraints_pass"]=False
                attempt+=1
                if not result_summary["success"]:break
            elif result_summary["status"]==-3:
                attempt+=1;continue
            else:
                done=True
                break
            attempt+=1
        if not done and report["status"]=="running":
            report["status"]="component_solver_incomplete"
        if time.monotonic()-global_start>BUDGET:break
    # Complete independent final gates, regardless of optimizer completion.
    Xcand=current/1000.0
    gate=full_metrics(Xcand,T,material)
    emitted=(Xcand.astype(np.float32).astype(np.float64))
    qTouch64=quality_mm(current,T,Ttouch); qTouch32=quality_mm(emitted*1000.,T,Ttouch)
    final_low=Ttouch[(qTouch64<QACTIVE)|(qTouch32<QACTIVE)]
    gap=np.einsum("ij,ij->i",emitted[b["node"]]-b["closest_world_m"],b["normal_world"])
    Xo,Xi,Xd=emitted[:2411],emitted[2411:4822],emitted[4822:]
    sproj=1000*np.einsum("ij,ij->i",Xo[ring]-Xi[ring],sn)
    fproj=1000*np.einsum("ij,ij->i",Xi[ring]-Xd[ring],fn)
    layerpass=bool(np.all(sproj>=sfloor-2e-5)&np.all(fproj>=ffloor-2e-5))
    plane_pass=bool(np.all(gap>=margin_m-2e-7))
    base=full_metrics(X0,T,material)
    moved=np.flatnonzero(np.linalg.norm(emitted-X0,axis=1)>0)
    completed_components=sorted(set(int(e["component"]) for e in report["component_solve"] if "attempt" in e and e.get("component_candidate_constraints_pass",True) and e.get("solver",{}).get("success",False)))
    final_status="candidate_only_not_admitted" if len(completed_components)==len(groups) else "bounded_partial_or_failed_fit"
    report.update({"status":final_status,
        "elapsed_s":time.monotonic()-global_start,"callback_global_quality_scans":allscans,"components_with_solver_success":completed_components,"components_expected":len(groups),
        "final":{"all_cell_metrics_float32_emitted":gate,"baseline_metrics":base,
            "bone_plane":{"minimum_gap_m":float(gap.min()),"target_margin_m":margin_m,"violations_tolerance_2e-7m":int(np.sum(gap<margin_m-2e-7)),
                          "pass_with_declared_tolerance":plane_pass},
            "layer":{"minimum_skin_projection_mm":float(sproj.min()),"minimum_fat_projection_mm":float(fproj.min()),
                    "skin_floor_mm_min":float(sfloor.min()),"fat_floor_mm_min":float(ffloor.min()),
                    "violations_skin":int(np.sum(sproj<sfloor-2e-5)),"violations_fat":int(np.sum(fproj<ffloor-2e-5)),
                    "pass_with_20nm_emission_tolerance":layerpass,"physiological_thickness":"not qualified"},
            "affected_quality":{"min_float64":float(qTouch64.min()),"min_emitted_float32":float(qTouch32.min()),
                               "low_rows_global":final_low.astype(int).tolist(),"active_row_count":len(active)},
            "movement":{"moved_node_count":int(len(moved)),"max_displacement_m":float(np.linalg.norm(emitted-X0,axis=1).max()),
                        "moved_layer_counts":{str(k):int(np.sum(moved//2411==k)) for k in range(3)}}},
        "limits":["Bone planes are necessary constraints, not final exact solid-containment verification.",
                  "No fresh self/retained-surface audit or native emission replay has run.",
                  "Detailed-source embedding, source ancestry and conservative mass mapping are not regenerated.",
                  "The construction layer floors are feasibility bounds only; physiological thickness qualification remains pending.",
                  "Any donor-local deltas are diagnostic until native emission roundtrip is separately verified."]})
    np.savez_compressed(OUT/"candidate.npz",points_world_f32=emitted.astype(np.float32),tetrahedra=T.astype(np.int32),
        material_ids=material.astype(np.int8),global_node_ids=np.asarray(ap["global_nodes"]),variable_local_rows=Vrows,
        component_id_by_variable=comp_of_v,witness_local_rows=b["node"],witness_body_index=b["body"],witness_bone_stable_id=b["bone"],
        witness_nearest_triangle=b["triangle"],witness_closest_world_m=b["closest_world_m"],witness_normal_world=b["normal_world"],
        witness_final_plane_gap_m=gap,variable_start_mm=xrestart,variable_final_mm=current[Vrows],
        ring_columns=ring,skin_layer_projection_mm=sproj,fat_layer_projection_mm=fproj,
        skin_layer_floor_mm=sfloor,fat_layer_floor_mm=ffloor,active_global_tet_rows=np.asarray(sorted(active),np.int64),
        affected_global_tet_rows=Ttouch,affected_q_f64=qTouch64,affected_q_emitted_f32=qTouch32)
    report["output_sha256"]=sha(OUT/"candidate.npz")
    (OUT/"fit-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("FINAL",json.dumps({"output":str(OUT/"candidate.npz"),"sha256":report["output_sha256"],"status":report["status"],"elapsed_s":report["elapsed_s"],"final":report["final"],"component_entries":len(report["component_solve"])},indent=2),flush=True)

if __name__=="__main__": main()
