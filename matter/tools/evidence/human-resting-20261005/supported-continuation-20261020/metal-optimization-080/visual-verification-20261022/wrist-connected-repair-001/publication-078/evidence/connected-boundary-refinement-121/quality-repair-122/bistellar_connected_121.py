#!/usr/bin/env python3
"""Bounded same-material 2-3/3-2/4-4 flips from the frozen original cavity baseline."""
from __future__ import annotations
import hashlib, itertools, json, socket, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path("/Users/numi/Work/numi-human-connected-surface-069/connected-boundary-refinement-121")
CANONICAL = Path("/Users/numi/Work/numi-human-connected-surface-069/canonical/matter/tools/human_skin_fat_connected_repair.py")
SOURCE = BASE / "run-003/tetgen-reference-identity.npz"
OUT = HERE / "run-002"
spec = __import__("importlib.util").util.spec_from_file_location("canonical_repair", CANONICAL)
repair = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(repair)
Q_FLOOR = 1e-5
STANDARD_J = (0.35, 2.5)
J_HARD_MAX = 50.0
MAX_SWEEPS = 4
MAX_SECONDS = 70.0
MAX_ACCEPTED_FLIPS = 10


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def dets(points,tets):
    x=np.asarray(points,dtype=np.float64)[np.asarray(tets,dtype=np.int64)]
    return np.einsum("ij,ij->i",x[:,1]-x[:,0],np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]))


def orient_ref(tets,Xr):
    t=np.asarray(tets,dtype=np.int64).copy()
    d=dets(Xr,t)
    if np.any(d==0): raise ValueError("zero reference determinant")
    neg=d<0
    if np.any(neg):
        t[neg,2],t[neg,3]=t[neg,3].copy(),t[neg,2].copy()
    if np.any(dets(Xr,t)<=0): raise ValueError("could not orient cell from trusted reference")
    return t


def sign3(v):
    ix=[sorted(v).index(int(x)) for x in v]
    return -1 if sum(ix[i]>ix[j] for i in range(3) for j in range(i+1,3))%2 else 1


def boundary_chain(tets,Xr):
    t=orient_ref(tets,Xr); c=Counter()
    for a,b,c0,d in t:
        for f in ((b,c0,d),(a,d,c0),(a,b,d),(a,c0,b)):
            c[tuple(sorted(map(int,f)))]+=sign3(f)
    return {k:v for k,v in c.items() if v}


def boundary_face_counts(tets):
    c=Counter()
    for tet in tets:
        a,b,d,e=map(int,tet)
        for f in ((b,d,e),(a,e,d),(a,b,e),(a,d,b)):
            c[tuple(sorted(f))]+=1
    return c


def make_topology(tets,Xr):
    face_inc=defaultdict(list); edge_inc=defaultdict(set); cell_owners=defaultdict(set)
    oriented=orient_ref(tets,Xr)
    for ci,tet in enumerate(oriented):
        a,b,c,d=map(int,tet)
        for f,o in (((b,c,d),a),((a,d,c),b),((a,b,d),c),((a,c,b),d)):
            key=tuple(sorted(f))
            face_inc[key].append((ci,tuple(map(int,f)),int(o),sign3(f)))
        for u,v in itertools.combinations(map(int,tet),2):
            edge_inc[tuple(sorted((u,v)))].add(ci)
        cell_owners[tuple(sorted(map(int,tet)))].add(ci)
    return face_inc,edge_inc,cell_owners


def line_crosses_face(X,u,v,face):
    a,b,c=np.asarray(X,dtype=np.float64)[np.asarray(face,dtype=np.int64)]
    pu,pv=np.asarray(X)[int(u)],np.asarray(X)[int(v)]
    n=np.cross(b-a,c-a); du=float(n@(pu-a)); dv=float(n@(pv-a))
    if abs(du-dv)<1e-30 or du*dv>=0: return False
    s=du/(du-dv)
    if not 0<s<1:return False
    q=pu+s*(pv-pu)
    uv=np.linalg.lstsq(np.column_stack((b-a,c-a)),q-a,rcond=None)[0]
    bary=np.array([1-uv.sum(),uv[0],uv[1]])
    return bool(np.min(bary)>1e-10)


def segment_intersection_interior(X,a,b,c,d):
    # Closest points on two finite segments; require a robust interior crossing.
    X=np.asarray(X,dtype=np.float64)
    p0,p1,q0,q1=X[[a,b,c,d]]
    u=p1-p0; v=q1-q0; w=p0-q0
    A=float(u@u); B=float(u@v); C=float(v@v); D=float(u@w); E=float(v@w)
    den=A*C-B*B
    if den<=1e-30:return False
    s=(B*E-C*D)/den; t=(A*E-B*D)/den
    if not (1e-8<s<1-1e-8 and 1e-8<t<1-1e-8):return False
    dist=np.linalg.norm((p0+s*u)-(q0+t*v))
    scale=max(np.linalg.norm(u),np.linalg.norm(v),1e-12)
    return bool(dist<=max(1e-10,scale*1e-8))


def link_cycle(edge,cell_ids,T):
    u,v=map(int,edge); g=defaultdict(set)
    for ci in cell_ids:
        opp=[int(x) for x in T[ci] if int(x) not in (u,v)]
        if len(opp)!=2 or opp[0]==opp[1]:return None
        a,b=opp;g[a].add(b);g[b].add(a)
    if any(len(x)!=2 for x in g.values()):return None
    if len(g)!=len(cell_ids):return None
    start=min(g)
    n1,n2=sorted(g[start])
    def walk(first):
        out=[start,first]; prev,cur=start,first
        while len(out)<len(g):
            nxt=[x for x in g[cur] if x!=prev]
            if len(nxt)!=1:return None
            prev,cur=cur,nxt[0]
            if cur==start:return None
            out.append(cur)
        return out if start in g[out[-1]] else None
    c1=walk(n1);c2=walk(n2)
    if c1 is None or c2 is None:return None
    return min(tuple(c1),tuple(c2))


def candidate_specs(T,face_inc,edge_inc,bad_ids):
    seen=set()
    # 2 -> 3 across an internal face.
    for f,inc in face_inc.items():
        if len(inc)!=2:continue
        i,fi,u,si=inc[0];j,fj,v,sj=inc[1]
        if (i not in bad_ids and j not in bad_ids) or si==sj:continue
        key=("2to3",tuple(sorted((i,j))),f)
        if key not in seen:
            seen.add(key);yield "2to3",(i,j),f,(u,v)
    # 3 -> 2 at a valence-3 edge.
    for e,ids0 in edge_inc.items():
        ids=sorted(ids0)
        if len(ids)!=3 or not (set(ids)&bad_ids):continue
        cyc=link_cycle(e,ids,T)
        if cyc is None or len(cyc)!=3:continue
        key=("3to2",tuple(ids),e)
        if key not in seen:
            seen.add(key);yield "3to2",tuple(ids),e,cyc
    # 4 -> 4 at a valence-4 edge; try either alternate octahedron diagonal.
    for e,ids0 in edge_inc.items():
        ids=sorted(ids0)
        if len(ids)!=4 or not (set(ids)&bad_ids):continue
        cyc=link_cycle(e,ids,T)
        if cyc is None or len(cyc)!=4:continue
        for diagonal in (0,1):
            key=("4to4",tuple(ids),e,diagonal)
            if key not in seen:
                seen.add(key);yield "4to4",(tuple(ids),diagonal),e,cyc


def raw_new_tets(kind, face_or_edge, opposites_or_cycle, alt=0):
    if kind=="2to3":
        a,b,c=map(int,face_or_edge);u,v=map(int,opposites_or_cycle)
        return np.array([[u,v,a,b],[u,v,b,c],[u,v,c,a]],dtype=np.int64)
    if kind=="3to2":
        u,v=map(int,face_or_edge);a,b,c=map(int,opposites_or_cycle)
        return np.array([[u,a,b,c],[v,a,c,b]],dtype=np.int64)
    if kind=="4to4":
        old_edge=tuple(map(int,face_or_edge));cycle=tuple(map(int,opposites_or_cycle))
        u,v=old_edge
        if alt==0:
            a,b,c,d=cycle
        else:
            b,c,d,a=cycle
        return np.array([[a,c,u,b],[a,c,b,v],[a,c,v,d],[a,c,d,u]],dtype=np.int64)
    raise ValueError(kind)


def face_count_compatible(old_ids,new_tets,T,global_face_counts):
    old=boundary_face_counts(T[list(old_ids)])
    new=boundary_face_counts(new_tets)
    # Every face's global multiplicity after replacing the cavity must be 1 or 2.
    for f in set(old)|set(new):
        after=global_face_counts.get(f,0)-old.get(f,0)+new.get(f,0)
        if after<0 or after>2:return False
    return True


def quality_values(Xr32,Xc32,T):
    # Reuse the canonical owner q=det/Lmax^3 and determinant/J equations.
    jr,_,qr,_=repair.cell_values_gradients(Xr32,Xr32,T)
    jc,_,qc,_=repair.cell_values_gradients(Xc32,Xr32,T)
    dr=dets(Xr32,T);dc=dets(Xc32,T)
    bad=(dr<=0)|(dc<=0)|(qr<Q_FLOOR)|(qc<Q_FLOOR)
    return dr,dc,jc,qr,qc,bad


def eval_cavity(kind,old_ids,feature,new_raw,T,mats,Xr64,Xr32,Xc32,edge_inc,cell_owners,face_counts):
    old_ids=tuple(map(int,old_ids))
    if len(set(int(mats[i]) for i in old_ids))!=1:return None,"material_mismatch"
    old_t=T[list(old_ids)]
    if np.any(dets(Xr64,new_raw)==0):return None,"reference_degenerate"
    try:
        new=orient_ref(new_raw,Xr64)
    except ValueError:
        return None,"reference_degenerate"
    if boundary_chain(old_t,Xr64)!=boundary_chain(new,Xr64):return None,"oriented_boundary_mismatch"
    keys=[tuple(sorted(map(int,t))) for t in new]
    if len(set(keys))!=len(keys):return None,"duplicate_new_cell"
    if any(cell_owners.get(k,set())-set(old_ids) for k in keys):return None,"duplicate_existing_cell"
    if not face_count_compatible(old_ids,new,T,face_counts):return None,"postflip_face_incidence"
    if kind=="2to3":
        face=tuple(map(int,feature));u,v=int(new_raw[0,0]),int(new_raw[0,1])
        if not line_crosses_face(Xr64,u,v,face):return None,"nonconvex_reference_bipyramid"
        if edge_inc.get(tuple(sorted((u,v))),set())-set(old_ids):return None,"new_edge_exists_outside_cavity"
    elif kind=="3to2":
        u,v=map(int,feature);link=sorted(set(map(int,old_t.ravel()))-set((u,v)))
        if len(link)!=3 or not line_crosses_face(Xr64,u,v,link):return None,"nonconvex_reference_edge_collapse"
    elif kind=="4to4":
        newedge=set.intersection(*(set(map(int,t)) for t in new))
        if len(newedge)!=2:return None,"invalid_new_diagonal"
        a,c=sorted(newedge);oldedge=set(map(int,feature))
        if len(oldedge)!=2:return None,"invalid_old_diagonal"
        if edge_inc.get(tuple(sorted((a,c))),set())-set(old_ids):return None,"new_edge_exists_outside_cavity"
    # The canonical owner metric is evaluated on emitted Float32 coordinates.
    _,_,_,qr32,qc32,new_bad=quality_values(Xr32,Xc32,new)
    dr64=dets(Xr64,new)
    if np.any(dr64<=0):return None,"reference_nonpositive_f64"
    if np.any(new_bad):return None,"canonical_reference_or_current_q_floor"
    old_dr,old_dc,old_j,old_qr,old_qc,old_bad_mask=quality_values(Xr32,Xc32,old_t)
    old_bad=int(np.count_nonzero(old_bad_mask))
    if old_bad<1:return None,"no_invalid_cell_in_cavity"
    new_j=repair.cell_values_gradients(Xc32,Xr32,new)[0]
    old_j=repair.cell_values_gradients(Xc32,Xr32,old_t)[0]
    pos_old=old_j[old_j>0]
    lower=min(STANDARD_J[0],float(pos_old.min()) if len(pos_old) else STANDARD_J[0])
    upper=min(J_HARD_MAX,max(STANDARD_J[1],float(old_j.max())))
    if np.any(new_j<lower) or np.any(new_j>upper):return None,"individual_J_bound"
    lo,hi=STANDARD_J
    old_out=int(np.count_nonzero((old_j<lo)|(old_j>hi)))
    new_out=int(np.count_nonzero((new_j<lo)|(new_j>hi)))
    if new_out>old_out:return None,"J_violation_count_increase"
    vold=float(dets(Xr64,old_t).sum()/6);vnew=float(dets(Xr64,new).sum()/6)
    if abs(vnew-vold)>max(1e-24,abs(vold)*1e-9):return None,"reference_volume_mismatch"
    qmin=float(min(qr32.min(),qc32.min()))
    score=(old_bad,new_out-old_out,qmin,-float(np.max(new_j)))
    return {"new_tets":new,"old_bad":old_bad,"old_j":old_j,"new_j":new_j,
            "old_j_out":old_out,"new_j_out":new_out,"ref_q32":qr32,"cur_q32":qc32,
            "ref_det32":dets(Xr32,new),"cur_det32":dets(Xc32,new),
            "volume_delta":vnew-vold,"score":score},None

def face_or_edge_from_old_tets(old_tets):
    # A 3->2 cavity has a unique common central edge.
    common=set(map(int,old_tets[0])) & set(map(int,old_tets[1]))
    for t in old_tets[2:]:
        common &= set(map(int,t))
    if len(common)!=2:raise ValueError("3->2 cavity lacks unique common edge")
    return tuple(sorted(common))


def summarize(T,Xr32,Xc32):
    dr,dc,j,qr,qc,bad=quality_values(Xr32,Xc32,T)
    return {"cells":len(T),"reference_nonpositive_f32":int(np.sum(dr<=0)),
            "reference_q_below_1e-5":int(np.sum(qr<Q_FLOOR)),
            "current_nonpositive_f32":int(np.sum(dc<=0)),
            "current_q_below_1e-5":int(np.sum(qc<Q_FLOOR)),
            "canonical_q_bad_union":int(np.count_nonzero(bad)),
            "canonical_q_min_reference":float(qr.min()),"canonical_q_min_current":float(qc.min()),
            "J_below_0.35":int(np.sum(j<STANDARD_J[0])),
            "J_above_2.5":int(np.sum(j>STANDARD_J[1])),"J_above_50":int(np.sum(j>J_HARD_MAX)),
            "Jmin":float(j.min()),"Jmax":float(j.max()),
            "bad_cell_rows":np.flatnonzero(bad).astype(int).tolist()}


def run():
    # Unit tetra smoke proves this run gates on the canonical det/Lmax^3 metric.
    smoke=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    sj,_,sq,_=repair.cell_values_gradients(smoke,smoke,np.array([[0,1,2,3]],dtype=np.int64))
    expected_q=1.0/(2.0**1.5)
    if abs(float(sq[0])-expected_q)>1e-14 or abs(float(sj[0])-1.)>1e-14:
        raise AssertionError("canonical owner metric smoke failed")
    t0=time.time();OUT.mkdir(parents=True,exist_ok=False)
    z=np.load(SOURCE);Xr64=np.asarray(z["reference_points_m"],dtype=np.float64)
    Xr32=Xr64.astype(np.float32).astype(np.float64)
    Xc32=np.asarray(z["points_world_f32"],dtype=np.float32).astype(np.float64)
    Traw=np.asarray(z["tetrahedra"],dtype=np.int64)
    T=orient_ref(Traw,Xr64)
    if not np.array_equal(T,Traw):raise ValueError("input cells unexpectedly require reference reorientation")
    mats=np.asarray(z["material_ids"],dtype=np.int64).copy()
    if Xr64.ndim!=2 or Xr64.shape[1]!=3 or Xc32.shape!=Xr64.shape or T.ndim!=2 or T.shape[1]!=4 or len(mats)!=len(T):
        raise ValueError("116 candidate dimensions changed")
    if set(np.unique(mats))!={1,2}:raise ValueError("unexpected material domain")
    surface_faces=np.asarray(z["surface_faces_zero_based"],dtype=np.int64)
    surface_markers=np.asarray(z["surface_face_markers"],dtype=np.int64)
    source_rows=np.asarray(z["source_input_row"],dtype=np.int64)
    if not (len(surface_faces)==len(surface_markers) and len(source_rows)==len(Xr64)):
        raise ValueError("116 node/surface identity domain changed")
    base=summarize(T,Xr32,Xc32)
    expected=[5068,5560,21368,33727]
    if base["bad_cell_rows"]!=expected:raise ValueError(f"unexpected canonical baseline q rows {base['bad_cell_rows']}")
    uids=np.asarray(z["cell_uids"],dtype=np.int64).copy()
    if len(np.unique(uids))!=len(uids):raise ValueError("input cell UIDs are not unique")
    next_uid=int(uids.max())+1;accepted=[]
    rejected=Counter();sweep_rows=[];pair_checks=0;deadline=t0+MAX_SECONDS
    while len(accepted)<MAX_ACCEPTED_FLIPS and time.time()<deadline:
        before=summarize(T,Xr32,Xc32)
        bad=set(before["bad_cell_rows"])
        if not bad:break
        face_inc,edge_inc,cell_owners=make_topology(T,Xr64)
        face_counts=Counter({f:len(inc) for f,inc in face_inc.items()})
        proposals=[];checked=0
        for kind,ids_arg,face_or_edge,opp_or_cycle in candidate_specs(T,face_inc,edge_inc,bad):
            if kind=="2to3":ids=ids_arg;raw=raw_new_tets(kind,face_or_edge,opp_or_cycle)
            elif kind=="3to2":ids=ids_arg;raw=raw_new_tets(kind,face_or_edge,opp_or_cycle)
            else:
                ids,alt=ids_arg;raw=raw_new_tets(kind,face_or_edge,opp_or_cycle,alt)
            result,reason=eval_cavity(kind,ids,face_or_edge,raw,T,mats,Xr64,Xr32,Xc32,edge_inc,cell_owners,face_counts)
            checked+=1;pair_checks+=1
            if reason:rejected[f"{kind}:{reason}"]+=1;continue
            proposals.append((result["score"],kind,tuple(ids),face_or_edge,opp_or_cycle,raw,result))
        if not proposals:
            sweep_rows.append({"accepted":0,"proposals_checked":checked,"seconds":time.time()-t0,"stalled_on_bad_rows":before["bad_cell_rows"]})
            break
        proposals.sort(key=lambda r:(r[0],r[1],tuple(map(int,np.ravel(r[3]))),tuple(map(int,np.ravel(r[4])))),reverse=True)
        score,kind,ids,feature,link,raw,res=proposals[0]
        mat=int(mats[ids[0]]);old_uid=[int(uids[i]) for i in ids]
        keep=np.ones(len(T),dtype=bool);keep[list(ids)]=False
        T=np.concatenate((T[keep],res["new_tets"]),axis=0)
        mats=np.concatenate((mats[keep],np.full(len(res["new_tets"]),mat,dtype=np.int64)))
        new_uid=list(range(next_uid,next_uid+len(res["new_tets"])));next_uid+=len(new_uid)
        uids=np.concatenate((uids[keep],np.asarray(new_uid,dtype=np.int64)))
        after=summarize(T,Xr32,Xc32)
        if after["canonical_q_bad_union"]>=before["canonical_q_bad_union"]:
            raise AssertionError("accepted bistellar move did not strictly reduce canonical q failures")
        if after["J_below_0.35"]+after["J_above_2.5"]>before["J_below_0.35"]+before["J_above_2.5"]:
            raise AssertionError("accepted move increased global standard-J violations")
        accepted.append({"kind":kind,"old_cell_uids":old_uid,"new_cell_uids":new_uid,"material_id":mat,
          "feature":list(map(int,np.ravel(feature))),"link_or_opposites":list(map(int,np.ravel(link))),
          "old_q_bad_cells":res["old_bad"],"new_ref_q":res["ref_q32"].tolist(),"new_current_q":res["cur_q32"].tolist(),
          "new_J":res["new_j"].tolist(),"reference_cavity_volume_delta_m3":res["volume_delta"],"before":before,"after":after})
        sweep_rows.append({"accepted":1,"proposals_checked":checked,"seconds":time.time()-t0,"chosen_kind":kind,"score":list(map(float,score))})
    # This bounded pass tests internal same-material bistellar flips only.
    collapse_attempt=None
    final=summarize(T,Xr32,Xc32)
    # Save original positions and locked PLC facets byte-for-byte; only topology/material/cell IDs can change.
    out=OUT/"candidate-tetrahedra.npz"
    payload={k:z[k] for k in z.files}
    payload["tetrahedra"]=T.astype(np.int64)
    payload["material_ids"]=mats.astype(np.int8)
    payload["cell_uids"]=uids.astype(np.int64)
    payload["points_world_f32"]=Xc32.astype(np.float32)
    np.savez_compressed(out,**payload)
    report={"status":"CANONICAL_Q_CONNECTIVITY_COUNTERFACTUAL_PASS" if final["canonical_q_bad_union"]==0 else "REJECTED_CANONICAL_Q_FLOOR_REMAINS",
      "classification":"connectivity_only_no_physical_admission","inputs":{"candidate_npz":str(SOURCE),"candidate_sha256":sha(SOURCE),
      "canonical_owner":str(CANONICAL),"canonical_owner_sha256":sha(CANONICAL)},
      "method":{"families":["2->3","3->2","4->4"],"coordinates_fixed":True,
      "outer_interface_and_side_facets_fixed":True,"same_material_only":True,"canonical_q":"determinant / longest current edge length cubed","q_floor":Q_FLOOR,
      "max_accepted_flips":MAX_ACCEPTED_FLIPS,"max_seconds":MAX_SECONDS,"reference_orientation":"trusted 084 Float32 reference only"},
      "baseline":base,"final":final,"accepted_flips":accepted,"sweeps":sweep_rows,"collapse_attempt":collapse_attempt,
      "rejections":dict(sorted(rejected.items())),"counts":{"accepted_flips":len(accepted),"candidate_cavities_checked":pair_checks,
      "accepted_collapse":bool(collapse_attempt and collapse_attempt.get("accepted"))},
      "invariants":{"all_node_coordinates_bitwise_unchanged":True,"marked_faces_and_markers_bitwise_unchanged":True,
      "O_I_D_faces_bitwise_unchanged":True},"output":{"path":str(out),"sha256":sha(out)},
      "limitations":["This repairs only internal connectivity/quality; it does not establish material mass transfer, global clearance, stress, native runtime, or physical qualification."],
      "runtime":{"host":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-t0,
        "script_sha256":sha(Path(__file__)),"unit_test_sha256":sha(HERE/"test_bistellar_121.py"),
        "input_sha256":sha(SOURCE)}}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"baseline":base,"final":final,"accepted_flips":len(accepted),"collapse_attempt":collapse_attempt,
      "rejections":dict(sorted(rejected.items())),"candidate":str(out),"candidate_sha256":sha(out),"report":str(OUT/"report.json"),"seconds":time.time()-t0},indent=2))


if __name__=="__main__":run()
