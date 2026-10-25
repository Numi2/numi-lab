#!/usr/bin/env python3
"""Nearest complete-bank boundary directions with exact first-union ray certification."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, sys, time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
DETAIL=D/"publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
BASE=D/"full-connected-clearance-150/run-028-current-target-connected-fit-001/candidate.npz"
ORIENTATION=D/"digit-source-diagnosis-110/source-path-orientation-audit-112.json"
ORIENTATION_BATCH=D/"digit-source-diagnosis-110/batch-145-curved-source-midpoints-004.json"
ORIENTATION_RUNNER=D/"digit-source-diagnosis-110/source_path_batch_110_v2.py"
ORIENTATION_COMPACT=D/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
ORIENTATION_TRANSFER=D/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz"
BANKHELPER=D/"publication-084/canonical/matter/tools/human_skin_fat_bone_audit.py"
EVIDENCE=Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
ORIENTATION_DETAIL=EVIDENCE/"bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
ORIENTATION_ROWS=EVIDENCE/"accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv"
BANK=EVIDENCE/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BANK_FIX=EVIDENCE/"route-audit-counter-fix-030/canonical/matter"
EXACT=EVIDENCE/"incremental-exact-audit-owner-253/src/numilab_human/cardiac_cavity_intersections.py"
SEGMENT_HELPER=BANK_FIX/"root_exact_segment_witness.py"
CONSTRAINT=D/"barycentric-virtual-point-170/human_skin_fat_virtual_point_constraints.py"
CLEARANCE_M=5e-5
STRICT_SAMPLE_INWARD_FRACTIONS=(.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625,.00048828125)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def loadmod(p,n):
 spec=importlib.util.spec_from_file_location(n,p)
 m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m

def json_default(o):
 if isinstance(o,np.ndarray):return o.tolist()
 if isinstance(o,np.generic):return o.item()
 if isinstance(o,Fraction):return {"numerator":str(o.numerator),"denominator":str(o.denominator),"value":float(o)}
 raise TypeError(type(o).__name__)

def csvrows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))

def ray_box(q,d,lo,hi):
 t0=-math.inf;t1=math.inf
 for k in range(3):
  if d[k]==0:
   if q[k]<lo[k] or q[k]>hi[k]:return None
   continue
  a=(lo[k]-q[k])/d[k];b=(hi[k]-q[k])/d[k]
  if a>b:a,b=b,a
  t0=max(t0,a);t1=min(t1,b)
  if t0>t1:return None
 if t1<0:return None
 return max(0.0,t0),t1

def first_empty_union_interval(initial_inside,events,clearance_m):
 grouped={}
 for ev in events:grouped.setdefault(float(ev["distance_m"]),[]).append(ev)
 distances=sorted(grouped);active=set(initial_inside)
 for i,dexit in enumerate(distances):
  toggles={}
  for ev in grouped[dexit]:toggles[ev["bank"]]=toggles.get(ev["bank"],0)+1
  for key,count in toggles.items():
   if count%2:
    if key in active:active.remove(key)
    else:active.add(key)
  if not active:
   nxt=distances[i+1] if i+1<len(distances) else None
   gap=math.inf if nxt is None else nxt-dexit
   if gap<=0:continue
   margin=clearance_m if math.isinf(gap) else min(clearance_m,.5*gap)
   if margin>0:return {"exit_distance_m":dexit,"target_distance_m":dexit+margin,"next_event_distance_m":nxt,"margin_m":margin,"free_gap_m":gap}
 return None

def check_first_union_fixture():
 r=first_empty_union_interval({"A"},[
  {"distance_m":.002,"bank":"A"},{"distance_m":.5,"bank":"B"},{"distance_m":.501,"bank":"B"}],CLEARANCE_M)
 assert r and abs(r["target_distance_m"]-.00205)<1e-15 and r["next_event_distance_m"]==.5

def world_triangles_from_local(triangles_local, world_to_local, body_position_world, query_offset_local):
    """Invert q=M(world-pos)-offset exactly for emitted local F32 triangles."""
    t=np.asarray(triangles_local,np.float64)
    m=np.asarray(world_to_local,np.float64)
    pos=np.asarray(body_position_world,np.float64)
    off=np.asarray(query_offset_local,np.float64)
    if t.ndim!=3 or t.shape[1:]!=(3,3) or m.shape!=(3,3) or pos.shape!=(3,) or off.shape!=(3,):
        raise ValueError("invalid bank affine/triangle shapes")
    if not all(np.all(np.isfinite(x)) for x in (t,m,pos,off)) or abs(float(np.linalg.det(m)))<1e-14:
        raise ValueError("nonfinite or singular bank affine")
    flat=t.reshape(-1,3)
    world=np.linalg.solve(m,(flat+off).T).T+pos
    return world.reshape(t.shape)

def closest_points_on_triangle_mesh(point_world, triangles_world, stable_ids):
    """Return deterministic closest triangle point; equal distances tie by stable ID."""
    p=np.asarray(point_world,np.float64); tri=np.asarray(triangles_world,np.float64); ids=np.asarray(stable_ids)
    if p.shape!=(3,) or tri.ndim!=3 or tri.shape[1:]!=(3,3) or ids.shape!=(len(tri),):
        raise ValueError("invalid closest-triangle inputs")
    if len(tri)==0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(tri)):
        raise ValueError("empty or nonfinite closest-triangle input")
    if not np.issubdtype(ids.dtype,np.integer):raise ValueError("stable IDs must be integers")
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    ab=b-a;ac=c-a;ap=p-a
    d00=np.einsum("ij,ij->i",ab,ab);d01=np.einsum("ij,ij->i",ab,ac);d11=np.einsum("ij,ij->i",ac,ac)
    d20=np.einsum("ij,ij->i",ap,ab);d21=np.einsum("ij,ij->i",ap,ac)
    den=d00*d11-d01*d01
    valid=(d00>0)&(d11>0)&(den>32*np.finfo(np.float64).eps*d00*d11)
    v=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=valid)
    w=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=valid)
    face=a+v[:,None]*ab+w[:,None]*ac
    face_valid=valid&(v>=0)&(w>=0)&(v+w<=1)
    cand=np.empty((len(tri),7,3),np.float64)
    cand[:,0]=a;cand[:,1]=b;cand[:,2]=c
    for slot,x,y in ((3,a,b),(4,b,c),(5,c,a)):
        e=y-x;ee=np.einsum("ij,ij->i",e,e)
        u=np.divide(np.einsum("ij,ij->i",p-x,e),ee,out=np.zeros_like(ee),where=ee>0)
        u=np.clip(u,0.0,1.0);cand[:,slot]=x+u[:,None]*e
    cand[:,6]=face
    delta=cand-p[None,None,:]
    dist2=np.einsum("ijk,ijk->ij",delta,delta)
    dist2[:,6]=np.where(face_valid,dist2[:,6],np.inf)
    local_slot=np.argmin(dist2,axis=1)
    rows=np.arange(len(tri));points=cand[rows,local_slot];best2=dist2[rows,local_slot]
    if not np.all(np.isfinite(best2)):raise ValueError("bank contains only degenerate triangle candidates")
    order=np.lexsort((np.arange(len(tri),dtype=np.int64),ids.astype(np.int64),best2))
    i=int(order[0])
    return {"point_world_m":points[i],"distance_m":float(math.sqrt(max(0.0,best2[i]))),
        "distance_squared_m2":float(best2[i]),"triangle_row":i,"triangle_stable_id":int(ids[i]),
        "tie_count":int(np.count_nonzero(best2==best2[i]))}

def closest_points_on_triangle_mesh_in_hemisphere(point_world, triangles_world, stable_ids, hemisphere_normal):
    """Closest point on triangles clipped by n·(x-point)>=0.

    The candidate set includes vertices, edge/face projections, triangle-edge
    intersections with the clipping plane, and the closest point on the clipped
    chord. The exact full-union ray/endpoint audit remains the clearance owner.
    """
    p=np.asarray(point_world,dtype=np.float64)
    tri=np.asarray(triangles_world,dtype=np.float64)
    ids=np.asarray(stable_ids)
    n=np.array(hemisphere_normal,dtype=np.float64,copy=True)
    if p.shape!=(3,) or tri.ndim!=3 or tri.shape[1:]!=(3,3) or ids.shape!=(len(tri),):
        raise ValueError("invalid hemisphere closest-point inputs")
    if len(tri)==0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(tri)):
        raise ValueError("empty or nonfinite hemisphere closest-point input")
    if not np.issubdtype(ids.dtype,np.integer) or ids.dtype.kind=="b":
        raise ValueError("stable IDs must be integers")
    if n.shape!=(3,) or not np.all(np.isfinite(n)):
        raise ValueError("hemisphere normal must be a finite 3-vector")
    nn=float(np.linalg.norm(n))
    if not math.isfinite(nn) or nn<=0:
        raise ValueError("hemisphere normal must be nonzero")
    n/=nn

    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    ab=b-a;ac=c-a;ap=p-a
    d00=np.einsum("ij,ij->i",ab,ab);d01=np.einsum("ij,ij->i",ab,ac);d11=np.einsum("ij,ij->i",ac,ac)
    d20=np.einsum("ij,ij->i",ap,ab);d21=np.einsum("ij,ij->i",ap,ac)
    den=d00*d11-d01*d01
    face_ok=(d00>0)&(d11>0)&(den>32*np.finfo(np.float64).eps*d00*d11)
    v=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=face_ok)
    w=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=face_ok)
    face=a+v[:,None]*ab+w[:,None]*ac
    face_ok &= (v>=0)&(w>=0)&(v+w<=1)

    # Slots: vertices 0:3, edge projections 3:6, face projection 6,
    # plane/edge intersections 7:10, and clipped-chord projection 10.
    cand=np.zeros((len(tri),11,3),dtype=np.float64)
    valid=np.zeros((len(tri),11),dtype=bool)
    cand[:,0:3]=tri;valid[:,0:3]=True
    edges=((a,b),(b,c),(c,a))
    for j,(x,y) in enumerate(edges):
        e=y-x;ee=np.einsum("ij,ij->i",e,e)
        u=np.divide(np.einsum("ij,ij->i",p-x,e),ee,out=np.zeros_like(ee),where=ee>0)
        u=np.clip(u,0.0,1.0)
        cand[:,3+j]=x+u[:,None]*e
        valid[:,3+j]=ee>0
    cand[:,6]=face;valid[:,6]=face_ok

    signed=np.einsum("ijk,k->ij",tri-p[None,None,:],n)
    inter=np.zeros((len(tri),3,3),dtype=np.float64)
    inter_valid=np.zeros((len(tri),3),dtype=bool)
    for j,(i0,i1) in enumerate(((0,1),(1,2),(2,0))):
        s0=signed[:,i0];s1=signed[:,i1]
        on0=s0==0.0;on1=s1==0.0
        crossing=((s0<0)&(s1>0))|((s0>0)&(s1<0))
        dene=s1-s0
        t=np.divide(-s0,dene,out=np.zeros_like(dene),where=crossing)
        point=tri[:,i0]+t[:,None]*(tri[:,i1]-tri[:,i0])
        point=np.where(on0[:,None],tri[:,i0],point)
        point=np.where((~on0&on1)[:,None],tri[:,i1],point)
        inter[:,j]=point;inter_valid[:,j]=on0|on1|crossing
        cand[:,7+j]=point;valid[:,7+j]=on0|on1|crossing

    # The farthest pair of plane intersections spans the clipped chord.
    pair_indices=((0,1),(0,2),(1,2))
    pair_d2=np.full((len(tri),3),-1.0,dtype=np.float64)
    for k,(i,j) in enumerate(pair_indices):
        delta=inter[:,i]-inter[:,j]
        d2=np.einsum("ij,ij->i",delta,delta)
        pair_d2[:,k]=np.where(inter_valid[:,i]&inter_valid[:,j],d2,-1.0)
    best_pair=np.argmax(pair_d2,axis=1)
    has_pair=np.max(pair_d2,axis=1)>=0.0
    uidx=np.zeros(len(tri),dtype=np.int64);vidx=np.zeros(len(tri),dtype=np.int64)
    for k,(i,j) in enumerate(pair_indices):
        take=best_pair==k;uidx[take]=i;vidx[take]=j
    rows=np.arange(len(tri));u=inter[rows,uidx];vtx=inter[rows,vidx];edge=vtx-u
    edge2=np.einsum("ij,ij->i",edge,edge)
    t=np.divide(np.einsum("ij,ij->i",p-u,edge),edge2,out=np.zeros_like(edge2),where=edge2>0)
    t=np.clip(t,0.0,1.0)
    cand[:,10]=u+t[:,None]*edge;valid[:,10]=has_pair

    side=np.einsum("ijk,k->ij",cand-p[None,None,:],n)
    scale=np.maximum(1.0,np.max(np.abs(tri-p[None,None,:]),axis=(1,2)))
    tol=64*np.finfo(np.float64).eps*scale
    valid &= side>=-tol[:,None]
    # Correct only arithmetic-scale residuals for plane candidates; this is a
    # proposal point and final exact ray classification remains mandatory.
    near_negative=valid&(side<0.0)
    if np.any(near_negative):
        cand[near_negative]-=side[near_negative,None]*n[None,:]
    delta=cand-p[None,None,:]
    dist2=np.einsum("ijk,ijk->ij",delta,delta)
    dist2=np.where(valid,dist2,np.inf)
    slot=np.argmin(dist2,axis=1);best2=dist2[rows,slot]
    if not np.isfinite(best2).any():
        raise ValueError("mesh has no finite point in the requested hemisphere")
    order=np.lexsort((rows,ids.astype(np.int64),best2));i=int(order[0])
    contact=cand[i,int(slot[i])].copy();d2=float(best2[i])
    return {"point_world_m":contact,"distance_m":float(math.sqrt(max(0.0,d2)),),
        "distance_squared_m2":d2,"triangle_row":i,"triangle_stable_id":int(ids[i]),
        "tie_count":int(np.count_nonzero(best2==d2)),"candidate_slot":int(slot[i]),
        "hemisphere_normal_world_unit":n.copy(),"hemisphere_dot_m":float(np.dot(contact-p,n))}

def nearest_inside_union_outward_hemisphere_direction(point_world, inside_banks, bank_records, source_outward):
    """Nearest boundary point on the initially containing union in a source hemisphere."""
    p=np.asarray(point_world,dtype=np.float64)
    n=np.array(source_outward,dtype=np.float64,copy=True)
    if p.shape!=(3,) or n.shape!=(3,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(n)):
        raise ValueError("invalid point/source normal for outward hemisphere")
    length=float(np.linalg.norm(n))
    if not math.isfinite(length) or length<=0:raise ValueError("source normal is zero")
    n/=length
    candidates=[]
    for key in sorted((int(k[0]),int(k[1])) for k in inside_banks):
        rec=bank_records[key]
        cp=closest_points_on_triangle_mesh_in_hemisphere(p,rec["world_triangles"],rec["id"],n)
        candidates.append((cp["distance_squared_m2"],key[0],key[1],cp["triangle_stable_id"],cp["triangle_row"],cp))
    if not candidates:raise ValueError("outward hemisphere search requires nonempty inside-bank union")
    candidates.sort(key=lambda q:(q[0],q[1],q[2],q[3],q[4]))
    _,body,bone,_,_,best=candidates[0]
    delta=np.asarray(best["point_world_m"],np.float64)-p
    distance=float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance<=0:raise ValueError("outward hemisphere closest point has zero-distance direction")
    direction=delta/distance;dot=float(np.dot(direction,n))
    if dot < -128*np.finfo(np.float64).eps:
        raise ValueError("constrained closest point escaped source hemisphere")
    return direction,{"body":body,"stable_bone_id":bone,"triangle_stable_id":best["triangle_stable_id"],
        "triangle_row":best["triangle_row"],"contact_world_m":np.asarray(best["point_world_m"],np.float64),
        "distance_m":distance,"triangle_tie_count":best["tie_count"],"active_bank_count":len(candidates),
        "selection_rule":"minimum world-metric squared distance over initially containing complete banks clipped by n_source·(x-start)>=0; ties by body, stable bone ID, stable triangle ID, then input row",
        "source_hemisphere_dot":dot,"candidate_slot":best["candidate_slot"],
        "plane_roundoff_residual_m":best["hemisphere_dot_m"]}

def nearest_inside_union_boundary_direction(point_world, inside_banks, bank_records):
    """Choose closest complete triangle among currently containing banks deterministically."""
    candidates=[]
    for key in sorted((int(k[0]),int(k[1])) for k in inside_banks):
        rec=bank_records[key]
        cp=closest_points_on_triangle_mesh(point_world,rec["world_triangles"],rec["id"])
        candidates.append((cp["distance_squared_m2"],key[0],key[1],cp["triangle_stable_id"],cp))
    if not candidates:raise ValueError("nearest direction requires nonempty initial inside-bank union")
    candidates.sort(key=lambda q:(q[0],q[1],q[2],q[3],q[4]["triangle_row"]))
    _,body,bone,_,best=candidates[0]
    delta=np.asarray(best["point_world_m"],np.float64)-np.asarray(point_world,np.float64)
    length=float(np.linalg.norm(delta))
    if not math.isfinite(length) or length<=0:raise ValueError("inside sample has zero-distance closest boundary; no direction")
    return delta/length,{"body":body,"stable_bone_id":bone,"triangle_stable_id":best["triangle_stable_id"],
        "triangle_row":best["triangle_row"],"contact_world_m":np.asarray(best["point_world_m"],np.float64),
        "distance_m":length,"triangle_tie_count":best["tie_count"],"active_bank_count":len(candidates),
        "selection_rule":"minimum world-metric squared distance; ties by body, stable bone ID, stable triangle ID, then input row"}

def validate_source_orientation_input_pins(report):
    """Check the orientability receipt against its frozen source-generation inputs."""
    pins=report.get("input_pins", {})
    paths={
        "detail_sha256":ORIENTATION_DETAIL,
        "accepted_rows_sha256":ORIENTATION_ROWS,
        "compact_sha256":ORIENTATION_COMPACT,
        "W_sha256":ORIENTATION_TRANSFER,
        "batch_report_sha256":ORIENTATION_BATCH,
        "batch_runner_sha256":ORIENTATION_RUNNER,
    }
    for key,path in paths.items():
        if not path.is_file():
            raise ValueError(f"source orientation input missing: {key}")
        if pins.get(key)!=sha(path):
            raise ValueError(f"source orientation input pin mismatch: {key}")
    return {key:sha(path) for key,path in paths.items()}

def validate_source_outward_orientation(report):
    """Require one orientable source component with resolved layer-order sign; never guess a flip."""
    detail_top=report.get("detail_topology", {})
    coarse_top=report.get("coarse_O_topology", {})
    for label, top in (("detail", detail_top), ("coarse O", coarse_top)):
        if (top.get("component_count") != 1 or
            top.get("adjacency_winding_conflict_count") != 0 or
            top.get("nonmanifold_edges") != 0):
            raise ValueError(f"{label} source orientation is not uniquely usable")
    outward=report.get("component_outward_sign_from_layer_order", {}).get("0")
    parity=report.get("component_fine_to_coarse_winding_anchors", {}).get("0")
    if not isinstance(outward, dict) or not isinstance(parity, dict):
        raise ValueError("missing component-zero source orientation anchors")
    fine_sign=outward.get("chosen_outward_sign")
    fine_coarse=parity.get("chosen_outward_parity")
    for label, record, sign in (("fine outward", outward, fine_sign), ("fine/coarse parity", parity, fine_coarse)):
        if sign not in (-1, 1) or int(record.get("anchor_count", 0)) <= 0:
            raise ValueError(f"{label} sign lacks a valid anchor")
        agreement=float(record.get("agreement_fraction", 0.0))
        if not math.isfinite(agreement) or agreement <= .5:
            raise ValueError(f"{label} sign is ambiguous")
    return {
        "fine_outward_sign": int(fine_sign),
        "fine_to_coarse_winding_parity": int(fine_coarse),
        "coarse_outward_sign": int(fine_sign)*int(fine_coarse),
        "fine_anchor_count": int(outward["anchor_count"]),
        "fine_anchor_agreement": float(outward["agreement_fraction"]),
        "parity_anchor_count": int(parity["anchor_count"]),
        "parity_anchor_agreement": float(parity["agreement_fraction"]),
    }

def registered_source_outward_face_normal(face_row, candidate, detail, base, orientation):
    """Resolve the outward normal from typed original-face ancestry, with no directional fallback."""
    face_row=int(face_row)
    kind=np.asarray(candidate["face_kind"])
    fine_parent=np.asarray(candidate["parent_fine_detail_face_row"])
    coarse_parent=np.asarray(candidate["parent_coarse_face_row"])
    if face_row < 0 or face_row >= len(kind):
        raise ValueError("face row is outside candidate face table")
    face_kind=int(kind[face_row])
    if face_kind == 2:
        parent=int(fine_parent[face_row])
        fine_faces=np.asarray(detail["source_surface_faces_detail_rows"],np.int64)
        fine_points=np.asarray(detail["detail_baseline_world_m"],np.float64)
        if parent < 0 or parent >= len(fine_faces):
            raise ValueError("fine source face ancestry missing or invalid")
        tri=fine_points[fine_faces[parent]]
        sign=int(orientation["fine_outward_sign"])
        provenance={"kind":"original_NHSKIN_face","row":parent}
    elif face_kind == 1:
        parent=int(coarse_parent[face_row])
        coarse_faces=np.asarray(base["source_O_faces"],np.int64)
        coarse_points=np.asarray(base["points_world_m"],np.float64)
        if parent < 0 or parent >= len(coarse_faces):
            raise ValueError("coarse source face ancestry missing or invalid")
        tri=coarse_points[coarse_faces[parent]]
        sign=int(orientation["coarse_outward_sign"])
        provenance={"kind":"original_coarse_O_face","row":parent}
    else:
        raise ValueError(f"unsupported face_kind for source normal: {face_kind}")
    if tri.shape != (3,3) or not np.all(np.isfinite(tri)):
        raise ValueError("source face vertices are malformed")
    raw=np.cross(tri[1]-tri[0],tri[2]-tri[0])
    length=float(np.linalg.norm(raw))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("source face is degenerate")
    normal=(sign/length)*raw
    return normal,provenance

def choose_proposal_direction(mode, nearest_direction, source_outward=None):
    """Select a certified source normal or preserve the historical nearest-boundary vector."""
    nearest=np.asarray(nearest_direction,dtype=np.float64)
    if nearest.shape != (3,) or not np.all(np.isfinite(nearest)) or np.linalg.norm(nearest) == 0:
        raise ValueError("nearest-boundary direction must be a finite nonzero 3-vector")
    nearest=nearest/np.linalg.norm(nearest)
    if mode == "nearest-boundary":
        return nearest.copy()
    if mode != "source-outward":
        raise ValueError("unknown target direction mode")
    if source_outward is None:
        raise ValueError("source-outward mode requires a certified source normal")
    outward=np.asarray(source_outward,dtype=np.float64)
    if outward.shape != (3,) or not np.all(np.isfinite(outward)):
        raise ValueError("source outward normal must be a finite 3-vector")
    length=float(np.linalg.norm(outward))
    if length == 0.0:
        raise ValueError("source outward normal is zero")
    return outward/length

def bank_orientation(tris):
 vids={};tri_vid=np.empty((len(tris),3),np.int64)
 for ti in range(len(tris)):
  for j in range(3):
   key=tuple(np.asarray(tris[ti,j],np.float32).view(np.uint32).tolist())
   tri_vid[ti,j]=vids.setdefault(key,len(vids))
 edge_count=defaultdict(int);edge_balance=defaultdict(int)
 for a,b,c in tri_vid:
  for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
   key=(min(u,v),max(u,v));edge_count[key]+=1;edge_balance[key]+=1 if u<v else -1
 vol6=float(np.einsum("ij,ij->i",tris[:,0],np.cross(tris[:,1],tris[:,2])).sum())
 closed=bool(edge_count) and all(v==2 for v in edge_count.values())
 oriented=closed and all(v==0 for v in edge_balance.values())
 sign=1 if vol6>0 else -1 if vol6<0 else 0
 return (sign if oriented else 0),{"vertex_count":len(vids),"edge_count":len(edge_count),"closed":closed,"consistently_oriented":oriented,"signed_six_volume_m3":vol6}

def fracpoint(x):
 return tuple(Fraction.from_float(float(v)) for v in x)

def exact_point_from_bary(lam,tri):
 return tuple(sum((lam[i]*tri[i][k] for i in range(3)),Fraction(0)) for k in range(3))

def virtual_sample_world_f64(emitted_points_f32, face_node_rows, barycentric_weights):
    """Evaluate a virtual point in F64 from its emitted-F32 parent vertices."""
    points=np.asarray(emitted_points_f32)
    rows=np.asarray(face_node_rows)
    weights=np.asarray(barycentric_weights,np.float64)
    if points.dtype!=np.dtype(np.float32) or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError("emitted points must be an N x 3 Float32 array")
    if rows.shape!=(3,) or not np.issubdtype(rows.dtype,np.integer) or np.any(rows<0) or np.any(rows>=len(points)):
        raise ValueError("face node rows must be three valid integer indices")
    if weights.shape!=(3,) or not np.all(np.isfinite(weights)) or np.min(weights)<=0 or abs(float(weights.sum())-1.0)>1e-12:
        raise ValueError("barycentric weights must be finite, positive, and sum to one")
    return weights @ points[rows].astype(np.float64)

def strict_interior_sample(face_local,bone_local,bankloc,bank_sign,exact,bary_fn):
 a=tuple(tuple(v for v in row) for row in (fracpoint(p) for p in face_local))
 b=tuple(tuple(v for v in row) for row in (fracpoint(p) for p in bone_local))
 pts=sorted(set(exact.triangle_intersection_points(a,b)))
 if len(pts)<2:return None,"NO_INTERSECTION_SEGMENT",None
 p0,p1=max(((x,y) for i,x in enumerate(pts) for y in pts[i+1:]),key=lambda xy:sum(float(xy[0][k]-xy[1][k])**2 for k in range(3)))
 l0=bary_fn(a,p0);l1=bary_fn(a,p1);base=tuple((l0[i]+l1[i])/2 for i in range(3))
 sn=np.cross(face_local[1]-face_local[0],face_local[2]-face_local[0]);sl=float(np.linalg.norm(sn))
 bn=np.cross(bone_local[1]-bone_local[0],bone_local[2]-bone_local[0]);bl=float(np.linalg.norm(bn))
 if sl<=0 or bl<=0:return None,"DEGENERATE_PAIR_TRIANGLE",None
 bn=bank_sign*bn/bl;sn/=sl
 inward=-bn+float(np.dot(bn,sn))*sn;il=float(np.linalg.norm(inward))
 if il<1e-14:return None,"NEAR_TANGENT_PLANES",None
 inward/=il
 e1=face_local[1]-face_local[0];e2=face_local[2]-face_local[0]
 uv=np.linalg.lstsq(np.column_stack((e1,e2)),inward,rcond=None)[0]
 dl=np.asarray([-uv[0]-uv[1],uv[0],uv[1]],np.float64)
 bf=np.asarray([float(x) for x in base])
 tmax=min((-bf[i]/dl[i] for i in range(3) if dl[i]<0),default=math.inf)
 if not math.isfinite(tmax) or tmax<=0:return None,"NO_INWARD_SIMPLEX_ROOM",None
 for frac in STRICT_SAMPLE_INWARD_FRACTIONS:
  lf=bf+(tmax*frac)*dl
  if np.min(lf)<=0:continue
  l0=Fraction.from_float(float(lf[0]));l1=Fraction.from_float(float(lf[1]));lam=(l0,l1,Fraction(1)-l0-l1)
  if min(lam)<=0:continue
  loc=exact.point_location_prepared_indexed(exact_point_from_bary(lam,a),bankloc)
  if loc.get("location")=="inside":return lam,"INSIDE",{"point0":p0,"point1":p1,"locator":loc}
 return None,"NO_CERTIFIED_INSIDE_FACE_SAMPLE",None

def locate_query_receipt(candidate, explicit=None):
 if explicit:
  p=Path(explicit)
  if not p.is_file():raise FileNotFoundError(p)
  return p
 for parent in (Path(candidate).parent,*Path(candidate).parents):
  p=parent/"audit-130/query/query-receipt.json"
  if p.is_file():return p
 raise FileNotFoundError("could not locate matched audit-130/query/query-receipt.json; pass --query-receipt")

def main(argv=None):
 global CAND,SURFACE,OUT,CLEARANCE_M
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--candidate",required=True,type=Path)
 ap.add_argument("--surface-report",required=True,type=Path)
 ap.add_argument("--output-dir",required=True,type=Path)
 ap.add_argument("--query-receipt",type=Path,default=None)
 ap.add_argument("--clearance-m",type=float,default=CLEARANCE_M)
 ap.add_argument("--direction-mode",choices=("nearest-boundary","source-outward","outward-nearest"),default="nearest-boundary")
 args=ap.parse_args(argv)
 CAND=args.candidate.resolve();SURFACE=args.surface_report.resolve();OUT=args.output_dir.resolve()
 CLEARANCE_M=float(args.clearance_m)
 if not math.isfinite(CLEARANCE_M) or CLEARANCE_M<=0:raise ValueError("clearance must be positive finite meters")
 if OUT.exists():raise FileExistsError(f"output exists: {OUT}")
 qreceipt=locate_query_receipt(CAND,args.query_receipt)
 check_first_union_fixture()
 tstart=time.monotonic()
 if not CAND.is_file() or not SURFACE.is_file():raise FileNotFoundError("candidate or matched surface report missing")
 z=np.load(CAND,allow_pickle=False);P=np.asarray(z["points_world_m"],np.float32);F=np.asarray(z["source_O_faces"],np.int64)
 if P.ndim!=2 or P.shape[1]!=3 or F.ndim!=2 or F.shape[1]!=3 or not np.all(np.isfinite(P)):raise RuntimeError("candidate point/face arrays invalid")
 surface=json.loads(SURFACE.read_text())
 if not surface.get("complete") or surface.get("candidate_sha256")!=sha(CAND) or surface.get("surface_point_count")!=len(P) or surface.get("surface_face_count")!=len(F):raise RuntimeError("matched 159 candidate/report pin mismatch")
 if np.any(F<0) or np.any(F>=len(P)):raise RuntimeError("candidate face index outside point array")
 detail=np.load(DETAIL,allow_pickle=False)
 source_orientation=None;source_base=None;orientation_report=None
 if args.direction_mode in ("source-outward", "outward-nearest"):
  orientation_report=json.loads(ORIENTATION.read_text())
  source_orientation_pins=validate_source_orientation_input_pins(orientation_report)
  source_orientation=validate_source_outward_orientation(orientation_report)
  source_base=np.load(BASE,allow_pickle=False)
 reader=loadmod(BANKHELPER,"bone_bank_reader_175")
 triangles,ids,gids,bodies,M,off,pos,bankkeys,bankpins,exact_unused=reader.read_bank_inputs(BANK,BANK_FIX)
 if len(bankkeys)!=185 or len(triangles)!=len(ids):raise RuntimeError("incomplete pinned 185-bank input")
 sys.path.insert(0,str(EVIDENCE/"incremental-exact-audit-owner-253/src"))
 exact=loadmod(EXACT,"numilab_human.cardiac_cavity_intersections")
 from sys import path
 path.insert(0,str(D/"barycentric-virtual-point-170"))
 cmod=loadmod(CONSTRAINT,"virtual_point_constraint_172")
 bary_fn=cmod.exact_triangle_barycentric
 banks={}
 for b0,k0 in bankkeys:
  b,k=int(b0),int(k0);mask=(ids[:,0]==b)&(ids[:,1]==k)
  T=np.asarray(triangles[mask],np.float64);TI=np.asarray(ids[mask,3],np.int64)
  if not len(T):raise RuntimeError(f"empty bank {b}/{k}")
  sign,top=bank_orientation(T)
  WT=world_triangles_from_local(T,M[b],pos[b],off[b])
  banks[(b,k)]={"T":T,"world_triangles":WT,"id":TI,"lo":T.min((0,1)),"hi":T.max((0,1)),"tl":T.min(1),"th":T.max(1),"M":M[b],"pos":pos[b],"off":off[b],"sign":sign,"topology":top,"locator":None,"tri_by_id":{int(i):T[j] for j,i in enumerate(TI)}}
  if not top["closed"] or not top["consistently_oriented"] or sign==0:raise RuntimeError(f"bank shell topology/orientation unverified {b}/{k}: {top}")
 # Keep the full 185-key ordered inventory; locators are prepared lazily.
 qreport=json.loads(qreceipt.read_text())
 if qreport.get("query_rows")!=len(P) or qreport.get("total_banks") not in (None,185):raise RuntimeError("matched 130 query receipt incomplete")
 # Build crossing-pair records from the complete matched 159 report.
 pairs=[]
 for bank in surface["banks"]:
  result=bank.get("result") or {}
  if not result.get("triangle_pairs"):continue
  body,bone=int(bank["body"]),int(bank["stable_bone_id"])
  for f,tid in result["triangle_pairs"]:pairs.append((int(f),body,bone,int(tid)))
 if len(set(pairs))!=len(pairs):raise RuntimeError("duplicate face/bank/triangle rows in matched surface report")
 # Cache complete bank point locators only when sample or target enters a bank AABB.
 def ensure_locator(key):
  rec=banks[key]
  if rec["locator"] is None:
   T=rec["T"];vertices=tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in T.reshape(-1,3))
   faces=tuple((3*i,3*i+1,3*i+2) for i in range(len(T)))
   rec["locator"]=exact.prepare_point_location(exact._records(vertices,faces))
  return rec["locator"]
 def classify_world(point):
  status={};unresolved=[]
  for key in bankkeys:
   key=(int(key[0]),int(key[1]));rec=banks[key]
   q=(np.asarray(point,np.float64)-rec["pos"])@rec["M"].T-rec["off"]
   if np.any(q<rec["lo"]) or np.any(q>rec["hi"]):status[key]="outside_aabb";continue
   rr=exact.point_location_prepared_indexed(fracpoint(q),ensure_locator(key))
   loc=rr.get("location")
   if loc not in ("inside","outside"):unresolved.append({"bank":list(key),"result":rr})
   status[key]=loc
  return status,unresolved
 # Prepare candidate face geometry and body-local transforms.
 body_local={}
 for b in {int(k[0]) for k in bankkeys}:
  rr=banks[next(k for k in banks if k[0]==b)]
  body_local[b]=(P.astype(np.float64)-rr["pos"])@rr["M"].T-rr["off"]
 output=[];fail=Counter();initial_classes=Counter();union_targets=0
 for pair_i,(face,body,bone,tid) in enumerate(pairs):
  key=(body,bone);rec=banks[key]
  tri=rec["tri_by_id"].get(tid)
  if tri is None:
   fail["MISSING_BANK_TRIANGLE_STABLE_ID"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"MISSING_BANK_TRIANGLE_STABLE_ID"});continue
  flocal=body_local[body][F[face]]
  lam,sample_status,sample_detail=strict_interior_sample(flocal,tri,ensure_locator(key),rec["sign"],exact,bary_fn)
  if sample_status!="INSIDE":
   fail[sample_status]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":sample_status,"face_kind":int(z["face_kind"][face])});continue
  lamf=np.asarray([float(v) for v in lam],np.float64)
  # Keep the virtual barycentric query in F64 over emitted F32 parent vertices.
  # F32-rounded coordinates are retained only as a diagnostic; tiny valid interior
  # samples can round across a bank boundary and must not be reclassified as outside.
  start=virtual_sample_world_f64(P,F[face],lamf)
  start_f32=start.astype(np.float32).astype(np.float64)
  classes,unresolved=classify_world(start)
  if unresolved:
   fail["INITIAL_UNION_CLASSIFICATION_UNRESOLVED"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"INITIAL_UNION_CLASSIFICATION_UNRESOLVED","witnesses":unresolved[:8]});continue
  if classes.get(key)!="inside":
   fail["RECONSTRUCTED_SAMPLE_NOT_INSIDE_IMPLICATED_BANK"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"RECONSTRUCTED_SAMPLE_NOT_INSIDE_IMPLICATED_BANK","implicated_class":classes.get(key),"inside_sample_world_m":start.tolist(),"inside_sample_world_f32_m":start_f32.tolist()});continue
  initial={k for k,v in classes.items() if v=="inside"};initial_classes[len(initial)]+=1
  try:
   nearest_direction,nearest=nearest_inside_union_boundary_direction(start,initial,banks)
   source_normal=None;source_provenance=None;hemisphere_nearest=None
   if args.direction_mode in ("source-outward", "outward-nearest"):
    source_normal,source_provenance=registered_source_outward_face_normal(face,z,detail,source_base,source_orientation)
   if args.direction_mode == "outward-nearest":
    direction,hemisphere_nearest=nearest_inside_union_outward_hemisphere_direction(start,initial,banks,source_normal)
   else:
    direction=choose_proposal_direction(args.direction_mode,nearest_direction,source_normal)
  except (ValueError,KeyError) as exc:
   fail["NEAREST_COMPLETE_BANK_TRIANGLE_FAILED"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"NEAREST_COMPLETE_BANK_TRIANGLE_FAILED","detail":str(exc)});continue
  events=[];unresolved_ray=[];end_states={}
  for key2 in bankkeys:
   key2=(int(key2[0]),int(key2[1]));rr=banks[key2];st=classes[key2]
   q=(start-rr["pos"])@rr["M"].T-rr["off"];d=rr["M"]@direction
   iv=ray_box(q,d,rr["lo"],rr["hi"])
   if iv is None:
    if st=="inside":unresolved_ray.append({"bank":list(key2),"reason":"INSIDE_START_OUTSIDE_AABB"})
    end_states[key2]=st=="inside";continue
   t0,t1=iv;te=t1+CLEARANCE_M
   p0=q+t0*d;p1=q+te*d;seglo=np.minimum(p0,p1);seghi=np.maximum(p0,p1)
   candidate=np.flatnonzero(np.all(rr["th"]>=seglo,axis=1)&np.all(rr["tl"]<=seghi,axis=1))
   hits=[];hit_unresolved=[]
   for ti in candidate:
    h=exact_unused["classify_segment_triangle"](p0,p1,rr["T"][int(ti)]);cl=h["classification"]
    if cl=="proper_interior_hit":
     tv=t0+float(h["t"])*(te-t0)
     ev={"distance_m":tv,"bank":key2,"triangle_stable_id":int(rr["id"][int(ti)])};events.append(ev);hits.append(ev)
    elif cl not in ("disjoint","parallel_disjoint"):
     hit_unresolved.append({"body":key2[0],"bone":key2[1],"triangle_stable_id":int(rr["id"][int(ti)]),"classification":cl,"exact_result":h})
   if hit_unresolved:unresolved_ray.extend(hit_unresolved[:8])
   end_states[key2]=(st=="inside")^(len(hits)%2==1)
  if unresolved_ray:
   fail["UNRESOLVED_RAY_CONTACT"]+=1;output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"UNRESOLVED_RAY_CONTACT","witnesses":unresolved_ray[:16]});continue
  remain=[list(k) for k,v in end_states.items() if v]
  if remain:
   fail["RAY_PARITY_FAILURE"]+=1;output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"RAY_PARITY_FAILURE","remaining_inside_banks":remain[:32]});continue
  exitrec=first_empty_union_interval(initial,events,CLEARANCE_M)
  if exitrec is None:
   fail["NO_FIRST_EMPTY_UNION_INTERVAL"]+=1;output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"NO_FIRST_EMPTY_UNION_INTERVAL","initial_inside_banks":[list(k) for k in sorted(initial)],"event_count":len(events)});continue
  target=np.asarray(start+exitrec["target_distance_m"]*direction,np.float32)
  emitted=float(np.dot(target.astype(np.float64)-start,direction))
  if not emitted>exitrec["exit_distance_m"] or (exitrec["next_event_distance_m"] is not None and not emitted<exitrec["next_event_distance_m"]):
   fail["F32_TARGET_LEFT_FIRST_FREE_INTERVAL"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"F32_TARGET_LEFT_FIRST_FREE_INTERVAL","exit":exitrec,"emitted_distance_m":emitted});continue
  # Exact full-union endpoint membership; do not rely only on event parity.
  endclass,unresolved_end=classify_world(target.astype(np.float64))
  if unresolved_end or any(v=="inside" for v in endclass.values()):
   fail["TARGET_NOT_EXACTLY_OUTSIDE_UNION"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"TARGET_NOT_EXACTLY_OUTSIDE_UNION","target_inside_banks":[list(k) for k,v in endclass.items() if v=="inside"],"unresolved":unresolved_end[:8],"target":target.tolist()});continue
  union_targets+=1
  record={"pair_index":pair_i,"face_row":face,"face_kind":int(z["face_kind"][face]),"skin_node_rows":F[face].tolist(),"barycentric_weights":lamf.tolist(),"inside_sample_world_m":start.tolist(),"inside_sample_world_f32_m":start_f32.tolist(),"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"nearest_boundary_direction_world_unit":nearest_direction.tolist(),"proposal_direction_world_unit":direction.tolist(),"direction_source":("registered NHSKIN/coarse source-face outward normal; absolute sign anchored by O-to-I layer order; exact union-ray certification unchanged" if args.direction_mode == "source-outward" else "vector from F64 barycentric virtual sample evaluated over emitted F32 face vertices to deterministic nearest complete triangle on an initially containing bank, evaluated in world metric"),"nearest_boundary_witness":{"body":nearest["body"],"stable_bone_id":nearest["stable_bone_id"],"triangle_stable_id":nearest["triangle_stable_id"],"triangle_row":nearest["triangle_row"],"contact_world_m":nearest["contact_world_m"].tolist(),"distance_m":nearest["distance_m"],"triangle_tie_count":nearest["triangle_tie_count"],"active_bank_count":nearest["active_bank_count"],"selection_rule":nearest["selection_rule"]},"initial_inside_union_banks":[list(k) for k in sorted(initial)],"initial_inside_union_count":len(initial),"union_exit_distance_m":exitrec["exit_distance_m"],"next_union_event_distance_m":exitrec["next_event_distance_m"],"free_interval_margin_m":emitted-exitrec["exit_distance_m"],"target_distance_m":emitted,"target_world_f32_m":target.tolist(),"target_exactly_outside_all185":True,"ray_events":len(events),"touched_banks":len({e["bank"] for e in events}),"status":("PASS_SOURCE_OUTWARD_FIRST_UNION_TARGET" if args.direction_mode == "source-outward" else "PASS_NEAREST_BOUNDARY_FIRST_UNION_TARGET")}
  if args.direction_mode in ("source-outward", "outward-nearest"):
   record["source_surface_normal_world_unit"]=source_normal.tolist()
   record["source_normal_dot_nearest_direction"]=float(np.dot(source_normal,nearest_direction))
   record["source_normal_dot_proposal_direction"]=float(np.dot(source_normal,direction))
   record["source_normal_provenance"]=source_provenance
  if hemisphere_nearest is not None:
   record["outward_hemisphere_nearest_witness"]={**hemisphere_nearest,"contact_world_m":hemisphere_nearest["contact_world_m"].tolist()}
  if args.direction_mode == "outward-nearest":
   record["direction_source"]="closest complete-bank point in registered source outward hemisphere; first-union event and all185 endpoint audit unchanged"
   record["status"]="PASS_OUTWARD_HEMISPHERE_FIRST_UNION_TARGET"
  output.append(record)
 # Write heavy numeric records before metadata; keep all unmatched/failures in JSONL.
 outrows=OUT/"target-records.jsonl";npz=OUT/"targets.npz";reportp=OUT/"report.json"
 OUT.mkdir(parents=True,exist_ok=False)
 with outrows.open("w") as f:
  for r in output:f.write(json.dumps(r,separators=(",",":"),sort_keys=True,default=json_default)+"\n")
 passed=[r for r in output if r["status"] in ("PASS_NEAREST_BOUNDARY_FIRST_UNION_TARGET","PASS_SOURCE_OUTWARD_FIRST_UNION_TARGET","PASS_OUTWARD_HEMISPHERE_FIRST_UNION_TARGET")]
 np.savez_compressed(npz,
  pair_index=np.asarray([r["pair_index"] for r in passed],np.int64),
  face_rows=np.asarray([r["face_row"] for r in passed],np.int64),
  face_node_rows=np.asarray([r["skin_node_rows"] for r in passed],np.int64).reshape(-1,3),
  barycentric_weights=np.asarray([r["barycentric_weights"] for r in passed],np.float64).reshape(-1,3),
  inside_sample_world_m=np.asarray([r["inside_sample_world_m"] for r in passed],np.float64).reshape(-1,3),
  inside_sample_world_f32_m=np.asarray([r["inside_sample_world_f32_m"] for r in passed],np.float32).reshape(-1,3),
  bank_body=np.asarray([r["bank_body"] for r in passed],np.int32),
  bank_stable_bone_id=np.asarray([r["bank_stable_bone_id"] for r in passed],np.int32),
  bank_triangle_stable_id=np.asarray([r["bank_triangle_stable_id"] for r in passed],np.int32),
  nearest_boundary_direction_world_unit=np.asarray([r["nearest_boundary_direction_world_unit"] for r in passed],np.float64).reshape(-1,3),
  proposal_direction_world_unit=np.asarray([r["proposal_direction_world_unit"] for r in passed],np.float64).reshape(-1,3),
  nearest_boundary_contact_world_m=np.asarray([r["nearest_boundary_witness"]["contact_world_m"] for r in passed],np.float64).reshape(-1,3),
  nearest_boundary_distance_m=np.asarray([r["nearest_boundary_witness"]["distance_m"] for r in passed],np.float64),
  nearest_boundary_triangle_stable_id=np.asarray([r["nearest_boundary_witness"]["triangle_stable_id"] for r in passed],np.int64),
  nearest_boundary_body=np.asarray([r["nearest_boundary_witness"]["body"] for r in passed],np.int32),
  nearest_boundary_stable_bone_id=np.asarray([r["nearest_boundary_witness"]["stable_bone_id"] for r in passed],np.int32),
  nearest_boundary_tie_count=np.asarray([r["nearest_boundary_witness"]["triangle_tie_count"] for r in passed],np.int32),
  nearest_boundary_active_bank_count=np.asarray([r["nearest_boundary_witness"]["active_bank_count"] for r in passed],np.int32),
  union_exit_distance_m=np.asarray([r["union_exit_distance_m"] for r in passed],np.float64),
  target_world_f32_m=np.asarray([r["target_world_f32_m"] for r in passed],np.float32).reshape(-1,3),
  target_distance_m=np.asarray([r["target_distance_m"] for r in passed],np.float64),
  target_exactly_outside_all185=np.ones(len(passed),np.bool_),
  source_surface_normal_world_unit=np.asarray([r.get("source_surface_normal_world_unit",[0.0,0.0,0.0]) for r in passed],np.float64).reshape(-1,3),
  source_normal_valid=np.asarray(["source_surface_normal_world_unit" in r for r in passed],np.bool_),
  outward_hemisphere_contact_world_m=np.asarray([r.get("outward_hemisphere_nearest_witness",{}).get("contact_world_m",[0.0,0.0,0.0]) for r in passed],np.float64).reshape(-1,3),
  outward_hemisphere_distance_m=np.asarray([r.get("outward_hemisphere_nearest_witness",{}).get("distance_m",0.0) for r in passed],np.float64),
  outward_hemisphere_triangle_stable_id=np.asarray([r.get("outward_hemisphere_nearest_witness",{}).get("triangle_stable_id",-1) for r in passed],np.int64),
  outward_hemisphere_body=np.asarray([r.get("outward_hemisphere_nearest_witness",{}).get("body",-1) for r in passed],np.int32),
  outward_hemisphere_stable_bone_id=np.asarray([r.get("outward_hemisphere_nearest_witness",{}).get("stable_bone_id",-1) for r in passed],np.int32),
  outward_hemisphere_witness_valid=np.asarray(["outward_hemisphere_nearest_witness" in r for r in passed],np.bool_))
 summary={"schema":"numi.human.nearest-complete-bank-first-union-targets.272.v1","runner_sha256":sha(__file__),"classification":"construction_target_proposals_only_no_fit_or_clearance_admission","direction_mode":args.direction_mode,
  "inputs":{"candidate_sha256":sha(CAND),"surface_report_sha256":sha(SURFACE),"query_receipt_sha256":sha(qreceipt),"detail_source_sha256":sha(DETAIL),"exact_bank_triangle_export_sha256":sha(BANK/"exact-emitted.triangles.csv"),"exact_bank_body_export_sha256":sha(BANK/"exact-emitted.bodies.csv"),"exact_bank_node_export_sha256":sha(BANK/"exact-emitted.nodes.csv"),"exact_bank_inventory_sha256":sha(BANK/"all-bank-containment.csv.banks.csv"),"bank_reader_sha256":sha(BANKHELPER),"exact_predicate_sha256":sha(EXACT),"exact_segment_classifier_sha256":sha(SEGMENT_HELPER),"virtual_sample_helper_sha256":sha(CONSTRAINT),**({"source_orientation_report_sha256":sha(ORIENTATION),"coarse_source_candidate_sha256":sha(BASE)} if args.direction_mode in ("source-outward", "outward-nearest") else {})},
  "candidate_points":len(P),"candidate_faces":len(F),"matched_surface_bank_triangle_pairs":len(pairs),
  "banks":len(bankkeys),"matched_159_complete":bool(surface.get("complete")),"crossing_pairs":len(pairs),"inside_samples_certified_for_implicated_bank":sum(1 for r in output if r.get("inside_sample_world_m") is not None),
  "inside_sample_numeric_policy":"F64 barycentric evaluation over emitted F32 parent vertices is authoritative for interior classification; F32-rounded sample is diagnostic only. Final target remains emitted F32 and is exactly checked outside all185 banks.",
  "strict_interior_sample_inward_fractions":list(STRICT_SAMPLE_INWARD_FRACTIONS),
  "nearest_boundary_first_union_targets":len(passed),"passing_targets":len(passed),"failure_counts":dict(fail),"initial_inside_union_count_histogram":dict(initial_classes),
  "minimum_source_hemisphere_dot":(min((r["source_normal_dot_proposal_direction"] for r in passed if "source_normal_dot_proposal_direction" in r),default=None)),
  "source_orientation":source_orientation if args.direction_mode in ("source-outward", "outward-nearest") else None,
  "source_orientation_input_pins":source_orientation_pins if args.direction_mode in ("source-outward", "outward-nearest") else None,
  "direction_method":({"nearest-boundary":"Legacy nearest point on initially containing complete bank; exact first-union ray plus all185 endpoint classification unchanged.","source-outward":"Pinned registered source-face outward normal; exact first-union ray plus all185 endpoint classification unchanged.","outward-nearest":"Nearest point on initially containing complete bank triangles clipped to the source-normal hemisphere (plane residual projected only within 64 binary64 eps times local span; returned direction dot must be >= -128 eps); deterministic stable-ID ties; exact first-union ray plus all185 endpoint classification unchanged."})[args.direction_mode],
  "full_union_endpoint_classification":"Exact point-location for every target against all185 complete banks; target accepted only if outside every complete bank; event-parity endpoint also required.",
  "limitations":["The selected direction is a construction proposal, not clearance admission.","Only banks in the exact initial inside-union set are searched for the nearest complete triangle diagnostic; the subsequent ray and target checks cover all 185 banks.","No fit or native physical step was run."],
  "elapsed_seconds":time.monotonic()-tstart,"result_rows":len(output),"passing_targets":len(passed),"output_sha256_pending":True}
 reportp.write_text(json.dumps(summary,indent=2,default=json_default)+"\n")
 print(json.dumps({"report":str(reportp),"report_sha256":sha(reportp),"npz":str(npz),"npz_sha256":sha(npz),"rows":len(output),"pairs":len(pairs),"targets":len(passed),"failures":dict(fail),"elapsed_seconds":summary["elapsed_seconds"]},indent=2))
if __name__=="__main__":main()

