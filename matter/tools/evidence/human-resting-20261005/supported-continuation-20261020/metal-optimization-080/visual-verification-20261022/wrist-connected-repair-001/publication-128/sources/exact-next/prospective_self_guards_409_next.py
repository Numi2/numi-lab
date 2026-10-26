"""Conservative near-pair candidates and explicit 026 guard-row classification.

Near pairs are broadphase construction proposals, not exact collision results.
"""
from __future__ import annotations
import hashlib
import numpy as np

def _pair_array(values, face_count, *, allow_empty_list=True):
    a=np.asarray(values)
    if a.size==0:
        if a.ndim==1 and a.shape==(0,) and allow_empty_list:
            return np.empty((0,2),dtype=np.int64)
        if a.ndim==2 and a.shape==(0,2) and np.issubdtype(a.dtype,np.integer):
            return np.empty((0,2),dtype=np.int64)
        raise ValueError("pair IDs must be integer Nx2")
    if a.ndim!=2 or a.shape[1]!=2 or not np.issubdtype(a.dtype,np.integer):
        raise ValueError("pair IDs must be integer Nx2")
    out=[]
    for a0,b0 in a:
        i,j=int(a0),int(b0)
        if i<0 or j<0 or i>=face_count or j>=face_count or i==j:
            raise ValueError("invalid pair face IDs")
        out.append((min(i,j),max(i,j)))
    if len(set(out))!=len(out): raise ValueError("duplicate pair IDs")
    return np.asarray(sorted(out),dtype=np.int64).reshape((-1,2))

def _sha_pairs(pairs):
    return hashlib.sha256(np.ascontiguousarray(pairs,dtype=np.int64).view(np.uint8)).hexdigest()

def _outward_f32(value, direction):
    value=np.asarray(value,dtype=np.float64)
    if not np.isfinite(value).all(): raise ValueError("expanded AABB bound is nonfinite")
    packed=np.asarray(value,dtype=np.float32)
    if not np.isfinite(packed).all(): raise ValueError("expanded AABB bound exceeds Float32")
    toward=np.float32(-np.inf if direction<0 else np.inf)
    return np.nextafter(packed,toward)

def enumerate_prospective_pairs(state, radius_m, seed_pair_ids=()):
    """Union input guard IDs with conservative expanded-query-box AABB pairs.

    Each query triangle's AABB is padded by radius_m once against the original
    cached full-surface index. This guarantees candidates whose box separation
    is at most radius_m on every axis (including every Euclidean-distance
    neighbor within radius_m); it may conservatively include farther pairs.
    Shared topological edges are omitted from prospective candidates; shared
    single vertices are retained. Existing seed IDs are always preserved.
    """
    radius=float(radius_m)
    if not np.isfinite(radius) or radius<0: raise ValueError("radius_m must be finite and nonnegative")
    owner=getattr(state,"owner",None); points=np.asarray(getattr(state,"points",None)); faces=np.asarray(getattr(state,"faces",None))
    records=getattr(state,"records",None); index=getattr(state,"index",None)
    if owner is None or records is None or index is None: raise ValueError("state must be a prepared exact-AABB state")
    if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError("state points must be finite Float32 Nx3")
    if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer) or len(records)!=len(faces):
        raise ValueError("state faces/records mismatch")
    seeds=_pair_array(seed_pair_ids,len(faces))
    seed_set={tuple(map(int,p)) for p in seeds}
    near=set()
    shared_edge_skip=0
    fids=faces.astype(np.int64,copy=False)
    for i,rec in enumerate(records):
        tri,_,_,row,eff_ids=rec
        if int(row)!=i: raise ValueError("cached AABB records must retain face-row IDs")
        xyz=points[fids[i]].astype(np.float64)
        lo=_outward_f32(xyz.min(axis=0)-radius,-1)
        hi=_outward_f32(xyz.max(axis=0)+radius,1)
        lo_key=owner.float32_point_lattice_key(lo)
        hi_key=owner.float32_point_lattice_key(hi)
        query=(tri,lo_key,hi_key,i,tuple(eff_ids))
        for _,other in owner._query_surface_aabb((query,),index):
            j=int(other[3])
            if j<=i: continue
            common=set(eff_ids).intersection(other[4])
            if len(common)>=2:
                shared_edge_skip+=1
                continue
            near.add((i,j))
    all_pairs=np.asarray(sorted(seed_set|near),dtype=np.int64).reshape((-1,2))
    near_pairs=np.asarray(sorted(near),dtype=np.int64).reshape((-1,2))
    seed_only=np.asarray(sorted(seed_set-near),dtype=np.int64).reshape((-1,2))
    report={
      "status":"CONSERVATIVE_EXPANDED_AABB_NEAR_PAIR_PROPOSAL",
      "exact_collision_test_performed":False,
      "geometry_admitted":False,
      "physical_steps":0,
      "base_candidate_sha256":str(getattr(state,"candidate_sha256","")),
      "predicate_sha256":str(getattr(state,"predicate_sha256","")),
      "radius_m":radius,
      "padding_semantics":"one query-side AABB expansion against cached unexpanded full surface",
      "coverage_bound":"includes all pairs with componentwise AABB gap <= radius_m; conservative superset, not exact triangle distance",
      "seed_pair_count":int(len(seeds)),
      "prospective_aabb_pair_count":int(len(near_pairs)),
      "prospective_shared_edge_pair_skips":int(shared_edge_skip),
      "seed_only_pair_count":int(len(seed_only)),
      "union_pair_count":int(len(all_pairs)),
      "seed_pair_ids_sha256":_sha_pairs(seeds),
      "prospective_pair_ids_sha256":_sha_pairs(near_pairs),
      "union_pair_ids_sha256":_sha_pairs(all_pairs),
      "prospective_pair_ids":near_pairs.tolist(),
      "seed_only_pair_ids":seed_only.tolist(),
      "union_pair_ids":all_pairs.tolist(),
    }
    return {"pair_ids":all_pairs,"prospective_pair_ids":near_pairs,"seed_only_pair_ids":seed_only,"report":report}

def build_guard_rows(points_mm, faces, pair_ids, pair_owner, *, margin_fraction=0.25, baseline_tolerance_mm=2e-8):
    """Build 026 rows one pair at a time, retaining every incompatible ID."""
    f=np.asarray(faces); p=np.asarray(points_mm,dtype=np.float64)
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer): raise ValueError("faces must be integer Fx3")
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError("points_mm must be finite Nx3")
    pairs=_pair_array(pair_ids,len(f))
    rows=[]; records=[]; compatible=[]; incompatible=[]
    for i,j in pairs.tolist():
        try:
            pair_rows,pair_records=pair_owner.build_pair_rows(p,f,np.asarray([[i,j]],dtype=np.int64),margin_fraction=margin_fraction)
            if not pair_rows or len(pair_records)!=1 or list(map(int,pair_records[0].get("faces",())))!=[i,j]:
                raise ValueError("026 owner did not return one typed pair record and rows")
            local=[]
            for row in pair_rows:
                nodes=np.asarray(row.get("nodes")); coeff=np.asarray(row.get("coefficients"),dtype=np.float64); lower=float(row.get("lower"))
                if nodes.ndim!=1 or not np.issubdtype(nodes.dtype,np.integer) or coeff.shape!=(len(nodes),3):
                    raise ValueError("malformed 026 row shape")
                if len(nodes)==0 or nodes.min()<0 or nodes.max()>=len(p) or not np.isfinite(coeff).all() or not np.isfinite(lower):
                    raise ValueError("nonfinite/out-of-domain 026 row")
                if lower>baseline_tolerance_mm:
                    raise ValueError(f"026 row not zero-feasible at baseline: {lower:.17g} mm")
                local.append(row)
            if not local: raise ValueError("026 owner returned no constraints")
            rows.extend(local); records.extend(pair_records); compatible.append([i,j])
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            incompatible.append({"pair":[i,j],"reason":f"{type(exc).__name__}: {exc}"})
    return rows,records,{
      "status":"026_NEAR_PAIR_GUARD_ROWS_CLASSIFIED",
      "geometry_admitted":False,
      "physical_steps":0,
      "input_pair_count":int(len(pairs)),
      "compatible_pair_count":len(compatible),
      "incompatible_pair_count":len(incompatible),
      "all_pair_ids_preserved":True,
      "pair_ids_sha256":_sha_pairs(pairs),
      "margin_fraction":float(margin_fraction),
      "baseline_tolerance_mm":float(baseline_tolerance_mm),
      "compatible_pair_ids":compatible,
      "incompatible_pairs":incompatible,
      "scope_limit":"Rows are local linear construction guards only; exact F32 full-surface self audit remains mandatory."
    }
