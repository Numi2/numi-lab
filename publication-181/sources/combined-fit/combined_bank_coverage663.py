"""Fail-closed coverage proof for requested positive 458 face rows."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

SCHEMA="numi.human.combined_bank_coverage663.v2"
INSTALLS=[]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def ident(m):
    return tuple(int(m[k]) for k in ("face_row","body","bone_id","triangle_id"))

def validate_rows(result, required, selected_required=True):
    C=result.get("C")
    h=np.asarray(result.get("h"),dtype=np.float64).reshape(-1)
    w=np.asarray(result.get("weights"),dtype=np.float64).reshape(-1)
    meta=result.get("metadata")
    guards=result.get("guard_metadata")
    if C is None or C.shape[0]!=len(h) or len(w)!=len(h) or not isinstance(meta,list) or len(meta)!=len(h):
        raise ValueError("soft face C/h/weights/metadata row counts disagree")
    if not isinstance(guards,list):raise ValueError("458 guard_metadata missing")
    if not np.isfinite(h).all() or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError("soft face rows contain invalid target or weight values")
    req=set(tuple(map(int,x)) for x in required)
    guard_rows={}
    for j,g in enumerate(guards):
        if not isinstance(g,dict):raise ValueError(f"guard row {j} is not an object")
        try:
            key=(*ident(g),int(g["corner_index"]))
            node=int(g["node_row"]); lower=float(g["lower_mm"])
            if not np.isfinite(lower): raise ValueError(f"nonfinite lower bound at guard row {j}")
            anchored=bool(g["anchor_zero_coefficients"])
        except (KeyError,TypeError,ValueError) as e:raise ValueError(f"malformed 458 guard row {j}") from e
        if key[:4] in req:
            if key in guard_rows:raise ValueError(f"duplicate 458 guard corner {key}")
            guard_rows[key]={"node_row":node,"lower_mm":lower,"anchored":anchored}
    guard_missing=[]
    for pair in sorted(req):
        got={c for (i,c) in ((k[:4],k[4]) for k in guard_rows) if i==pair}
        if got!={0,1,2}:guard_missing.append({"pair":list(pair),"corners":sorted(got)})
    if guard_missing:raise ValueError(f"target pairs lack exactly three 458 guard corners: {guard_missing}")
    expected_soft=set();subthreshold=[];anchored_positive=[]
    for key,g in guard_rows.items():
        pair=key[:4];corner=key[4];lo=g["lower_mm"]
        if lo>1e-12:
            if g["anchored"]:anchored_positive.append({"pair":list(pair),"corner":corner,"node_row":g["node_row"],"lower_mm":lo})
            else:expected_soft.add((*pair,corner,g["node_row"]))
        elif 0.0<lo<=1e-12 and not g["anchored"]:
            subthreshold.append({"pair":list(pair),"corner":corner,"node_row":g["node_row"],"lower_mm":lo})
    actual_soft={}; target_row_counts={}
    for i,(m,hi,wi) in enumerate(zip(meta,h,w)):
        if not isinstance(m,dict):raise ValueError(f"soft metadata row {i} is not an object")
        if m.get("kind")=="point":continue
        try:key=(*ident(m),int(m["corner_index"]),int(m["node_row"]))
        except (KeyError,TypeError,ValueError) as e:raise ValueError(f"malformed face soft row {i}") from e
        if hi<=0 or wi<=0:raise ValueError(f"face soft row {i} has nonpositive h or weight")
        if key[:4] in req:
            if key in actual_soft:raise ValueError(f"duplicate face soft row {key}")
            actual_soft[key]=i
            target_row_counts[key[:4]]=target_row_counts.get(key[:4],0)+1
    expected_target={x for x in expected_soft if x[:4] in req}
    actual_target={x for x in actual_soft if x[:4] in req}
    missing=sorted(expected_target-actual_target);extra=sorted(actual_target-expected_target)
    no_soft=sorted(pair for pair in req if not target_row_counts.get(pair,0))
    if selected_required:
        bad=[]
        for key,i in actual_soft.items():
            if key[:4] in req:
                m=meta[i];claimed=m.get("selector_identity")
                try:claimed=tuple(int(x) for x in claimed)
                except (TypeError,ValueError):claimed=()
                if m.get("selector_pair_selected") is not True or claimed!=key[:4]:bad.append(key)
        if bad:raise ValueError(f"requested typed pair rows lack selected 638 axes: {bad}")
    # Receipt survives caller failure so the omitted/subthreshold identities are explicit.
    return {
      "schema":SCHEMA,"required_pair_count":len(req),
      "guard_pair_coverage":len(req)-len(guard_missing),"guard_corner_count":3*len(req),
      "expected_positive_movable_corner_rows":len(expected_target),
      "actual_positive_soft_corner_rows":len(actual_target),
      "covered_pair_count":len(req)-len(no_soft),"pairs_with_no_positive_soft_row":[list(x) for x in no_soft],
      "missing_expected_soft_corners":[list(x) for x in missing],"unexpected_soft_corners":[list(x) for x in extra],
      "subthreshold_positive_corners":[x for x in subthreshold if tuple(x["pair"]) in req],
      "anchored_positive_corners":[x for x in anchored_positive if tuple(x["pair"]) in req],
      "selected_soft_row_coverage":bool(selected_required),
      "rows_per_pair":{",".join(map(str,x)):int(target_row_counts.get(x,0)) for x in sorted(req)},
      "target_bank_pair_counts":{f"{b}:{k}":sum(1 for x in req if x[1:3]==(b,k)) for b,k in sorted({x[1:3] for x in req})},
      "all_requested_pairs_have_positive_soft_rows":not no_soft and not missing and not extra and not anchored_positive and not [x for x in subthreshold if tuple(x["pair"]) in req],
      "native_admission":False,"accepted_physical_steps":0,
    }

def install(objective,crossing,required,expected_candidate_sha256,expected_audit_sha256):
    receipt=crossing.get("receipt",{})
    if receipt.get("candidate_sha256")!=expected_candidate_sha256 or receipt.get("audit159_sha256")!=expected_audit_sha256:
        raise ValueError("crossing identity does not match pinned candidate/audit")
    actual={ident(p) for p in crossing.get("pairs",[])}
    required=set(tuple(map(int,x)) for x in required)
    if len(required)!=359:raise ValueError("expected exactly 359 unique target typed pairs")
    if required-actual:raise ValueError(f"target pair inventory absent from current crossings: {sorted(required-actual)}")
    state={"schema":SCHEMA,"expected_candidate_sha256":expected_candidate_sha256,
       "expected_audit159_sha256":expected_audit_sha256,"required_pair_count":len(required),
       "target_banks":[list(x) for x in sorted({i[1:3] for i in required})],
       "calls":[],"native_admission":False,"accepted_physical_steps":0}
    old=objective.build_fixed_bone_outward_plane_rows
    def wrapped(*args,**kwargs):
        result=old(*args,**kwargs)
        try:summary=validate_rows(result,required,selected_required=True)
        except BaseException as exc:
            state["calls"].append({"status":"FAIL_COVERAGE_CHECK","error_type":type(exc).__name__,"error":str(exc)})
            raise
        state["calls"].append({"status":"PASS_COVERAGE_CHECK","call_index":len(state["calls"]),**summary})
        if not summary["all_requested_pairs_have_positive_soft_rows"]:
            raise ValueError("not all 359 pairs map exactly to positive movable selected soft rows; inspect coverage receipt for subthreshold, anchored, missing, or extra corners")
        return result
    objective.build_fixed_bone_outward_plane_rows=wrapped
    INSTALLS.append(state);return state

def write_result(out_dir):
    if not INSTALLS:return None
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    path=out/"combined-bank-coverage-663.json"
    path.write_text(json.dumps(INSTALLS[-1],indent=2,sort_keys=True,allow_nan=False)+"\n")
    return path

