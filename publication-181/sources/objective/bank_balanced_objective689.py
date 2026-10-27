"""Opt-in bank-balanced weights for 689's positive soft plane targets.

This adapter changes only normalized objective weights returned by the frozen
458 row builder. It never changes C/h/G/lower, anchors, boxes, selected axes,
source/trust limits, or exact acceptance gates.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA = "numi.human.bank_balanced_objective689.config.v1"
MAX_ALPHA = 0.5
INSTALLS: list[dict[str, Any]] = []

def _bank(meta: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(meta["body"]), int(meta["bone_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("soft-row metadata lacks typed (body, bone_id)") from exc

def reweight_soft_result(result: dict[str, Any], target_banks: list[tuple[int,int]], alpha: float=0.5):
    """Blend legacy row weights with equal aggregate mass across target banks."""
    a=float(alpha)
    if not np.isfinite(a) or a<0 or a>MAX_ALPHA: raise ValueError(f"alpha must be in [0,{MAX_ALPHA}]")
    weights=np.asarray(result.get("weights"),dtype=np.float64)
    meta=result.get("metadata")
    if weights.ndim!=1 or not np.isfinite(weights).all() or np.any(weights<0): raise ValueError("soft row weights must be finite and nonnegative")
    if not isinstance(meta,list) or len(meta)!=len(weights): raise ValueError("metadata/weight count mismatch")
    if not len(weights): raise ValueError("empty soft objective")
    mass=float(weights.sum())
    if not np.isfinite(mass) or mass<=0: raise ValueError("soft objective needs positive mass")
    requested=tuple(sorted({(int(b),int(k)) for b,k in target_banks}))
    if not requested: raise ValueError("target banks required")
    row_banks=[_bank(r) for r in meta]
    bank_mass={}
    for bank,w in zip(row_banks,weights): bank_mass[bank]=bank_mass.get(bank,0.)+float(w)
    missing=[b for b in requested if bank_mass.get(b,0.)<=0]
    if missing: raise ValueError(f"target banks have no positive movable soft rows: {missing}")
    balanced=np.zeros_like(weights);target_mass=mass/len(requested)
    for i,(bank,w) in enumerate(zip(row_banks,weights)):
        if bank in requested: balanced[i]=target_mass*float(w)/bank_mass[bank]
    out=(1.-a)*weights+a*balanced
    if np.any(out<0) or abs(float(out.sum())-mass)>max(1e-12,mass*1e-12):
        raise RuntimeError("weighting changed total soft objective mass")
    new_meta=[dict(r) for r in meta]
    for i,row in enumerate(new_meta): row["normalized_weight"]=float(out[i])
    result["weights"]=out;result["metadata"]=new_meta
    diag=result.setdefault("diagnostics",{});diag["soft_weight_mass"]=float(out.sum())
    before={f"{b}:{k}":float(v) for (b,k),v in sorted(bank_mass.items())}
    after={}
    for bank,w in zip(row_banks,out): after[bank]=after.get(bank,0.)+float(w)
    diag["bank_balanced_objective689"]={
      "schema":"numi.human.bank_balanced_objective689.result.v1",
      "target_banks":[list(x) for x in requested],"alpha":a,
      "legacy_weight_mass_by_bank":before,
      "weighted_mass_by_bank":{f"{b}:{k}":float(v) for (b,k),v in sorted(after.items())},
      "soft_weight_mass_before":mass,"soft_weight_mass_after":float(out.sum()),
      "modified_fields":["weights","metadata[*].normalized_weight","diagnostics"],
      "unchanged_fields":["C","h","G","lower","guard_metadata","row order"],
      "claim_limit":"Objective reweighting only; no hard positive rows or clearance certificate added.",
      "native_admission":False,"accepted_physical_steps":0}
    return result

def install(owner, objective, crossing: dict[str,Any], config: dict[str,Any]):
    """Wrap the post-axis-selection row builder; fail closed on stale 689 inputs."""
    if config.get("schema")!=SCHEMA: raise ValueError("unsupported config schema")
    target=config.get("target_banks")
    if not isinstance(target,list) or any(not isinstance(x,list) or len(x)!=2 for x in target):
        raise ValueError("target_banks must be explicit [body,bone_id] rows")
    target_banks=[(int(x[0]),int(x[1])) for x in target]
    alpha=float(config.get("alpha",-1.))
    if alpha<=0 or alpha>MAX_ALPHA: raise ValueError("alpha must be in (0,0.5]")
    receipt=crossing.get("receipt",{})
    expected_candidate=config.get("candidate_sha256");expected_audit=config.get("audit159_sha256")
    if receipt.get("candidate_sha256")!=expected_candidate or receipt.get("audit159_sha256")!=expected_audit:
        raise ValueError("config candidate/audit does not match exact crossing input")
    pair_counts={}
    for p in crossing.get("pairs",[]):
        key=(int(p["body"]),int(p["bone_id"]));pair_counts[key]=pair_counts.get(key,0)+1
    declared={(int(x[0]),int(x[1])):int(x[2]) for x in config.get("expected_pair_counts",[])}
    if declared!=pair_counts: raise ValueError("config pair counts differ from exact 159 inventory")
    if any(b not in pair_counts for b in target_banks): raise ValueError("target bank absent from exact inventory")
    if getattr(objective,"_bank_balanced689_installed",False): raise ValueError("adapter already installed")
    old=objective.build_fixed_bone_outward_plane_rows
    state={"schema":"numi.human.bank_balanced_objective689.install.v1","config":config,
      "exact_candidate_sha256":expected_candidate,"exact_audit159_sha256":expected_audit,
      "pair_counts":{f"{b}:{k}":n for (b,k),n in sorted(pair_counts.items())},
      "target_banks":[list(x) for x in sorted(target_banks)],"alpha":alpha,
      "native_admission":False,"accepted_physical_steps":0}
    def wrapped(*args,**kwargs):
        result=old(*args,**kwargs);reweight_soft_result(result,target_banks,alpha)
        policy=result["diagnostics"]["bank_balanced_objective689"]
        state.update({"soft_row_count":int(len(result["weights"])),
          "soft_weight_mass_before":policy["soft_weight_mass_before"],
          "soft_weight_mass_after":policy["soft_weight_mass_after"],
          "weighted_mass_by_bank":policy["weighted_mass_by_bank"],
          "legacy_weight_mass_by_bank":policy["legacy_weight_mass_by_bank"],
          "hard_lower_row_count":int(len(result["lower"])),
          "hard_G_shape":list(result["G"].shape),"soft_C_shape":list(result["C"].shape),
          "soft_h_shape":list(result["h"].shape)})
        return result
    objective.build_fixed_bone_outward_plane_rows=wrapped
    objective._bank_balanced689_installed=True;INSTALLS.append(state)
    return state

def write_result(out_dir: str|Path):
    if not INSTALLS:return None
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    path=out/"bank-balanced-objective-689.json"
    path.write_text(json.dumps(INSTALLS[-1],indent=2,sort_keys=True,allow_nan=False)+"\n")
    return path
