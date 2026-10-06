#!/usr/bin/env python3
"""Pure-Python FP64 material-point oracle for the current cardboard Hill law.

Synthetic material-point paths only: no FE mesh, board, contact, or calibration.
Uses only Python's standard library.
"""
from __future__ import annotations
import hashlib, json, math, re, sys
from pathlib import Path

# Locate the repository checkout containing this archived evidence.
ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "matter" / "materials").is_dir())

MATERIALS = {
    "liner": {
        "sigma_ref": 22.92,
        "C": [[4843.6261790411845,381.24384915298583,0.22861936478052688],
              [381.24384915298583,2118.009773355429,0.20899490085830671],
              [0.22861936478052688,0.20899490085830671,19.100028262603168]],
        "G": {"12":1222.0,"13":166.28,"23":137.28},
        "hill": {"F":6.307660915935419,"G":2.981864887595591,"H":-1.9818648875955915,
                 "L":456012.5,"M":456012.5,"N":59.56081632653061},
    },
    "medium": {
        "sigma_ref": 9.391,
        "C": [[4524.848543681839,293.59114732344125,0.21371636195674576],
              [293.59114732344125,1631.0512147406462,0.19286701249506086],
              [0.21371636195674576,0.19286701249506086,17.90002997076348]],
        "G": {"12":1039.0,"13":226.34,"23":198.72},
        "hill": {"F":5.612051993569077,"G":2.841099191623919,"H":-1.8410991916239194,
                 "L":76554.58420138889,"M":76554.58420138889,"N":9.998966099773243},
    },
}
Q_FLOOR = 2.0e-6
FB_EPS = 1.0e-8
ZERO = [0.0] * 6


def q_and_flow(s, p):
    h = p["hill"]
    s11,s22,s33,s23,s13,s12 = s
    q2 = (h["F"]*(s22-s33)**2 + h["G"]*(s33-s11)**2 +
          h["H"]*(s11-s22)**2 + 2*h["L"]*s23*s23 +
          2*h["M"]*s13*s13 + 2*h["N"]*s12*s12 + Q_FLOOR**2)
    if q2 <= 0.0:
        raise ArithmeticError(f"non-positive Hill radicand {q2}")
    q = math.sqrt(q2)
    flow = [
        (h["G"]*(s11-s33)+h["H"]*(s11-s22))/q,
        (h["F"]*(s22-s33)+h["H"]*(s22-s11))/q,
        (h["F"]*(s33-s22)+h["G"]*(s33-s11))/q,
        h["L"]*s23/q,
        h["M"]*s13/q,
        h["N"]*s12/q,
    ]
    return q, flow


def fb(a, b):
    # Algebraically equivalent stable evaluation of sqrt(a^2+b^2+eps^2)-a-b.
    root = math.sqrt(a*a + b*b + FB_EPS*FB_EPS)
    denom = root + a + b
    if denom > 1.0e-150:
        return (FB_EPS*FB_EPS - 2.0*a*b) / denom
    return root - a - b


def solve_linear(a, b):
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for k in range(n):
        pivot = max(range(k,n), key=lambda i: abs(m[i][k]))
        if abs(m[pivot][k]) < 1.0e-24:
            raise ArithmeticError("singular Newton Jacobian")
        m[k], m[pivot] = m[pivot], m[k]
        inv = 1.0/m[k][k]
        for j in range(k,n+1): m[k][j] *= inv
        for i in range(n):
            if i == k: continue
            f = m[i][k]
            if f == 0.0: continue
            for j in range(k,n+1): m[i][j] -= f*m[k][j]
    return [m[i][n] for i in range(n)]


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def residual(x, ep0, strain, p):
    s = x[:6]
    dl = x[6]
    q, flow = q_and_flow(s,p)
    ep = [ep0[i] + dl*flow[i] for i in range(6)]
    e11,e22,e33,g23,g13,g12 = strain
    elastic_normal = [e11-ep[0],e22-ep[1],e33-ep[2]]
    c = p["C"]
    s_pred = [sum(c[i][j]*elastic_normal[j] for j in range(3))/p["sigma_ref"]
              for i in range(3)]
    g = p["G"]
    s_pred.extend([
        g["23"]*(g23-2.0*ep[3])/p["sigma_ref"],
        g["13"]*(g13-2.0*ep[4])/p["sigma_ref"],
        g["12"]*(g12-2.0*ep[5])/p["sigma_ref"],
    ])
    b = 1.0-q
    return [s[i]-s_pred[i] for i in range(6)] + [fb(dl,b)]


def constitutive_step(ep0, strain, p, previous_stress=None):
    # Recompute the elastic trial at this step's current strain and old Ep.
    e11,e22,e33,g23,g13,g12 = strain
    en = [e11-ep0[0],e22-ep0[1],e33-ep0[2]]
    c = p["C"]
    s = [sum(c[i][j]*en[j] for j in range(3))/p["sigma_ref"] for i in range(3)]
    for name, gij, k in (("23",g23,3),("13",g13,4),("12",g12,5)):
        s.append(p["G"][name]*(gij-2.0*ep0[k])/p["sigma_ref"])
    qtrial,_ = q_and_flow(s,p)
    if qtrial > 1.0:
        dl_guess = max(1.0e-12,min(0.05,(qtrial-1.0)*0.005))
    else:
        dl_guess = FB_EPS*FB_EPS/(2.0*max(1.0-qtrial,FB_EPS))
    x = s + [dl_guess]
    # A forward-difference damped Newton solve of the seven exact local equations.
    for iteration in range(80):
        r = residual(x,ep0,strain,p)
        rn = norm(r)
        if rn < 2.0e-13:
            q,flow = q_and_flow(x[:6],p)
            ep1 = [ep0[i]+x[6]*flow[i] for i in range(6)]
            return ep1, x[:6], x[6], q, iteration+1, rn
        jac = [[0.0]*7 for _ in range(7)]
        for j in range(7):
            h = (2.0e-7 if j < 6 else 1.0e-8) * max(1.0,abs(x[j]))
            xp = list(x); xp[j] += h
            rp = residual(xp,ep0,strain,p)
            for i in range(7): jac[i][j] = (rp[i]-r[i])/h
        dx = solve_linear(jac,[-v for v in r])
        accepted = False
        alpha = 1.0
        for _ in range(60):
            trial = [x[i]+alpha*dx[i] for i in range(7)]
            if trial[6] >= 0.0:
                rt = residual(trial,ep0,strain,p)
                if norm(rt) < rn:
                    x=trial; accepted=True; break
            alpha *= 0.5
        if not accepted:
            raise ArithmeticError(f"line search failed iter={iteration}, residual={rn}, x={x}, r={r}")
    raise ArithmeticError(f"Newton failed; residual={norm(residual(x,ep0,strain,p))}, x={x}")


def run_path(p, n, path):
    ep=ZERO[:]; stress=None; increments=[]
    if path == "held_md_compression":
        target=[-0.010,0.0,0.0,0.0,0.0,0.0]
        for i in range(1,n+1):
            strain=[target[j]*i/n for j in range(6)]
            ep,stress,dl,q,it,r=constitutive_step(ep,strain,p,stress)
            increments.append(dl)
        loaded={"strain":target,"ep":ep[:],"stress_MPa":[v*p["sigma_ref"] for v in stress],"lambda":sum(increments),"q":q}
        hold_lambda=0.0
        stress_before=stress[:]; ep_before=ep[:]
        for _ in range(n):
            ep,stress,dl,q,it,r=constitutive_step(ep,target,p,stress)
            increments.append(dl); hold_lambda += dl
        held={"ep":ep[:],"stress_MPa":[v*p["sigma_ref"] for v in stress],"lambda":sum(increments),"q":q,
              "hold_delta_lambda":hold_lambda,
              "hold_delta_ep_norm":norm([ep[i]-ep_before[i] for i in range(6)]),
              "hold_delta_stress_norm_MPa":norm([(stress[i]-stress_before[i])*p["sigma_ref"] for i in range(6)])}
        return {"loaded":loaded,"held":held,"min_delta_lambda":min(increments),"negative_delta_lambda_count":sum(v < -1e-15 for v in increments),"iterations_max":None}
    elif path == "proportional_md_compression_plus_cd_shear_cycle":
        target=[-0.006,0.0,0.0,0.0,0.0,0.004]
        for i in range(1,n+1):
            strain=[target[j]*i/n for j in range(6)]
            ep,stress,dl,q,it,r=constitutive_step(ep,strain,p,stress)
            increments.append(dl)
        loaded={"strain":target,"ep":ep[:],"stress_MPa":[v*p["sigma_ref"] for v in stress],"lambda":sum(increments),"q":q}
        epmax=ep[:]; stressmax=stress[:]; load_lambda=sum(increments)
        for i in range(1,n+1):
            strain=[target[j]*(1.0-i/n) for j in range(6)]
            ep,stress,dl,q,it,r=constitutive_step(ep,strain,p,stress)
            increments.append(dl)
        unloaded={"strain":[0.0]*6,"ep":ep[:],"stress_MPa":[v*p["sigma_ref"] for v in stress],"lambda":sum(increments),"q":q,
                  "unload_delta_lambda":sum(increments[n:]),
                  "unload_ep_drift_norm":norm([ep[i]-epmax[i] for i in range(6)]),
                  "residual_stress_norm_MPa":norm([v*p["sigma_ref"] for v in stress])}
        return {"loaded":loaded,"unloaded":unloaded,"min_delta_lambda":min(increments),"negative_delta_lambda_count":sum(v < -1e-15 for v in increments)}
    else: raise ValueError(path)


def main():
    source_copies = {
        "liner": ROOT/"matter/examples/cardboard/evidence/structured-paper-and-glue/paper-calibration-009/liner.nmatter",
        "medium": ROOT/"matter/examples/cardboard/evidence/structured-paper-and-glue/paper-calibration-009/medium.nmatter",
    }
    canonical = {
        "liner": ROOT/"matter/materials/hajali2009_liner_hill_ideal.nmatter",
        "medium": ROOT/"matter/materials/hajali2009_medium_hill_ideal.nmatter",
    }
    source_metadata={}
    for name, path in source_copies.items():
        code=re.sub(r"//[^\n]*","",path.read_text())
        code_compact=re.sub(r"\s+","",code)
        canonical_code=re.sub(r"//[^\n]*","",canonical[name].read_text())
        canonical_compact=re.sub(r"\s+","",canonical_code)
        source_metadata[name]={
            "material_copy_path":str(path),
            "material_copy_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "canonical_material_path":str(canonical[name]),
            "canonical_material_sha256":hashlib.sha256(canonical[name].read_bytes()).hexdigest(),
            "copy_and_canonical_math_equal_after_comment_and_whitespace_removal":code_compact==canonical_compact,
            "explicit_dt_time_or_rate_token_in_code":bool(re.search(r"\b(dt|time|rate)\b",code,re.IGNORECASE)),
        }
    out={"title":"FP64 local Hill law material-point path audit","scope":"synthetic constitutive material point only; not a board, mesh, or physical diagnosis",
         "python_runtime":{"version":sys.version.split()[0],"float_mantissa_bits":sys.float_info.mant_dig,"float_epsilon":sys.float_info.epsilon,"dependencies":"Python standard library only"},
         "implementation":"independent Python standard-library double-precision solve of stress/Ep/DeltaLambda equations transcribed from the exact scale-10 material-copy equations; forward-difference damped Newton, local residual norm target 2e-13; stresses reported in MPa",
         "source_materials":source_metadata,
         "material_parameters":MATERIALS,
         "law_audit":{
             "stress_law":"normalized StVK second Piola stress from current Green strain minus additive Ep; diagonal stress couples through the symmetric 3x3 C matrix; shear stresses are Gij*(Cij-2*Epij)/sigma_ref",
             "hill_measure":"q=sqrt(Hill quadratic normalized stress + q_floor^2)",
             "flow_update":"Ep_next=Ep_old+DeltaLambda*HillFlow(sigma_bar_next,q_next)",
             "smooth_complementarity":"sqrt(DeltaLambda^2+(1-q)^2+fb_epsilon^2)-DeltaLambda-(1-q)=0",
             "explicit_dt_or_strain_rate_dependence":False,
             "rate_independence_limit":"no physical time or strain-rate argument occurs in the local constitutive equations; subdivision differences arise from incremental backward-Euler path integration and the smooth complementarity regularizer, not a rate law",
             "residual_scale_note":"all 13 implicit residual rows use residual_scale=10; multiplying equations leaves the exact root unchanged but can change finite-tolerance solver error",
             "monotonicity":"the oracle enforces DeltaLambda >= 0 in line search; positive increments, including tiny subyield smoothing increments, are expected. This checks mathematical increment monotonicity only, not GPU state transfer.",
         },
         "synthetic_paths":{
             "held_md_compression":"Green E11 increases linearly from 0 to -0.010 over n equal strain increments, then remains fixed for n updates; each update is a rate-independent local material update, with no physical hold-time assigned",
             "proportional_md_compression_plus_cd_shear_cycle":"Green E11 and engineering C12 increase proportionally from (0,0) to (-0.006,0.004) over n equal increments, then are strain-controlled back to (0,0) over n increments; final residual stress is therefore for prescribed zero total strain, not force-free relaxation",
         },
         "regularization":{"hill_q_floor":Q_FLOOR,"smooth_FB_epsilon":FB_EPS},
         "native_temporal_refinement_context_from_requesting_agent":{
             "source_status":"values supplied by parent for diagnosis; not recomputed by this CPU oracle",
             "coarse_dt_us":25.0,"coarse_steps":144,"fine_dt_us":12.5,"fine_steps":288,
             "protocol_duration_ms":3.6,"protocol":"clamped",
             "reported_endpoint_moment_difference_percent_range":[19.0,24.0],
             "reported_weighted_EP_RMS_final_coarse":0.00082457,
             "reported_weighted_EP_RMS_final_fine":0.000695219,
             "reported_weighted_EP_RMS_relative_change_percent":15.688,
         },
         "subdivisions":[32,64,128],"materials":{}}
    for name,p in MATERIALS.items():
        out["materials"][name]={}
        for path in ("held_md_compression","proportional_md_compression_plus_cd_shear_cycle"):
            out["materials"][name][path]={str(n):run_path(p,n,path) for n in (32,64,128)}
            runs=out["materials"][name][path]
            phases=("loaded","held") if path=="held_md_compression" else ("loaded","unloaded")
            comparisons={}
            for phase in phases:
                ref=runs["128"][phase]
                ref_stress_norm=norm(ref["stress_MPa"])
                ref_ep_norm=norm(ref["ep"])
                comparisons[phase]={}
                for n in (32,64):
                    value=runs[str(n)][phase]
                    stress_error=norm([value["stress_MPa"][i]-ref["stress_MPa"][i] for i in range(6)])
                    ep_error=norm([value["ep"][i]-ref["ep"][i] for i in range(6)])
                    lambda_error=abs(value["lambda"]-ref["lambda"])
                    comparisons[phase][str(n)+"_vs_128"]={
                        "stress_vector_L2_error_MPa":stress_error,
                        "stress_relative_error_percent":100.0*stress_error/max(ref_stress_norm,1.0e-300),
                        "plastic_strain_vector_L2_error":ep_error,
                        "plastic_strain_relative_error_percent":100.0*ep_error/max(ref_ep_norm,1.0e-300),
                        "accumulated_multiplier_absolute_error":lambda_error,
                        "accumulated_multiplier_relative_error_percent":100.0*lambda_error/max(abs(ref["lambda"]),1.0e-300),
                    }
            runs["subdivision_comparison_to_128"] = comparisons
            runs["monotonicity_check"] = {
                "all_increments_nonnegative_within_1e-15": all(runs[str(n)]["negative_delta_lambda_count"]==0 for n in (32,64,128)),
                "minimum_increment_by_subdivision": {str(n):runs[str(n)]["min_delta_lambda"] for n in (32,64,128)},
            }
    max64_stress=max(v["64_vs_128"]["stress_relative_error_percent"]
                     for m in out["materials"].values() for p in m.values()
                     for v in p["subdivision_comparison_to_128"].values())
    max64_ep=max(v["64_vs_128"]["plastic_strain_relative_error_percent"]
                 for m in out["materials"].values() for p in m.values()
                 for v in p["subdivision_comparison_to_128"].values())
    max64_lambda=max(v["64_vs_128"]["accumulated_multiplier_relative_error_percent"]
                     for m in out["materials"].values() for p in m.values()
                     for v in p["subdivision_comparison_to_128"].values())
    max32_ep=max(v["32_vs_128"]["plastic_strain_relative_error_percent"]
                 for m in out["materials"].values() for p in m.values()
                 for v in p["subdivision_comparison_to_128"].values())
    out["interpretation_and_limits"]={
        "same_path_twofold_subdivision_max_relative_differences_percent":{
            "stress_vector":max64_stress,"plastic_strain_vector":max64_ep,
            "accumulated_multiplier":max64_lambda,
        },
        "same_path_32_vs_128_max_plastic_strain_vector_relative_difference_percent":max32_ep,
        "held_near_yield_smooth_FB_drift_128_updates":{
            "liner_delta_lambda":out["materials"]["liner"]["held_md_compression"]["128"]["held"]["hold_delta_lambda"],
            "liner_delta_Ep_norm":out["materials"]["liner"]["held_md_compression"]["128"]["held"]["hold_delta_ep_norm"],
            "liner_delta_stress_norm_MPa":out["materials"]["liner"]["held_md_compression"]["128"]["held"]["hold_delta_stress_norm_MPa"],
            "medium_delta_lambda":out["materials"]["medium"]["held_md_compression"]["128"]["held"]["hold_delta_lambda"],
            "medium_delta_Ep_norm":out["materials"]["medium"]["held_md_compression"]["128"]["held"]["hold_delta_ep_norm"],
            "medium_delta_stress_norm_MPa":out["materials"]["medium"]["held_md_compression"]["128"]["held"]["hold_delta_stress_norm_MPa"],
        },
        "assessment":"The local update is rate-independent and has only small same-material-point subdivision differences on these prescribed paths. At held active yield, smooth FB epsilon causes tiny positive plastic drift that grows with update count. These oracle effects are much smaller than the parent-reported board endpoint differences, so this evidence does not support attributing the FE change to the local constitutive law alone.",
        "not_a_diagnosis_of":"board-level deformation-history changes, contact activation, changing Newton convergence, line-search choices, local GPU implicit-root tolerance, mesh/geometry, or physical material behavior",
    }
    path=Path(__file__).with_name("constitutive-path-audit.json")
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(path)
    for material, paths in out["materials"].items():
        for name, runs in paths.items():
            print(material,name)
            for n in ("32","64","128"):
                v=runs[n]
                key="held" if "held" in v else "unloaded"
                result=v[key]
                print(n,"lambda",result["lambda"],"q",result["q"],
                      "stress",result.get("hold_delta_stress_norm_MPa",result.get("residual_stress_norm_MPa")),
                      "ep_drift",result.get("hold_delta_ep_norm",result.get("unload_ep_drift_norm")),
                      "dlphase",result.get("hold_delta_lambda",result.get("unload_delta_lambda")),
                      "min_dl",v["min_delta_lambda"],"neg",v["negative_delta_lambda_count"])
if __name__=="__main__": main()
