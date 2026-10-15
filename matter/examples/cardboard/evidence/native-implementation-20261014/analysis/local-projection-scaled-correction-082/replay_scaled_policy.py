#!/usr/bin/env python3
"""FP64 captured-state replay of a candidate-scaled local Newton stop/solve policy.

This is an independent equation replay, not production/native evidence. The
residual equation, authored L-infinity Armijo merit, and source coefficients
are unchanged. The experimental policy requires both the existing authored
residual criterion and a small componentwise Newton correction in dimensionless
state units; the Newton system is solved with diagonal row/column scaling.
"""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np

BASE = Path("/Users/home/cardboard-implementation-20261014")
ANALYSIS = BASE / "analysis/bilinear15-fp64-replay-267"
sys.path.insert(0, str(ANALYSIS))
import bilinear15_reference as ref

N0_CAPTURE = BASE / "solver/sourcefit15-fg32-control1-capture-cycle-258/operator-capture"
N63_CAPTURE = BASE / "solver/sourcefit15-fg32-control1-newton63-capture-266/operator-capture"
N0_ARRAYS = ANALYSIS / "early-output/arrays.npz"
N63_STRICT_ROWS = ANALYSIS / "late-output-002/rows.npz"
OUT = Path(__file__).parent

RESIDUAL_TOL = 2.0e-8       # authored production local tolerance, unchanged
CORRECTION_TOL = 2.0e-8     # scoped dimensionless backward-correction bound
MAXITER = 32                # no relaxation; explicit bounded CPU replay
MAX_BACKTRACK = 16           # production bounded Armijo count
SCALE_FLOOR_MULTIPLIER = 1.0 # the authored positive FB epsilon

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def residual_norm(p, f, old, z):
    r = ref.residual(p, f, old, z)
    return float(10.0 * np.max(np.abs(r))), float(np.max(np.abs(z))), r

def solve_scaled(J, r, old, z, eps):
    # All state variables in the retained 15-state program are dimensionless.
    # The authored positive FB epsilon supplies a source-bound scale floor.
    xs = np.maximum(np.maximum(np.abs(old), np.abs(z)), SCALE_FLOOR_MULTIPLIER * eps)
    if not np.isfinite(xs).all() or np.any(xs <= 0):
        raise FloatingPointError("invalid state scale")
    row = np.sum(np.abs(J) * xs[None, :], axis=1)
    if not np.isfinite(row).all() or np.any(row <= 0):
        raise FloatingPointError("invalid row scale")
    A = (J * xs[None, :]) / row[:, None]
    b = -r / row
    y = np.linalg.solve(A, b)
    d = xs * y
    correction = float(np.max(np.abs(d) / xs))
    if not np.isfinite(d).all() or not np.isfinite(correction):
        raise FloatingPointError("nonfinite scaled correction")
    return d, correction, xs, row

def project_scaled(p, f, old, zinit):
    z = np.array(zinit, dtype=np.float64, copy=True)
    eps = float(p[26])
    for it in range(MAXITER + 1):
        rn, sn, r = residual_norm(p, f, old, z)
        if not np.isfinite(rn):
            raise FloatingPointError("nonfinite residual")
        J = ref.jacobian(p, f, old, z)
        d, cn, xs, row = solve_scaled(J, r, old, z, eps)
        residual_ok = rn <= RESIDUAL_TOL * (1.0 + sn)
        correction_ok = cn <= CORRECTION_TOL
        if residual_ok and correction_ok:
            return z, {
                "status": "converged",
                "iterations": it,
                "residual_inf_scaled": rn,
                "authored_residual_threshold": RESIDUAL_TOL * (1.0 + sn),
                "state_inf": sn,
                "scaled_correction_inf": cn,
                "correction_threshold": CORRECTION_TOL,
                "minimum_variable_scale": float(xs.min()),
                "maximum_variable_scale": float(xs.max()),
                "minimum_row_scale": float(row.min()),
                "maximum_row_scale": float(row.max()),
            }
        if it == MAXITER:
            return z, {
                "status": "iteration_exhausted",
                "iterations": it,
                "residual_inf_scaled": rn,
                "authored_residual_threshold": RESIDUAL_TOL * (1.0 + sn),
                "state_inf": sn,
                "scaled_correction_inf": cn,
                "correction_threshold": CORRECTION_TOL,
            }
        accepted = False
        alpha = 1.0
        for bt in range(MAX_BACKTRACK):
            trial = z + alpha * d
            trn, _, _ = residual_norm(p, f, old, trial)
            if np.isfinite(trn) and trn <= (1.0 - 1.0e-4 * alpha) * rn:
                z = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return z, {
                "status": "armijo_no_candidate",
                "iterations": it,
                "residual_inf_scaled": rn,
                "authored_residual_threshold": RESIDUAL_TOL * (1.0 + sn),
                "state_inf": sn,
                "scaled_correction_inf": cn,
                "correction_threshold": CORRECTION_TOL,
            }
    raise AssertionError("unreachable")

def project_existing(p, f, old, zinit, tol=RESIDUAL_TOL):
    # Replays the existing production stop rule from the exact captured local
    # candidate. This is independent FP64 equation code from the existing
    # diagnosis utility; it preserves the original unweighted L-infinity merit.
    z = np.array(zinit, dtype=np.float64, copy=True)
    for it in range(MAXITER + 1):
        rn, sn, r = residual_norm(p, f, old, z)
        if rn <= tol * (1.0 + sn):
            return z, {"status":"converged","iterations":it,"residual_inf_scaled":rn,
                       "authored_residual_threshold":tol*(1+sn),"state_inf":sn}
        if it == MAXITER: break
        J = ref.jacobian(p, f, old, z)
        d = np.linalg.solve(J, -r)
        alpha = 1.0
        accepted = False
        for _ in range(MAX_BACKTRACK):
            trial = z + alpha*d
            trn, _, _ = residual_norm(p, f, old, trial)
            if np.isfinite(trn) and trn <= (1.0-1.0e-4*alpha)*rn:
                z = trial
                accepted = True
                break
            alpha *= .5
        if not accepted:
            return z, {"status":"armijo_no_candidate","iterations":it,"residual_inf_scaled":rn,
                       "authored_residual_threshold":tol*(1+sn),"state_inf":sn}
    return z, {"status":"iteration_exhausted","iterations":MAXITER}

def material_rows(capture):
    records = [json.loads(line) for line in Path(capture).read_text().splitlines() if line.strip()]
    header = records[0]
    mats = {int(x["index"]):x for x in header["materials"]}
    params = header["parameters_by_environment"]
    selected = []
    for rec in records:
        if rec.get("record") != "tetrahedron" or rec.get("material_index") not in (0, 1):
            continue
        c = rec["center"]
        env = int(rec["environment"])
        mi = int(rec["material_index"])
        m = mats[mi]
        start = int(m["parameter_offset"])
        count = int(rec["material_parameter_count"])
        p = np.asarray(params[env][start:start+count], dtype=np.float32).astype(np.float64)
        f = np.asarray(c["F_local"], dtype=np.float32).astype(np.float64) + np.asarray(c["F_local_low"], dtype=np.float32).astype(np.float64)
        old = np.asarray(c["accepted_state"], dtype=np.float32).astype(np.float64)
        z = np.asarray(c["projected_state"], dtype=np.float32).astype(np.float64) + np.asarray(c["projected_state_low"], dtype=np.float32).astype(np.float64)
        h = np.asarray(c["F_direction_local_per_s"], dtype=np.float32).astype(np.float64)
        native_t = np.asarray(c["tangent_local_Pa_per_s"], dtype=np.float32).astype(np.float64)
        selected.append({"tet_id":int(rec["tet_id"]),"environment":env,"material_index":mi,
                         "p":p,"F":f,"old":old,"candidate":z,"direction":h,"native_tangent":native_t,
                         "projection_report":c["projection"]})
    return header, selected

def tangent_metrics(rows, roots, label):
    native_sq=0.0; root_sq=0.0; delta_sq=0.0
    max_tet_rel=0.0; max_tet_abs=0.0; rows_with_tangent=0
    per=[]
    for row,z in zip(rows,roots):
        t = ref.tangent(row["p"],row["F"],row["old"],z,row["direction"])
        n = row["native_tangent"]
        diff = t-n
        norm_t=float(np.linalg.norm(t)); norm_n=float(np.linalg.norm(n)); norm_d=float(np.linalg.norm(diff))
        if max(norm_t,norm_n)>1e-20:
            rows_with_tangent+=1
        rel=norm_d/max(norm_n,1e-300)
        max_tet_rel=max(max_tet_rel,rel)
        max_tet_abs=max(max_tet_abs,float(np.max(np.abs(diff))))
        native_sq += norm_n**2
        root_sq += norm_t**2
        delta_sq += norm_d**2
        per.append({"tet_id":row["tet_id"],"environment":row["environment"],"material_index":row["material_index"],
                    "relative_l2_to_native":rel,"absolute_max_component_to_native":float(np.max(np.abs(diff))),
                    "native_max_component":float(np.max(np.abs(n)))})
    return {"label":label,"count":len(rows),"rows_with_tangent":rows_with_tangent,
            "global_l2_relative_to_native":float(np.sqrt(delta_sq)/max(np.sqrt(native_sq),1e-300)),
            "global_l2_relative_to_reference":float(np.sqrt(delta_sq)/max(np.sqrt(root_sq),1e-300)),
            "max_per_tet_l2_relative_to_native":max_tet_rel,"max_abs_component_error":max_tet_abs,
            "worst_rows":sorted(per,key=lambda r:r["relative_l2_to_native"],reverse=True)[:12],
            "per_row":per}

def run_capture(capture, label, strict_roots=None):
    header, rows = material_rows(capture)
    old_roots=[]; new_roots=[]; old_results=[]; new_results=[]
    for row in rows:
        z0=row["candidate"]
        zo, ro=project_existing(row["p"],row["F"],row["old"],z0)
        zn, rn=project_scaled(row["p"],row["F"],row["old"],z0)
        old_roots.append(zo);new_roots.append(zn);old_results.append(ro);new_results.append(rn)
    if strict_roots is None:
        strict_roots = [project_existing(row["p"],row["F"],row["old"],row["candidate"],tol=2e-14)[0]
                        for row in rows]
    old_metric=tangent_metrics(rows,old_roots,f"{label}: existing authored residual stop")
    new_metric=tangent_metrics(rows,new_roots,f"{label}: authored residual plus scaled correction")
    strict_metric=tangent_metrics(rows,strict_roots,f"{label}: strict FP64 reference (2e-14)")
    counts={}
    for result in new_results:
        counts[result["status"]]=counts.get(result["status"],0)+1
    n0delta=np.array([np.max(np.abs(a-b)) for a,b in zip(new_roots,strict_roots)])
    relative_root=[]
    for a,b in zip(new_roots,strict_roots):
        ta=ref.tangent(rows[len(relative_root)]["p"],rows[len(relative_root)]["F"],
                       rows[len(relative_root)]["old"],a,rows[len(relative_root)]["direction"])
        tb=ref.tangent(rows[len(relative_root)]["p"],rows[len(relative_root)]["F"],
                       rows[len(relative_root)]["old"],b,rows[len(relative_root)]["direction"])
        relative_root.append(float(np.linalg.norm(ta-tb)/max(np.linalg.norm(tb),1e-300)))
    worst=[dict({"tet_id":rows[i]["tet_id"],"environment":rows[i]["environment"],
                 "material_index":rows[i]["material_index"],
                 "old_result":old_results[i],"new_result":new_results[i],
                 "state_inf_difference_to_strict":float(n0delta[i]),
                 "tangent_relative_l2_to_strict":relative_root[i]},**{}) for i in
           np.argsort(np.asarray(relative_root))[::-1][:12]]
    return {"label":label,"capture":str(capture),"capture_sha256":sha(capture),
            "header":{"root":header.get("root"),"newton_iteration":header.get("newton_iteration"),
                      "world_fingerprint_hex":header.get("world_fingerprint_hex"),
                      "execution_fingerprint_hex":header.get("execution_fingerprint_hex"),
                      "environment_count":header.get("environment_count"),
                      "epsilon_values":header.get("epsilon_values"),
                      "dt_seconds_by_environment":header.get("dt_seconds_by_environment")},
            "tetrahedra":len(rows),"new_policy_status_counts":counts,
            "new_policy_iteration_distribution":{"mean":float(np.mean([x["iterations"] for x in new_results])),
                "p95":float(np.percentile([x["iterations"] for x in new_results],95)),
                "max":int(max(x["iterations"] for x in new_results))},
            "max_new_policy_authored_residual_fraction":float(max(
                x.get("residual_inf_scaled",0.0)/max(x.get("authored_residual_threshold",1e-300),1e-300)
                for x in new_results)),
            "max_new_policy_correction_fraction":float(max(
                x.get("scaled_correction_inf",0.0)/CORRECTION_TOL for x in new_results)),
            "native_vs_existing":old_metric,"native_vs_scaled_policy":new_metric,
            "native_vs_strict_reference":strict_metric,
            "scaled_policy_vs_strict":{"global_l2_relative_tangent":float(
                np.sqrt(sum(np.linalg.norm(ref.tangent(r["p"],r["F"],r["old"],a,r["direction"])-
                                       ref.tangent(r["p"],r["F"],r["old"],b,r["direction"]))**2
                             for r,a,b in zip(rows,new_roots,strict_roots))) /
                max(np.sqrt(sum(np.linalg.norm(ref.tangent(r["p"],r["F"],r["old"],b,r["direction"]))**2
                                for r,b in zip(rows,strict_roots))),1e-300)),
                "max_per_tet_l2_relative_tangent":max(relative_root)},
            "worst_scaled_policy_vs_strict":worst}

def main():
    start=time.time()
    n0=run_capture(N0_CAPTURE,"N0 control 1")
    # Use sealed existing N63 strict-root states/tangents for exact comparison
    # while independently rerunning the new projection from the N63 capture.
    late=np.load(N63_STRICT_ROWS)
    _,n63rows=material_rows(N63_CAPTURE)
    n63roots=[]
    for i,row in enumerate(n63rows):
        z,info=project_scaled(row["p"],row["F"],row["old"],row["candidate"])
        n63roots.append(z)
    # To avoid recomputing/reference-drifting the strict N63 root, compare
    # directly against the preserved strict tangent and its source history.
    n63_native=np.asarray([r["native_tangent"] for r in n63rows])
    # late-output report has per-row native/strict tangents and converged count.
    n63_fp=np.asarray(late["fp64_refined"])
    n63_diff=n63_fp-n63_native
    n63_rel=float(np.linalg.norm(n63_diff)/max(np.linalg.norm(n63_native),1e-300))
    n63_tangent=[]
    for row,z in zip(n63rows,n63roots):
        n63_tangent.append(ref.tangent(row["p"],row["F"],row["old"],z,row["direction"]))
    n63_scaled_rel=float(np.linalg.norm(np.asarray(n63_tangent)-n63_fp)/max(np.linalg.norm(n63_fp),1e-300))
    n63_per=[]
    for row,t,strict in zip(n63rows,n63_tangent,n63_fp):
        n63_per.append({"tet_id":row["tet_id"],"environment":row["environment"],
                        "material_index":row["material_index"],
                        "relative_l2_to_strict":float(np.linalg.norm(t-strict)/max(np.linalg.norm(strict),1e-300)),
                        "absolute_max_component_to_strict":float(np.max(np.abs(t-strict)))})
    n63_results=[]
    for row in n63rows:
        _,ri=project_scaled(row["p"],row["F"],row["old"],row["candidate"])
        n63_results.append(ri)
    n63counts={}
    for x in n63_results: n63counts[x["status"]]=n63counts.get(x["status"],0)+1
    lateReport=json.load(open(N63_STRICT_ROWS.parent/'report.json'))
    out={
      "schema":"numi.cardboard.bilinear15-scaled-local-projection-replay.v1",
      "classification":"Independent captured-state FP64 equation replay only. No production/native patch execution, no material or global solver qualification.",
      "policy":{"authored_residual_tolerance":RESIDUAL_TOL,"authored_residual_norm":"unweighted L-infinity over all 15 authored rows after the source residual_scale=10","row_scaling":"row i divided by sum_j |J_ij|*xscale_j for Newton solve only","variable_scale":"max(|acceptedState_j|,|candidateState_j|,fb_epsilon)","scaled_correction_gate":"max_j |delta_j|/xscale_j <= 2e-8","line_search":"same authored residual L-infinity Armijo decrease; 16 bounded backtracks","max_iterations":MAXITER,"accepted_history":"read-only old state; only the projected candidate is adjusted"},
      "n0":n0,
      "n63":{"label":"N63 control 1","capture":str(N63_CAPTURE),"capture_sha256":sha(N63_CAPTURE),
             "header":{"root":1,"newton_iteration":63,"world_fingerprint_hex":"see capture header",
                       "epsilon_values":[0.03125,0.015625],"dt_seconds":9.99999975e-05},
             "tetrahedra":len(n63rows),"new_policy_status_counts":n63counts,
             "new_policy_iteration_distribution":{"mean":float(np.mean([x["iterations"] for x in n63_results])),
                "p95":float(np.percentile([x["iterations"] for x in n63_results],95)),
                "max":int(max(x["iterations"] for x in n63_results))},
             "native_vs_strict_reference_global_l2_relative":n63_rel,
             "scaled_policy_vs_strict_reference_global_l2_relative":n63_scaled_rel,
             "max_scaled_policy_vs_strict_reference_per_tet_l2_relative":max(x["relative_l2_to_strict"] for x in n63_per),
             "worst_scaled_policy_vs_strict_rows":sorted(n63_per,key=lambda x:x["relative_l2_to_strict"],reverse=True)[:12],
             "strict_reference_report_sha256":sha(N63_STRICT_ROWS.parent/'report.json'),
             "strict_reference_rows_sha256":sha(N63_STRICT_ROWS)},
      "source_inputs":{"n0_capture_sha256":sha(N0_CAPTURE),"n63_capture_sha256":sha(N63_CAPTURE),
          "n0_arrays_sha256":sha(N0_ARRAYS),"n63_strict_rows_sha256":sha(N63_STRICT_ROWS),
          "reference_sha256":sha(ANALYSIS/'bilinear15_reference.py'),
          "snapshot081_common_metalinc_sha256":sha(Path("/Users/home/cardboard-implementation-20261014/snapshots/source-portable-bct-binding-tests-081/matter/src/metal/common.metalinc"))},
      "elapsed_seconds":time.time()-start}
    (OUT/"scaled_local_projection_replay.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"n0":{"status":n0["new_policy_status_counts"],"iterations":n0["new_policy_iteration_distribution"],
                              "native_to_scaled":n0["native_vs_scaled_policy"]["global_l2_relative_to_native"],
                              "scaled_to_strict":n0["scaled_policy_vs_strict"]["global_l2_relative_tangent"],
                              "max_per_tet_scaled_to_strict":n0["scaled_policy_vs_strict"]["max_per_tet_l2_relative_tangent"]},
                      "n63":{"status":n63counts,"iterations":{"mean":float(np.mean([x["iterations"] for x in n63_results])),
                                 "max":int(max(x["iterations"] for x in n63_results))},
                             "native_to_strict":n63_rel,"scaled_to_strict":n63_scaled_rel,
                             "max_per_tet_scaled_to_strict":max(x["relative_l2_to_strict"] for x in n63_per)}},
                     indent=2))

if __name__ == "__main__":
    main()

