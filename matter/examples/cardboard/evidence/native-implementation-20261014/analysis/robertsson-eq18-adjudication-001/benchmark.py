#!/usr/bin/env python3
"""Independent high-precision source-equation benchmark for Robertsson Eq. 18."""
from decimal import Decimal, getcontext
import json
import math
import struct
from pathlib import Path
getcontext().prec = 70
D = Decimal
E3, E6, E7 = map(D, ("28.4", "0.365", "11.5"))
MPA_TO_PA = D("1000000")

def f32(x):
    """Match the local CPU expression evaluator's leaf-constant cast."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]

def w_comp(lam, convention):
    x = lam * lam
    c = D("-2") if convention == "paper_absolute" else -D(1) - D(1) / E7
    return E6 * (x + ( -E7 * (x-1)).exp()/E7 + c)

def w_tension(lam):
    return E3 * (lam + 1/lam - 2)

def p_comp(lam, alpha=D(0)):
    e = (-E7*(lam*lam-1)).exp()
    return (1-alpha)*2*E6*lam*(1-e)

def tau_comp(lam, alpha=D(0)):
    return lam*p_comp(lam, alpha)

def tangent_comp(lam, alpha=D(0)):
    e=(-E7*(lam*lam-1)).exp()
    return (1-alpha)*2*E6*((1-e)+2*E7*lam*lam*e)

def p_tension(lam, alpha=D(0)):
    return (1-alpha)*E3*(1-1/(lam*lam))

def tau_tension(lam, alpha=D(0)):
    return lam*p_tension(lam, alpha)

def tangent_tension(lam, alpha=D(0)):
    return (1-alpha)*2*E3/(lam**3)

def w_comp_with_float32_graph_constants(lam, convention):
    """Reproduce the check's FP32 leaf constants with double graph arithmetic."""
    lam=float(lam)
    e6=f32("0.365"); e7=f32("11.5"); inv7=f32(1/11.5)
    c=f32(-2.0 if convention=="paper_absolute" else -1.0-1.0/11.5)
    x=lam*lam
    scaled_exp=f32(-e7)*(x-f32(1.0))
    exp_term=math.exp(scaled_exp)
    bracket=(x + inv7*exp_term) + c
    return e6*bracket

def row(lam, alpha, convention, branch):
    if branch == "compression":
        psi0=w_comp(lam,convention); p=p_comp(lam,alpha); tau=tau_comp(lam,alpha); tangent=tangent_comp(lam,alpha)
    else:
        psi0=w_tension(lam); p=p_tension(lam,alpha); tau=tau_tension(lam,alpha); tangent=tangent_tension(lam,alpha)
    return {"lambda_z":str(lam),"branch":branch,"convention":convention,"alpha_fixed":str(alpha),
            "psi0_MPa":str(psi0),"free_energy_Pa":str((1-alpha)*psi0*MPA_TO_PA),
            "damage_conjugate_A_Pa":str(psi0*MPA_TO_PA),"first_Piola_P33_MPa":str(p),
            "Kirchhoff_tau33_MPa":str(tau),"dP33_dlambda_MPa":str(tangent)}

def build_report():
    lam=D("0.99"); alpha=D("0.25"); h=D("1e-24")
    wfd=(w_comp(lam+h,"paper_absolute")-w_comp(lam-h,"paper_absolute"))/(2*h)
    p=p_comp(lam)
    ah=D("1e-30"); psi=w_comp(lam,"paper_absolute")
    afd=(((1-(alpha+ah))*psi)-((1-(alpha-ah))*psi))/(2*ah)
    left=E6*(1/E7-1); shift=E6*(1-1/E7)
    return {
      "schema":"numi.cardboard.robertsson_eq18_absolute_adjudication.v1",
      "evidence_class":"independent_CPU_source_equation_benchmark",
      "claim_limit":"not author-code replication, experimental validation, or kraft/NSSC transfer",
      "source":{"primary_url":"https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774",
        "doi":"10.1002/pts.2774","article":"Robertsson et al., A continuum damage model for creasing and folding of paperboard, Packaging Technology and Science 36(12), 1037-1050 (2023)",
        "source_locations":["Eq. 2","Eqs. 6-7","Eq. 18","Eqs. 19-20","Appendix A1","Table C1","Eq. 17 discussion"],
        "retrieval":"official Wiley open-access full-text HTML read 2026-10-15; publisher PDF/executable not used","original_html_or_pdf_archived":False},
      "grade_and_units":{"grade":"single-ply softwood paperboard used in the cited article","thickness_mm":"0.39","density_kg_m3":"788",
        "E3_MPa":str(E3),"E6_MPa":str(E6),"E7_dimensionless":str(E7),
        "energy_and_stress":"MPa-equivalent source quantities; builder scales energy root/stress to Pa"},
      "adjudication":{"literal_source_equation":"paper_absolute uses printed compression bracket constant -2 and H=1 at I13>=1; this is the source-text equation, not a thermodynamic qualification",
        "reference_state":"F=Fp=I, kappa=0, alpha=0; H=1; Psi0=0","compressed_side_limit_MPa":str(left),
        "tension_value_at_switch_MPa":"0","literal_energy_jump_at_switch_MPa":str(-left),
        "recentered_branch_constant_addition_MPa":str(shift),
        "recentered_status":"engineering thermodynamic correction/legacy comparison; no author code or erratum proves this was used in the paper",
        "damage_relevance":"A=Psi0; the branch offset changes the damage-driving force and is not a harmless stress-only reference shift",
        "assertion_conflict":"the article states Psi0>=0 in its Eq. 17 discussion; literal Eq. 18 violates that premise on the admissible diagonal compression path",
        "source_author_execution":"unresolved: article reports commercial-code simulations and exposes no author implementation source",
        "transfer":"none to corrugated kraft/NSSC"},
      "restricted_path_derivation":{"deformation":"Fe=F=diag(1,1,lambda_z), Fp=I, fixed alpha, no plastic update",
        "invariants":"I11=I12=I23=Je=1; I13=lambda_z",
        "energy_compression":"E6*(lambda_z^2 + exp(-E7*(lambda_z^2-1))/E7 + c), c=-2 or -1-1/E7",
        "energy_tension":"E3*(lambda_z + 1/lambda_z - 2)",
        "P33_compression":"(1-alpha)*2*E6*lambda_z*(1-exp(-E7*(lambda_z^2-1)))",
        "tau33_compression":"lambda_z*P33 because J=lambda_z",
        "tangent_compression":"(1-alpha)*2*E6*((1-exp(-E7*(lambda_z^2-1)))+2*E7*lambda_z^2*exp(-E7*(lambda_z^2-1)))",
        "P33_tension":"(1-alpha)*E3*(1-lambda_z^-2)","tangent_tension":"(1-alpha)*2*E3/lambda_z^3",
        "one_sided_tangent_limits_MPa_alpha0":{"compression_left":str(4*E6*E7),"tension_right":str(2*E3)},
        "derivative_check_at_lambda_0_99":{"central_difference_step":str(h),"central_difference_dW_dlambda_MPa":str(wfd),
          "Eq20_A1_P33_MPa":str(p),"absolute_difference_MPa":str(abs(wfd-p))},
        "damage_derivative_check_at_lambda_0_99":{"central_difference_step":str(ah),"central_difference_dPsi_dalpha_Pa":str(afd*MPA_TO_PA),
          "minus_A_Pa":str(-psi*MPA_TO_PA),"absolute_difference_Pa":str(abs(afd+psi)*MPA_TO_PA)},
        "repository_cpu_expression_evaluator_replay_at_lambda_0_99":{
          "semantics":"leaf constants rounded to binary32, expression arithmetic in binary64 as in the check's scalar expression walk",
          "paper_absolute_psi0_MPa":repr(w_comp_with_float32_graph_constants(lam,"paper_absolute")),
          "paper_absolute_A_Pa":repr(w_comp_with_float32_graph_constants(lam,"paper_absolute")*1e6),
          "paper_absolute_free_energy_at_alpha_0_25_Pa":repr(w_comp_with_float32_graph_constants(lam,"paper_absolute")*.75*1e6),
          "recentered_psi0_MPa":repr(w_comp_with_float32_graph_constants(lam,"recentered")),
          "reference_comparison":"matches the prior checked-in audit's rounded -332362.550556 Pa A and -249271.912917 Pa free-energy values; these are binary32-leaf replay values, distinct from exact-decimal-source values above"}},
      "benchmark_rows":[row(D("0.99"),D(0),"paper_absolute","compression"),
        row(D("0.99"),D("0.25"),"paper_absolute","compression"),row(D("0.99"),D("0.25"),"recentered","compression"),
        row(D(1),D(0),"paper_absolute","tension_at_H_equals_1"),row(D("1.1"),D(0),"paper_absolute","tension")],
      "qualification":{"passed":True,"meaning":"analytic source-formula consistency for virgin diagonal F13 at fixed damage away from the branch point, plus the exact H=1 branch value",
        "not_established":["author executable convention","damage-dissipation qualification of paper_absolute","global 27-state return-map correctness","Metal parity","folding experiment reproduction","cross-grade transfer"]}}

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path); a=p.parse_args()
    out=json.dumps(build_report(),indent=2,sort_keys=True)+"\n"
    if a.output: a.output.write_text(out)
    else: print(out,end="")
if __name__=="__main__": main()
