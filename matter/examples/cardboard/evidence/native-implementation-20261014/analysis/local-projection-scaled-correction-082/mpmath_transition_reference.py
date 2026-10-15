#!/usr/bin/env python3
"""High-precision independent root/tangent oracle for selected N0 transition tets."""
from pathlib import Path
import sys, json, hashlib, struct, time
import numpy as np
import mpmath as mp
BASE=Path("/Users/home/cardboard-implementation-20261014")
ANALYSIS=BASE/"analysis/bilinear15-fp64-replay-267"
sys.path.insert(0,str(ANALYSIS))
import bilinear15_reference as ref
ARR=np.load(ANALYSIS/"early-output/arrays.npz")
CAP=BASE/"solver/sourcefit15-fg32-control1-capture-cycle-258/operator-capture"
OUT=Path(__file__).parent
IDS=[750,751,758,608,763,1722,1723,1704]
REF_TOL=mp.mpf("1e-60")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mpv(x):
    return mp.mpf(float(x))

def mp_mat(a):
    return [[mpv(v) for v in row] for row in a]

def residual_mp(p,f,old,z):
    eps=p[26]; qfloor=p[25]
    s=z[6:12]
    a,b,c,d,e,v=s
    q=mp.sqrt(p[11]*(b-c)**2+p[12]*(c-a)**2+p[13]*(a-b)**2+
              2*p[14]*d*d+2*p[15]*e*e+2*p[16]*v*v+qfloor*qfloor)
    flow=[
        (p[12]*(a-c)+p[13]*(a-b))/q,
        (p[11]*(b-c)+p[13]*(b-a))/q,
        (p[11]*(c-b)+p[12]*(c-a))/q,
        p[14]*d/q,p[15]*e/q,p[16]*v/q]
    dl=z[12]-old[12]
    # Green strain components (11,22,33,23,13,12).
    eM=[[mp.mpf("0") for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            eM[i][j]=(sum(f[k][i]*f[k][j] for k in range(3))-(1 if i==j else 0))/2
    strain=[eM[0][0],eM[1][1],eM[2][2],eM[1][2],eM[0][2],eM[0][1]]
    C=[[p[1],p[4],p[5],0,0,0],
       [p[4],p[2],p[6],0,0,0],
       [p[5],p[6],p[3],0,0,0],
       [0,0,0,2*p[9],0,0],
       [0,0,0,0,2*p[8],0],
       [0,0,0,0,0,2*p[7]]]
    Ce=[[C[i][j]/p[10] for j in range(6)] for i in range(6)]
    rr=[mp.mpf("0")]*15
    for i in range(6):
        rr[i]=z[i]-old[i]-dl*flow[i]
        rr[6+i]=s[i]-sum(Ce[i][j]*(strain[j]-z[j]) for j in range(6))
    pos=[(s[i]+mp.sqrt(s[i]*s[i]+p[23]*p[23]))/2 for i in range(3)]
    den=sum(x*x for x in pos)+qfloor*qfloor
    weights=[x*x/den for x in pos]
    rr[13]=z[13]-old[13]-dl*weights[0]
    rr[14]=z[14]-old[14]-dl*weights[1]
    y=1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]-q
    root=mp.sqrt(dl*dl+y*y+eps*eps)
    summ=dl+y
    # Exact rationalization for the cancellation branch used by production.
    if mp.re(summ)>0:
        rr[12]=(eps*eps-2*dl*y)/(root+summ)
    else:
        rr[12]=root-dl-y
    return [p[27]*x for x in rr]

def jacobian_mp(p,f,old,z):
    result=[]
    for i in range(15):
        row=[]
        for j in range(15):
            def scalar(x):
                zz=list(z);zz[j]=x
                return residual_mp(p,f,old,zz)[i]
            row.append(mp.diff(scalar,z[j],method="step"))
        result.append(row)
    return mp.matrix(result)

def project_mp(p,f,old,z0,dps):
    mp.mp.dps=dps
    p=[mpv(v) for v in p];old=[mpv(v) for v in old]
    f=mp_mat(f);z=[mpv(v) for v in z0]
    stop = mp.mpf("1e-35") if dps <= 50 else mp.mpf("1e-60")
    for it in range(40):
        r=residual_mp(p,f,old,z)
        rn=max(abs(x) for x in r); sn=max(abs(x) for x in z)
        if rn <= stop*(1+sn):
            return z,{"iterations":it,"residual_inf_scaled":str(rn),"state_inf":str(sn)}
        J=jacobian_mp(p,f,old,z)
        d=mp.lu_solve(J,mp.matrix([-x for x in r]))
        alpha=mp.mpf(1); accepted=False
        for bt in range(64):
            trial=[z[i]+alpha*d[i] for i in range(15)]
            tr=residual_mp(p,f,old,trial)
            trn=max(abs(x) for x in tr)
            if mp.isfinite(trn) and trn < rn:
                z=trial;accepted=True;break
            alpha/=2
        if not accepted:
            raise RuntimeError(f"Armijo failed dps={dps} iter={it}, rn={mp.nstr(rn,8)}")
    raise RuntimeError(f"iteration exhausted dps={dps}")

def stress_mp(p,f,z):
    # StVK stress in local components; P=F*S.
    p=[mpv(v) for v in p]
    eM=[[mp.mpf("0") for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            eM[i][j]=(sum(f[k][i]*f[k][j] for k in range(3))-(1 if i==j else 0))/2
    C=[[p[1],p[4],p[5],0,0,0],
       [p[4],p[2],p[6],0,0,0],
       [p[5],p[6],p[3],0,0,0],
       [0,0,0,2*p[9],0,0],
       [0,0,0,0,2*p[8],0],
       [0,0,0,0,0,2*p[7]]]
    em=[eM[0][0]-z[0],eM[1][1]-z[1],eM[2][2]-z[2],eM[1][2]-z[3],eM[0][2]-z[4],eM[0][1]-z[5]]
    sv=[sum(C[i][j]*em[j] for j in range(6)) for i in range(6)]
    S=[[sv[0],sv[5],sv[4]],[sv[5],sv[1],sv[3]],[sv[4],sv[3],sv[2]]]
    return mp.matrix([[sum(f[i][k]*S[k][j] for k in range(3)) for j in range(3)] for i in range(3)])

def tangent_mp(p,f,old,z,h,dps):
    mp.mp.dps=dps
    p=[mpv(v) for v in p];old=[mpv(v) for v in old]
    f=mp_mat(f);h=mp_mat(h);z=[mpv(v) for v in z]
    J=jacobian_mp(p,f,old,z)
    t=mp.mpf("1e-20")
    fm=[[f[i][j]-t*h[i][j] for j in range(3)] for i in range(3)]
    fp=[[f[i][j]+t*h[i][j] for j in range(3)] for i in range(3)]
    rm=residual_mp(p,fm,old,z)
    rp=residual_mp(p,fp,old,z)
    rf=mp.matrix([(rp[i]-rm[i])/(2*t) for i in range(15)])
    dz=mp.lu_solve(J,-rf)
    # Central differences at 80-digit precision; this is an independent
    # directional derivative of the complete implicit stress chain.
    tau=mp.mpf("1e-20")
    def directional_stress(sign):
        ff=[[f[i][j]+sign*tau*h[i][j] for j in range(3)] for i in range(3)]
        zz=[z[i]+sign*tau*dz[i] for i in range(15)]
        return stress_mp(p,ff,zz)
    plus=directional_stress(1);minus=directional_stress(-1)
    out=[]
    for i in range(3):
        row=[]
        for j in range(3): row.append((plus[i,j]-minus[i,j])/(2*tau))
        out.append(row)
    return out

def main():
    t0=time.time(); out=[]; d=json.load(open(ANALYSIS/"diagnosis-002/report.json"))
    selected=[]; ids_by_tet={}
    for line in CAP.read_text().splitlines():
        rec=json.loads(line)
        if rec.get("record")=="tetrahedron":
            ids_by_tet.setdefault(int(rec["tet_id"]),rec)
    # N0 arrays were made from the exact widened FP32 words and same capture order.
    tets=[r for r in [json.loads(x) for x in CAP.read_text().splitlines()] if r.get("record")=="tetrahedron" and r.get("material_index") in (0,1)]
    for tid in IDS:
        ix=next(i for i,r in enumerate(tets) if int(r["tet_id"])==tid and r["environment"]==0)
        p=ARR["p"][ix];f=ARR["F"][ix];old=ARR["old"][ix];z0=ARR["captured_candidate"][ix];h=ARR["direction"][ix]
        row={"array_index":ix,"tet_id":tid,"material_index":int(tets[ix]["material_index"]),
             "environment":0,"inputs_fp32_words":{
                "p":[float(x) for x in p],"F":[[float(x) for x in rr] for rr in f],
                "accepted_state":[float(x) for x in old],"candidate":[float(x) for x in z0],
                "direction":[[float(x) for x in rr] for rr in h]},
             "input_word_sha256":hashlib.sha256(np.asarray(np.concatenate([p,f.reshape(-1),old,z0,h.reshape(-1)]),dtype=np.float64).tobytes()).hexdigest(),
             "input_float32_bits_sha256":hashlib.sha256(np.asarray(np.concatenate([p,f.reshape(-1),old,z0,h.reshape(-1)]),dtype=np.float32).tobytes()).hexdigest()}
        sols={}; roots_mp={}; tangents_mp={}
        for dps in [50,80]:
            z,meta=project_mp(p,f,old,z0,dps)
            T=tangent_mp(p,f,old,z,h,dps)
            roots_mp[dps]=z; tangents_mp[dps]=T
            zf=np.array([float(v) for v in z])
            tf=np.array([[float(x) for x in rr] for rr in T])
            sols[str(dps)]={"root":zf.tolist(),"iterations":meta["iterations"],"residual_inf_scaled":meta["residual_inf_scaled"],
                            "tangent":tf.tolist()}
        mp.mp.dps=80
        z50=np.asarray(sols["50"]["root"]);z80=np.asarray(sols["80"]["root"])
        t50=np.asarray(sols["50"]["tangent"]);t80=np.asarray(sols["80"]["tangent"])
        root_mp_diff=max(abs(a-b) for a,b in zip(roots_mp[50],roots_mp[80]))
        tangent_mp_diff=mp.sqrt(sum((tangents_mp[50][i][j]-tangents_mp[80][i][j])**2
                                   for i in range(3) for j in range(3)))
        tangent_mp_norm=mp.sqrt(sum(tangents_mp[80][i][j]**2 for i in range(3) for j in range(3)))
        nref=ARR["nativeT"][ix]; native50=np.linalg.norm(t50-nref)/max(np.linalg.norm(nref),1e-300)
        native80=np.linalg.norm(t80-nref)/max(np.linalg.norm(nref),1e-300)
        row["mpmath_50_vs_80"]={"root_inf_difference_float64_export":float(np.max(np.abs(z50-z80))),
            "tangent_relative_l2_difference_float64_export":float(np.linalg.norm(t50-t80)/max(np.linalg.norm(t80),1e-300)),
            "root_inf_difference_high_precision":mp.nstr(root_mp_diff,24),
            "tangent_relative_l2_difference_high_precision":mp.nstr(tangent_mp_diff/tangent_mp_norm,24)}
        rp_spec=__import__("importlib.util").util.spec_from_file_location("scaled_replay",OUT/"replay_scaled_policy.py")
        rp=__import__("importlib.util").util.module_from_spec(rp_spec);rp_spec.loader.exec_module(rp)
        zscaled,scaled_meta=rp.project_scaled(p,f,old,z0)
        Tscaled=ref.tangent(p,f,old,zscaled,h)
        row["scaled_policy_candidate_vs_mpmath80"]={"tangent_relative_l2":float(np.linalg.norm(Tscaled-t80)/max(np.linalg.norm(t80),1e-300)),
            "state_inf_to_mpmath80":float(np.max(np.abs(zscaled-z80))),"policy":scaled_meta}
        row["reference_stability_50_vs_80"]=row["mpmath_50_vs_80"]
        row["native_tangent_relative_l2"]={"mp50":float(native50),"mp80":float(native80)}
        row["production_capture_projection_diagnostics"]=tets[ix]["center"]["projection"]
        row["captured_candidate_fb"]={"yield_residual":float(ref.residual(p,f,old,z0)[12]),"state_multiplier":float(z0[12]-old[12]),
           "epsilon":float(p[26])}
        row["solutions"]=sols
        out.append(row)
    report={"schema":"numi.cardboard.bilinear15-mpmath-transition-reference.v1",
       "classification":"Independent arbitrary-precision verification for selected captured N0 material transitions; equation replay only, not a production or full-coupon qualification.",
       "input_capture":str(CAP),"input_capture_sha256":sha(CAP),
       "reference_equations_sha256":sha(ANALYSIS/"bilinear15_reference.py"),
       "source_snapshot":"source-portable-bct-binding-tests-081",
       "source_common_sha256":sha(BASE/"snapshots/source-portable-bct-binding-tests-081/matter/src/metal/common.metalinc"),
       "input_rule":"NPZ arrays generated by casting each captured float32 high/low word to float32, widening to FP64, then summing paired high+low for F and candidate.",
       "mpmath_version":mp.__version__,"dps":[50,80],"rows":out,"seconds":time.time()-t0}
    (OUT/"mpmath_transition_reference.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"seconds":report["seconds"],"rows":[{"tet_id":r["tet_id"],"root_diff":r["mpmath_50_vs_80"]["root_inf_difference_high_precision"],
         "tangent_diff":r["mpmath_50_vs_80"]["tangent_relative_l2_difference_high_precision"],
         "native_rel":r["native_tangent_relative_l2"]} for r in out]},indent=2))
if __name__=="__main__": main()

