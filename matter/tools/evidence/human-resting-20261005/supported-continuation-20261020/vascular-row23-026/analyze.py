#!/usr/bin/env python3
"""Decode bounded native vascular trace and audit dense source rows."""
import argparse, hashlib, json, math, pathlib, struct

ROOT = pathlib.Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
RUN = ROOT / "build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-vascular-row23-026"
GRAPH = pathlib.Path("/Users/numi/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json")

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def f32(x): return struct.unpack("<f",struct.pack("<f",x))[0]
def floats(h):
    b=bytes.fromhex(h)
    return struct.unpack("<"+"f"*(len(b)//4),b)
def vecs(h):
    a=floats(h)
    return [a[i:i+4] for i in range(0,len(a),4)]
def records(path,root,tick):
    out=[]
    with open(path,errors="replace") as f:
        for ln,line in enumerate(f,1):
            pos=line.find("vascular_iterate=")
            if pos<0: continue
            try: d=json.loads(line[pos+17:])
            except Exception: continue
            if d.get("root")==root and d.get("microtick")==tick:
                d["_line"]=ln; out.append(d)
    return out
def pressure(comp,v,dv,e):
    disp=v-float(comp["reference_volume_m3"])
    if comp.get("pressure_law")=="atan_compliance":
        m=float(comp["maximum_volume_displacement_m3"]); pi=float(comp["source_pi"]); c=float(comp["compliance_m3_per_pa"])
        t=math.tan((.5*pi)*(disp/m))
        return (float(comp.get("external_pressure_pa",0))+float(comp.get("reference_pressure_pa",0))+(2*m/pi/c)*t,
                ((1+t*t)/c)*dv)
    return (float(comp.get("external_pressure_pa",0))+float(comp.get("reference_pressure_pa",0))+disp*e,e*dv)
def evaluate(x,accepted,p,working,g,el,dt):
    comps=g["compartments"]; edges=g["connections"]; C=len(comps); N=C+len(edges)
    vols=[x[i][0]*float(c["volume_scale_m3"]) for i,c in enumerate(comps)]
    dvol=[p[i][0]*float(c["volume_scale_m3"]) for i,c in enumerate(comps)]
    prs=[pressure(c,vols[i],dvol[i],el[i]) for i,c in enumerate(comps)]
    # The native run appends the registered aortic wall as one deforming
    # cavity on stable compartment 7 (abdominal aorta). The pressure is the
    # extra last unknown, not that compartment's linear-compliance pressure.
    cavity_row=N
    if len(x)==N+1:
        cavity_pressure_scale=13300.0  # NumiHumanRestingAorticWall::build Config
        prs[6]=(x[cavity_row][0]*cavity_pressure_scale,
                p[cavity_row][0]*cavity_pressure_scale)
    F=[0.0]*N; J=[0.0]*N
    for i,c in enumerate(comps):
        vs=float(c["volume_scale_m3"])
        F[i]=vs*(x[i][0]-accepted[i][0]); J[i]=vs*p[i][0]
        for k,e in enumerate(edges):
            row=C+k
            if working[row]: continue
            qscale=float(e["flow_scale_m3_per_s"])
            sign=(1.0 if int(e["from"])-1==i else (-1.0 if int(e["to"])-1==i else 0.0))
            if sign:
                F[i]+=dt*sign*x[row][0]*qscale
                J[i]+=dt*sign*p[row][0]*qscale
        F[i]/=vs
        J[i]/=vs
    for k,e in enumerate(edges):
        row=C+k
        if working[row]:
            F[row]=J[row]=0.0; continue
        qs=float(e["flow_scale_m3_per_s"]); q=x[row][0]*qs; dq=p[row][0]*qs
        a=int(e["from"])-1; b=int(e["to"])-1
        pa,dpa=prs[a]; pb,dpb=prs[b]
        if e["flow_law"]=="starling_resistance":
            floor=float(e.get("downstream_pressure_floor_pa",0.0))
            if pb<=floor: pb,dpb=floor,0.0
        dp=pa-pb; ddp=dpa-dpb
        law=e["flow_law"]
        if law=="resistance_inertance":
            R=float(e["resistance_pa_s_per_m3"]); L=float(e["inertance_pa_s2_per_m3"])
            q0=accepted[row][0]*qs
            raw=L*(q-q0)/dt+R*q-dp
            tangent=(L/dt+R)*dq-ddp
        elif law=="directional_resistance":
            R=float(e["reverse_resistance_pa_s_per_m3"] if q<0 or (q==0 and dp<0) else e["resistance_pa_s_per_m3"])
            raw=R*q-dp; tangent=R*dq-ddp
        elif law in ("one_way_orifice","one_way_resistance","starling_resistance"):
            fp=float(e["pressure_scale_pa"])/qs
            closed=q*fp
            if law=="one_way_orifice":
                cv=float(e["orifice_coefficient_m3_per_s_sqrt_pa"])
                opened=(q/cv)*(q/cv)-dp; opened_t=2*q/(cv*cv)*dq-ddp
            else:
                R=float(e["resistance_pa_s_per_m3"])
                opened=R*q-dp; opened_t=R*dq-ddp
            raw,tangent=(closed,fp*dq) if closed<=opened else (opened,opened_t)
        else: raise ValueError(law)
        z=float(e["pressure_scale_pa"])
        F[row]=raw/z; J[row]=tangent/z
    return F,J,prs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log",type=pathlib.Path,default=RUN/"stderr.log")
    ap.add_argument("--graph",type=pathlib.Path,default=GRAPH)
    ap.add_argument("--root",type=int,default=20); ap.add_argument("--tick",type=int,default=3)
    ap.add_argument("--frame-dt",type=float,default=.000125); ap.add_argument("--max-rate-exponent",type=int,default=2)
    ap.add_argument("--out",type=pathlib.Path)
    z=ap.parse_args(); g=json.load(open(z.graph)); dt=z.frame_dt/(1<<z.max_rate_exponent)
    invocation=RUN/"invocation.json"; provenance=RUN/"provenance.json"
    result={"schema":"native_vascular_row23_audit_v1",
            "input":{"stderr_path":str(z.log),"stderr_sha256":sha(z.log),"graph_path":str(z.graph),"graph_sha256":sha(z.graph),
                     "invocation_path":str(invocation),"invocation_sha256":sha(invocation),
                     "provenance_path":str(provenance),"provenance_sha256":sha(provenance),
                     "frame_timestep_s":z.frame_dt,"max_rate_exponent":z.max_rate_exponent,"microstep_dt_s":dt},
            "selector":{"root":z.root,"microtick":z.tick},"records":[]}
    for d in records(z.log,z.root,z.tick):
        a=d["arenas"]; cand=vecs(a["candidate"]); acc=vecs(a["accepted"]); sol=vecs(a["solution"])
        resid=vecs(a["residual"]); el=floats(a["elastance"])
        wb=struct.unpack("<"+"I"*(len(bytes.fromhex(a["working_set"]))//4),bytes.fromhex(a["working_set"]))
        C=len(g["compartments"]); row=C+2
        stat=struct.unpack("<8I4f",bytes.fromhex(a["status"])[:48])
        F,J,prs=evaluate(cand,acc,sol,wb,g,el,dt)
        # Reproduce the ordinary float32 line-search affine update for this row.
        alpha=float(stat[8]) if stat[0] else None
        one={"iteration":d.get("iteration"),"stage":d.get("stage"),"stderr_line":d.get("_line"),
             "arena_rows":len(cand),"known_source_unknowns":C+len(g["connections"]),
             "status_code":stat[0],"status_row":stat[3],"status_diagnostics":list(stat[8:12]),
             "flow_row":row,"flow_name":g["connections"][2]["id"],"from":g["compartments"][1]["id"],
             "to":g["compartments"][2]["id"],"candidate_normalized":[cand[i][0] for i in (1,2,row)],
             "accepted_normalized":[acc[i][0] for i in (1,2,row)],
             "correction_normalized":[sol[i][0] for i in (1,2,row)],
             "pressure_from_Pa":prs[1][0],"pressure_to_Pa":prs[2][0],
             "RQ_minus_deltaP_Pa":F[row]*float(g["connections"][2]["pressure_scale_pa"]),
             "equation_row23":F[row],"residual_arena_row23":resid[row][0],
             "residual_arena_plus_equation":resid[row][0]+F[row],
             "Jp_row23_per_alpha":J[row],
             "row23_merit_slope_continuous":2.0*F[row]*J[row],
             "dP_from_per_alpha_Pa":prs[1][1],"dP_to_per_alpha_Pa":prs[2][1],
             "dQ_per_alpha_m3_s":sol[row][0]*float(g["connections"][2]["flow_scale_m3_per_s"]),
             "row23_merit_base":F[row]*F[row],
             "aggregate_known_rows_merit_base":sum(v*v for v in F),
             "aggregate_known_rows_merit_directional_derivative":2*sum(x*y for x,y in zip(F,J)),
             "fgmres_diagnostics":list(floats(a["fgmres"])),
             "cavity_pressure_row_index":len(F),
             "cavity_pressure_row_normalized":cand[len(F)][0] if len(cand)>len(F) else None,
             "cavity_pressure_scale_Pa":13300.0 if len(cand)>len(F) else None,
             "cavity_owner_compartment_stable_id":7 if len(cand)>len(F) else None,
             "cavity_geometry_equation_omitted_from_reconstruction":len(cand)>len(F),
             "residual_dot_Jp":sum(resid[i][0]*J[i] for i in range(min(len(resid),len(J))))}
        if alpha is not None:
            tr=[list(r) for r in cand]
            for i in range(len(tr)):
                if i>=C and wb[i]:
                    tr[i][0]=f32(f32(1-alpha)*cand[i][0])
                else:
                    tr[i][0]=f32(cand[i][0]+f32(alpha*sol[i][0]))
            Ft,Jt,pt=evaluate(tr,acc,sol,wb,g,el,dt)
            one.update({"tested_alpha":alpha,"trial_row_normalized":[tr[i][0] for i in (1,2,row,45)],
                        "trial_RQ_minus_deltaP_Pa":Ft[row]*float(g["connections"][2]["pressure_scale_pa"]),
                        "trial_equation_row23":Ft[row],"trial_row23_merit":Ft[row]*Ft[row],
                        "trial_known_rows_merit":sum(v*v for v in Ft),
                        "row23_secant_per_alpha":(Ft[row]-F[row])/alpha if alpha else None,
                        "row23_merit_change":Ft[row]*Ft[row]-F[row]*F[row],
                        "known_rows_merit_change":sum(v*v for v in Ft)-sum(v*v for v in F),
                        "native_merit_base":stat[9],"native_merit_trial":stat[10],
                        "native_minus_known_rows_merit_base":stat[9]-sum(v*v for v in F),
                        "native_minus_known_rows_merit_trial":stat[10]-sum(v*v for v in Ft),
                        "native_minus_known_rows_merit_delta":(stat[10]-sum(v*v for v in Ft))-(stat[9]-sum(v*v for v in F)),
                        "merit_rows_omitted":len(cand)-(C+len(g["connections"]))})
        result["records"].append(one)
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if z.out: z.out.write_text(text)
    print(text,end="")
if __name__=="__main__": main()
