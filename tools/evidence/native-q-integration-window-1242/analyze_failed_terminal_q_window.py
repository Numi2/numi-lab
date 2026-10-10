#!/usr/bin/env python3
"""Hash-bound accounting for accepted q/v stages and frozen-J support diagnostics."""
import argparse, csv, ctypes, hashlib, json, math, pathlib, platform, re, struct, sys
from collections import defaultdict

DEFAULT_RUN = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/q-integration-audit-window-1240/late-window-310s-001/window/native-run")
OUT_DEFAULT = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/q-integration-audit-window-1240/late-window-analysis-preparation-001/revision-007")
Q_APP = pathlib.Path("/Users/n/numi-human-q-integration-audit-window-001/apps/numilab_human_myosim_visual_probe.mm")
L_ROOT = pathlib.Path("/Users/n/numi-human-performance-source-014")
L_SOLVE = L_ROOT / "src/metal/NumiHumanStandSolve.metalinc"
L_QUATERNION_SOURCE = L_ROOT / "src/metal/NumiHumanStand.metal"
L_OPERATOR_SOURCE = L_ROOT / "src/metal/MetalArticulatedOperator.mm"
ROOT_HEADER = L_ROOT / "include/metalrobo/compensated_translation_gpu.h"
FRICTION_HEADER = L_ROOT / "include/metalrobo/numi_human_friction.h"
SUPPORT_HEADER = L_ROOT / "include/metalrobo/NumiHumanSupport.hpp"
SUPPORT_DECODER_SOURCE = L_ROOT / "src/core/NumiHumanSupport.cpp"
BUILD_OBJECT = pathlib.Path("/Users/n/numi-human-q-integration-audit-window-build-006/obj/numilab_human_myosim_visual_probe.mm.o")
BINARY = pathlib.Path("/Users/n/numi-human-q-integration-audit-window-build-006/bin/numi-human-native")
LIBRARY = pathlib.Path("/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib")
PYTHON_EXECUTABLE = pathlib.Path(sys.executable).resolve()
_FMAF = ctypes.CDLL(None).fmaf
_FMAF.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_FMAF.restype = ctypes.c_float
CSV_NAMES = ("resting-com-q-integration.csv", "resting-com-q-index-map.csv",
             "resting-com-q-support-slip.csv", "resting-com-support-impulses.csv",
             "resting-com-momentum-diagnostic.csv")
REVISION_006_SCRIPT = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/q-integration-audit-window-1240/late-window-analysis-preparation-001/revision-006/analyze_q_window.py")
EXPECTED_REVISION_006_SHA256 = "41a069c250afb6864490e6d502b3885dc7bfc0eefaa4576dc456f948cf485723"
FAILED_PREFIX_REPORT = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/q-integration-audit-window-1240/terminal-tendon-failure-diagnosis-001/failed-accepted-prefix-comparison.json")
EXPECTED_PREFIX_SHA256 = "e2bb08a97eea90ac9ffb140f1ec04dd8c46db5539dd5dee96dde3f50e6520ef6"
EXPECTED_EXECUTION_SHA256 = "b131647d607ddd9eb02bd697b2cda67b65ba112ce7069d15123da6957e3fde5a"
EXPECTED_FAILURE = 'myosim_articulated_visual=failed error="persistent Human borrowed tendon-load snapshot disagreed with publication"'
EXPECTED_FIRST_STEP = 152501
EXPECTED_LAST_STEP = 155000

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def f32(x): return struct.unpack("<f",struct.pack("<f",float(x)))[0]
def finite(x): return math.isfinite(float(x))
def fused_f32(a,b,c): return float(_FMAF(ctypes.c_float(a),ctypes.c_float(b),ctypes.c_float(c)))
def parse_vec(s):
    v=[float(x) for x in s.split(";")]
    if not all(map(finite,v)): raise ValueError("nonfinite vector")
    return v
def norm3(v): return math.sqrt(sum(float(x)*float(x) for x in v))
def quantile(xs,q):
    if not xs:return None
    a=sorted(float(x) for x in xs); p=(len(a)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    return a[lo] if lo==hi else a[lo]*(hi-p)+a[hi]*(p-lo)
def stats(xs):
    xs=[float(x) for x in xs]
    if not xs:return {"count":0}
    return {"count":len(xs),"min":min(xs),"p50":quantile(xs,.5),"p95":quantile(xs,.95),
            "mean":sum(xs)/len(xs),"max":max(xs),"max_abs":max(abs(x) for x in xs),
            "rms":math.sqrt(sum(x*x for x in xs)/len(xs))}
def read_csv(path):
    with path.open(newline="") as f:
        rd=csv.DictReader(f); return rd.fieldnames,list(rd)
def root_xyz(fields):
    if len(fields)!=12: raise ValueError("root tuple must have 12 xyzw scalars")
    return [float(fields[a])+float(fields[4+a])+float(fields[8+a]) for a in range(3)]
def quat_mul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return [aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz]
def quat_next(q,omega,dt):
    rv=[float(dt)*float(x) for x in omega]; theta=norm3(rv)
    inc=[0.,0.,0.,1.] if theta==0 else [*(x*math.sin(theta/2)/theta for x in rv),math.cos(theta/2)]
    z=quat_mul(inc,q); n=math.sqrt(sum(x*x for x in z)); return [x/n for x in z]
def quat_angle_error(a,b):
    na=math.sqrt(sum(float(x)*float(x) for x in a))
    nb=math.sqrt(sum(float(x)*float(x) for x in b))
    if not na or not nb: return float("inf")
    an=[float(x)/na for x in a]; bn=[float(x)/nb for x in b]
    rel=quat_mul([-an[0],-an[1],-an[2],an[3]],bn)
    return 2.0*math.atan2(norm3(rel[:3]),abs(rel[3]))
def parse_nhcnt1(path):
    b=path.read_bytes()
    if len(b)<84 or b[:8]!=b"NHCNT1\0\0": raise ValueError("expected NHCNT1 support payload")
    abi,body_count,count,reserved=struct.unpack_from("<4I",b,8)
    if abi!=1 or reserved or len(b)!=84+count*48: raise ValueError("malformed NHCNT1 dimensions")
    head=struct.unpack_from("<7f",b,56); contacts=[]
    for i in range(count):
        v=struct.unpack_from("<II10f",b,84+i*48); body,geom=v[:2]; mu=v[8]
        if not finite(mu) or mu<0: raise ValueError("invalid support friction")
        contacts.append({"contact_index":i,"body_index":body,"source_geometry_index":geom,"mu":mu})
    return {"sha256":sha(path),"payload_abi":abi,"body_count":body_count,"contact_count":count,
            "ground_friction":head[6],"ground_normal":list(head[3:6]),"contacts":contacts}
def input_paths(run):
    win=run.parent
    paths=[run/"native.log",run/"run-metadata.json",run/"invocation.json",
           win/"execution.json",win/"run-declaration.json",Q_APP,L_SOLVE,L_QUATERNION_SOURCE,
           L_OPERATOR_SOURCE,ROOT_HEADER,FRICTION_HEADER,SUPPORT_HEADER,SUPPORT_DECODER_SOURCE,
           BUILD_OBJECT,BINARY,LIBRARY,PYTHON_EXECUTABLE,pathlib.Path(__file__).resolve()]
    paths += [run/x for x in CSV_NAMES]
    paths += [REVISION_006_SCRIPT, FAILED_PREFIX_REPORT]
    inv=json.loads((run/"invocation.json").read_text())
    paths += [pathlib.Path(p) for p in inv.get("asset_sha256",{})]
    return list(dict.fromkeys(paths))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",type=pathlib.Path,default=DEFAULT_RUN)
    ap.add_argument("--output-dir",type=pathlib.Path,default=OUT_DEFAULT)
    ap.add_argument("--first-step",type=int); ap.add_argument("--last-step",type=int); a=ap.parse_args()
    run=a.run_dir.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    dest=out/"failed-terminal-q-window-accounting.json"
    if dest.exists(): raise SystemExit("refusing to overwrite "+str(dest))
    paths=input_paths(run); missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise SystemExit("missing inputs: "+", ".join(missing))
    before={str(p):sha(p) for p in paths}
    exe=json.loads((run.parent/"execution.json").read_text()); meta=json.loads((run/"run-metadata.json").read_text())
    inv=json.loads((run/"invocation.json").read_text()); decl=json.loads((run.parent/"run-declaration.json").read_text())
    log=(run/"native.log").read_text(errors="replace")
    prefix=json.loads(FAILED_PREFIX_REPORT.read_text())
    execution_path=run.parent/"execution.json"
    declaration_path=run.parent/"run-declaration.json"
    if sha(execution_path)!=EXPECTED_EXECUTION_SHA256: raise SystemExit("not the pinned terminal-failure execution")
    if sha(FAILED_PREFIX_REPORT)!=EXPECTED_PREFIX_SHA256: raise SystemExit("prefix comparison hash mismatch")
    if sha(REVISION_006_SCRIPT)!=EXPECTED_REVISION_006_SHA256: raise SystemExit("revision-006 ancestry changed")
    if exe.get("returncode")!=1 or exe.get("changed_inputs")!={}: raise SystemExit("expected exit-1 failure with unchanged inputs")
    if exe.get("input_pin_count")!=85 or exe.get("declaration_sha256")!=sha(declaration_path): raise SystemExit("execution/declaration pin mismatch")
    if meta.get("exit_code")!=1 or meta.get("source_files_changed_during_run")!=[]: raise SystemExit("expected failed-run metadata with unchanged sources")
    if log.count(EXPECTED_FAILURE)!=1: raise SystemExit("exact terminal error line absent or duplicated")
    accepted_final=re.findall(r"human_standing_progress=accepted step=(\d+) simulated_seconds=([0-9.eE+-]+)",log)
    if not accepted_final or int(accepted_final[-1][0])!=EXPECTED_LAST_STEP: raise SystemExit("terminal accepted progress row missing")
    if "myosim_articulated_visual=completed" in log: raise SystemExit("unexpected successful native footer")
    if prefix.get("failed_execution_sha256")!=EXPECTED_EXECUTION_SHA256 or prefix.get("failed_run_exit_code")!=1: raise SystemExit("prefix report is for another run")
    if prefix.get("input_pins_unchanged") is not True or prefix.get("successful_run_completion") is not False: raise SystemExit("prefix report does not retain the failed-run boundary")
    qgrid=prefix.get("grids",{}).get("q",{})
    if qgrid.get("rows")!=2500 or qgrid.get("complete") is not True: raise SystemExit("prefix report does not certify 2500 complete q rows")
    if int(decl.get("accepted_steps",-1))!=EXPECTED_LAST_STEP: raise SystemExit("declaration does not describe 155000 accepted steps")
    if (run/"accepted-geometry"/f"step-{EXPECTED_LAST_STEP}.mrvpack").exists(): raise SystemExit("unexpected terminal MRVPACK in failed run")
    loaded=meta.get("loaded_metal_runtime",{})
    if not loaded.get("verified") or loaded.get("expected_path")!=str(LIBRARY) or loaded.get("expected_sha256")!=before[str(LIBRARY)]: raise SystemExit("loaded runtime pin mismatch")
    for p,expected in inv.get("asset_sha256",{}).items():
        if before.get(str(pathlib.Path(p)))!=expected: raise SystemExit("invocation asset pin mismatch: "+p)
    _,rows=read_csv(run/"resting-com-q-integration.csv")
    if not rows: raise SystemExit("empty q integration table")
    steps=[int(r["accepted_step"]) for r in rows]; first=EXPECTED_FIRST_STEP; last=EXPECTED_LAST_STEP
    if steps!=list(range(first,last+1)) or len(rows)!=2500: raise SystemExit("failed-run q window is not the exact 2500-row terminal interval")
    env=inv.get("environment",{})
    if env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT")!="1" or int(env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP",-1))!=first or int(env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP",-1))!=last: raise SystemExit("invocation window mismatch")
    q_header=re.search(r"resting_com_q_integration_audit=enabled .*?actual_float_dt_s=([0-9.eE+-]+) rows_file=resting-com-q-integration\.csv",log)
    if not q_header: raise SystemExit("Q audit activation record missing")
    dtf=float(q_header.group(1))
    if abs(dtf-0.0020000000949949026)>1e-16: raise SystemExit("native float dt differs from pinned run")
    _,imap=read_csv(run/"resting-com-q-index-map.csv"); scalar_map=[]
    for x in imap:
        if x["record_kind"]=="scalar_dof" and x["q_index_valid"]=="1":
            qi,vi=int(x["global_q_index"]),int(x["global_v_index"])
            if qi>=7: scalar_map.append((qi,vi))
    if scalar_map!=[(qi,qi-1) for qi in range(7,129)] or len(set(v for q,v in scalar_map))!=122: raise SystemExit("unexpected scalar q/v map")

    scalar_err_nonfused=[]; scalar_err_fused=[]; scalar_inc=defaultdict(list); scalar_fused_inc=defaultdict(list); scalar_proj=defaultdict(list)
    root_err=[[],[],[]]; rb_err=[[],[],[]]; rp_err=[[],[],[]]; ra_err=[[],[],[]]
    root_vel=[[],[],[]]; qproj_root=[[],[],[]]; qproj_quat=[]; qproj_scalar=[]
    vproj_linear=[[],[],[]]; vproj_angular=[[],[],[]]; vproj_scalar=[]
    qrot_pre=[]; qrot_acc=[]; momentum=[]; continuity={"q_exact_mismatches":0,"v_exact_mismatches":0}; prev=None
    mfields=[("before","source_body_linear_momentum_before"),("free_same_q","source_body_linear_momentum_free_same_q"),
             ("preprojection_velocity_same_q","source_body_linear_momentum_preprojection_velocity_same_q"),
             ("preprojection_qv","source_body_linear_momentum_preprojection_qv"),("accepted_qv","source_body_linear_momentum_accepted_qv")]
    for r in rows:
        q0=[f32(x) for x in parse_vec(r["q_before_f32_semicolon"])]
        v0=[f32(x) for x in parse_vec(r["v_before_f32_semicolon"])]
        qp=[f32(x) for x in parse_vec(r["q_preprojection_f32_semicolon"])]
        vp=[f32(x) for x in parse_vec(r["v_preprojection_f32_semicolon"])]
        qa=[f32(x) for x in parse_vec(r["q_accepted_f32_semicolon"])]
        va=[f32(x) for x in parse_vec(r["v_accepted_f32_semicolon"])]
        dt=float(r["dt_s"])
        if (len(q0),len(qp),len(qa),len(v0),len(vp),len(va))!=(129,129,129,128,128,128) or abs(dt-dtf)>1e-15: raise SystemExit("q/v shape or dt mismatch")
        for qi,vi in scalar_map:
            dt32=f32(dt); v32=f32(vp[vi]); q32=f32(q0[qi])
            inc=f32(dt32*v32)
            expected_nonfused=f32(q32+inc)
            expected_fused=fused_f32(dt32,v32,q32)
            scalar_err_nonfused.append(qp[qi]-expected_nonfused)
            scalar_err_fused.append(qp[qi]-expected_fused)
            scalar_inc[qi].append(inc)
            scalar_fused_inc[qi].append(expected_fused-q32)
            scalar_proj[qi].append(qa[qi]-qp[qi])
        rb=parse_vec(r["root_before_reference_displacement_correction_xyzw_semicolon"]); rr=parse_vec(r["root_after_reference_displacement_correction_xyzw_semicolon"])
        wb=root_xyz(rb); wa=root_xyz(rr)
        for k in range(3):
            root_err[k].append((wa[k]-wb[k])-dt*vp[k]); root_vel[k].append(dt*vp[k])
            rb_err[k].append(q0[k]-wb[k]); rp_err[k].append(qp[k]-wa[k]); ra_err[k].append(qa[k]-wa[k])
        for k in range(3): qproj_root[k].append(qa[k]-qp[k])
        qproj_quat.extend(qa[3:7][k]-qp[3:7][k] for k in range(4))
        qproj_scalar.extend(qa[qi]-qp[qi] for qi,vi in scalar_map)
        for k in range(3):
            vproj_linear[k].append(va[k]-vp[k])
            vproj_angular[k].append(va[k+3]-vp[k+3])
        vproj_scalar.extend(va[vi]-vp[vi] for qi,vi in scalar_map)
        ideal=quat_next(q0[3:7],vp[3:6],dt); qrot_pre.append(quat_angle_error(ideal,qp[3:7])); qrot_acc.append(quat_angle_error(ideal,qa[3:7]))
        momentum.append([[float(r[f"{prefix}_{axis}_kg_m_s"]) for axis in "xyz"] for name,prefix in mfields])
        if prev:
            if prev[0]!=q0: continuity["q_exact_mismatches"]+=1
            if prev[1]!=v0: continuity["v_exact_mismatches"]+=1
        prev=(qa,va)
    qfirst=[f32(x) for x in parse_vec(rows[0]["q_before_f32_semicolon"])]
    qlast=[f32(x) for x in parse_vec(rows[-1]["q_accepted_f32_semicolon"])]
    scalar_net=[]
    for qi,vi in scalar_map:
        integrated=sum(scalar_inc[qi])
        fused_integrated=sum(scalar_fused_inc[qi])
        projection=sum(scalar_proj[qi])
        observed=qlast[qi]-qfirst[qi]
        scalar_net.append({"q_index":qi,"v_index":vi,"joint_name":imap[0].get("joint_name") if False else next((x["joint_name"] for x in imap if x["record_kind"]=="scalar_dof" and x["global_q_index"]==str(qi)),None),
                           "dof_name":next((x["dof_name"] for x in imap if x["record_kind"]=="scalar_dof" and x["global_q_index"]==str(qi)),None),
                           "observed_net_coordinate_delta":observed,"sum_f32_dt_v_preprojection_nonfused_reference":integrated,
                           "sum_fmaf_dt_v_plus_q_preprojection_increment":fused_integrated,
                           "sum_accepted_minus_preprojection":projection,
                           "unclosed_fmaf_or_projection_sum_residual":observed-fused_integrated-projection})
    root_start=root_xyz(parse_vec(rows[0]["root_before_reference_displacement_correction_xyzw_semicolon"]))
    root_end=root_xyz(parse_vec(rows[-1]["root_after_reference_displacement_correction_xyzw_semicolon"]))
    root_net=[]
    for k,axis in enumerate("xyz"):
        integrated=sum(root_vel[k]); observed=root_end[k]-root_start[k]
        root_net.append({"axis":axis,"observed_net_displacement_m":observed,"sum_dt_v_preprojection_m":integrated,
                         "difference_m":observed-integrated,"sum_step_compensation_residual_m":sum(root_err[k])})
    md={}
    for i in range(4):
        d=[[momentum[j][i+1][k]-momentum[j][i][k] for k in range(3)] for j in range(len(rows))]
        md[mfields[i][0]+"_to_"+mfields[i+1][0]]={"component_kg_m_s":{a:stats([v[k] for v in d]) for k,a in enumerate("xyz")},"vector_norm_kg_m_s":stats([norm3(v) for v in d]),"interpretation":"kinematic stage delta, not force attribution"}

    sh,slips=read_csv(run/"resting-com-q-support-slip.csv")
    slipstats={}
    for col in [h for h in sh if h.endswith("_speed_m_s")]:
        key=col[len("slip_pre_step_contact_J_v_"):-len("_speed_m_s")]
        slipstats[key]={"tangent_speed_m_s":stats([float(x[col]) for x in slips])}
    pre="slip_pre_step_contact_J_v_preprojection_velocity_same_q_speed_m_s"; acc="slip_pre_step_contact_J_v_accepted_speed_m_s"
    slipstats["accepted_minus_preprojection_speed_change_m_s"]=stats([float(x[acc])-float(x[pre]) for x in slips])
    for step in steps:
        if sum(int(x["accepted_step"])==step for x in slips)!=32: raise SystemExit("support slip coverage mismatch")

    support_path=None; argv=inv.get("argv",[])
    for i,x in enumerate(argv[:-1]):
        if x=="--support-contact-payload": support_path=pathlib.Path(argv[i+1]); break
    if support_path is None: raise SystemExit("support payload absent from invocation")
    support=parse_nhcnt1(support_path); contacts={(x["body_index"],x["source_geometry_index"]):x for x in support["contacts"]}
    _,impulses=read_csv(run/"resting-com-support-impulses.csv"); ir=[x for x in impulses if first<=int(x["accepted_step"])<=last]
    grouped=defaultdict(list); basiserr=[]; fr=[]
    for x in ir:
        key=(int(x["body_index"]),int(x["source_geometry_index"]))
        if key not in contacts: raise SystemExit("impulse/contact identity mismatch")
        mu=contacts[key]["mu"]; n=float(x["normal_impulse_ns"]); t0=float(x["tangent0_impulse_ns"]); t1=float(x["tangent1_impulse_ns"]); tn=math.hypot(t0,t1); cap=mu*n
        u=tn/cap if cap>0 else (0.0 if tn==0 else float("inf"))
        world=[float(x[f"impulse_world_{a}_ns"]) for a in "xyz"]
        rec=[n*float(x[f"normal_{a}"])+t0*float(x[f"tangent0_{a}"])+t1*float(x[f"tangent1_{a}"]) for a in "xyz"]
        basiserr.append(norm3([world[k]-rec[k] for k in range(3)]))
        z={**x,"mu":mu,"tangent_norm_ns":tn,"coulomb_disk_radius_ns":cap,"tangent_to_normal_ratio":tn/n if n>0 else None,"friction_disk_utilization":u}
        fr.append(z); grouped[int(x["accepted_step"])].append(z)
    segs=[]
    for step,rr in sorted(grouped.items()):
        utils=[float(x["friction_disk_utilization"]) for x in rr]
        ratios=[x["tangent_to_normal_ratio"] for x in rr if x["tangent_to_normal_ratio"] is not None]
        segs.append({"accepted_step":step,"sample_start_step":int(rr[0]["sample_start_step"]),"segment_steps":int(rr[0]["segment_steps"]),"contacts":len(rr),
            "normal_impulse_sum_ns":sum(float(x["normal_impulse_ns"]) for x in rr),"tangent_impulse_norm_sum_ns":sum(float(x["tangent_norm_ns"]) for x in rr),
            "max_tangent_to_normal_ratio":max(ratios,default=None),"max_coulomb_disk_utilization":max(utils,default=0.0),
            "disk_violations_gt_1e-5":sum(x>1.00001 for x in utils)})

    root={"compensated_increment_minus_dt_times_preprojection_velocity_m_by_axis":{a:stats(root_err[i]) for i,a in enumerate("xyz")},
          "integrated_root_displacement_m_by_axis":{a:stats([root_xyz(parse_vec(r["root_after_reference_displacement_correction_xyzw_semicolon"]))[i]-root_xyz(parse_vec(r["root_before_reference_displacement_correction_xyzw_semicolon"]))[i] for r in rows]) for i,a in enumerate("xyz")},
          "dt_times_preprojection_root_velocity_m_by_axis":{a:stats(root_vel[i]) for i,a in enumerate("xyz")},
          "q_before_minus_root_before_world_m_by_axis":{a:stats(rb_err[i]) for i,a in enumerate("xyz")},
          "q_preprojection_minus_root_after_world_m_by_axis":{a:stats(rp_err[i]) for i,a in enumerate("xyz")},
          "q_accepted_minus_root_after_world_m_by_axis":{a:stats(ra_err[i]) for i,a in enumerate("xyz")}}
    native_simulated_s=float(accepted_final[-1][1])
    native_wall_s=float(exe["wall_seconds"])
    native_rtf=native_simulated_s/native_wall_s
    report={"schema":"numi.human.q-window-failed-terminal-diagnostic.v1","status":"validated_failed_terminal_diagnostic_not_validated_publication",
        "scope":{"run_dir":str(run),"first_accepted_step":first,"last_accepted_step":last,"row_count":len(rows),"accepted_steps_contiguous":True,
                 "window_is_full_native_run":False,"qualification_claim":False,"publication_validated":False},
        "native_failure":{"exact_error_line":EXPECTED_FAILURE,"exit_code":1,"accepted_terminal_step":EXPECTED_LAST_STEP,
            "terminal_mrvpack_present":False,"successful_terminal_footer_present":False,
            "interpretation":"The native run failed after accepting step 155000. This diagnostic does not turn the failed run into a successful publication."},
        "prefix_comparison":{"path":str(FAILED_PREFIX_REPORT),"sha256":EXPECTED_PREFIX_SHA256,
            "scope":"Hash-bound accepted-prefix comparison; its own report says successful_run_completion=false."},
        "execution":{"returncode":exe.get("returncode"),"changed_inputs":exe.get("changed_inputs"),"outer_wall_seconds":exe.get("wall_seconds"),"metadata_exit_code":meta.get("exit_code"),"metadata_wall_seconds":meta.get("wall_seconds"),"runtime_verified":loaded.get("verified")},
        "timebase":{"dt_s":dtf,"native_simulated_s":native_simulated_s,"native_wall_s":native_wall_s,"native_rtf":native_rtf,
            "simulated_time_source":"last accepted progress line, not a successful integration footer"},
        "semantics":{"accepted_submission":"Rows are emitted after each accepted physical step; before is the adjacent prior accepted state, preprojection is immediately before equality projection, accepted is the reconciled committed state.",
            "integration":"The L Metal owner writes candidate velocity, advances compensated root xyz by dt*candidateV[0:3], integrates quaternion from dt*candidateV[3:6], and updates remaining mapped scalar q as q += dt*candidateV; equality projection follows. CSV state values are round-tripped to F32 before comparison. Scalar comparison includes both a separately rounded F32 multiply/add reference and a C fmaf call with C-float arguments/results; exact agreement with fmaf is evidence consistent with contraction, not machine-code proof of the GPU instruction.",
            "root":"Root reference is fixed episode origin; displacement and correction form a normalized F32 expansion in metres. The binary64 sum of its components is an accounting reference, not a bit-for-bit replay of the GPU round-to-odd expansion/projection arithmetic. The recorded increment is compared with dt*v_pre.",
            "momentum":"Five-stage values are CPU-only mass-weighted source-body COM linear momentum in kg m/s. Stage differences are kinematic snapshots, not force or impulse attribution.",
            "support_slip":"Tangent speed is same pre-step point Jacobian times each stage velocity, projected into the fixed support-plane tangent basis. Even accepted-stage velocity uses the pre-step Jacobian, so it is not endpoint displacement/sliding.",
            "friction":"NHCNT1 contact mu is joined by body/source-geometry identity; tangent impulse norm is checked against mu times normal impulse. The observer emits only the last-physical-step impulse at each 8-step endpoint, not an interval sum; the first sample segment may begin before the Q window.",
            "causal_limit":"Exact coordinate integration closure explains q displacement from velocity plus projection bookkeeping, not why velocity is nonzero. No gravity/muscle/contact/controller causality follows from this report."},
        "failed_terminal_diagnosis":{"required_execution_sha256":EXPECTED_EXECUTION_SHA256,
            "required_prefix_report_sha256":EXPECTED_PREFIX_SHA256,"q_window_rows":2500,
            "source_hypothesis":{}},
        "source_runtime":{"q_observer_source":{"path":str(Q_APP),"sha256":before[str(Q_APP)]},"L_metal_solver":{"path":str(L_SOLVE),"sha256":before[str(L_SOLVE)]},
            "L_quaternion_helpers":{"path":str(L_QUATERNION_SOURCE),"sha256":before[str(L_QUATERNION_SOURCE)]},
            "L_operator_readback":{"path":str(L_OPERATOR_SOURCE),"sha256":before[str(L_OPERATOR_SOURCE)]},
            "compensated_root_source":{"path":str(ROOT_HEADER),"sha256":before[str(ROOT_HEADER)]},"friction_owner_source":{"path":str(FRICTION_HEADER),"sha256":before[str(FRICTION_HEADER)]},
            "support_payload_owner_source":{"path":str(SUPPORT_HEADER),"sha256":before[str(SUPPORT_HEADER)]},
            "support_payload_decoder_source":{"path":str(SUPPORT_DECODER_SOURCE),"sha256":before[str(SUPPORT_DECODER_SOURCE)]},
            "q_observer_compiled_object":{"path":str(BUILD_OBJECT),"sha256":before[str(BUILD_OBJECT)]},
            "analysis_python_runtime":{"path":str(PYTHON_EXECUTABLE),"sha256":before[str(PYTHON_EXECUTABLE)],"version":sys.version,"platform":platform.platform(),"fmaf_symbol":"ctypes.CDLL(None).fmaf; arguments and result are C float"},
            "native_binary":{"path":str(BINARY),"sha256":before[str(BINARY)]},
            "loaded_L_runtime":{"path":str(LIBRARY),"sha256":before[str(LIBRARY)]},"support_payload":support},
        "input_hashes_before":before,
        "results":{"mapped_scalar_q_nonfused_f32_reference_residual":stats(scalar_err_nonfused),
            "mapped_scalar_q_nonfused_f32_reference_exact_match_count":sum(x==0.0 for x in scalar_err_nonfused),
            "mapped_scalar_q_fused_fmaf_residual":stats(scalar_err_fused),
            "mapped_scalar_q_fused_fmaf_exact_match_count":sum(x==0.0 for x in scalar_err_fused),
            "mapped_scalar_q_fused_fmaf_comparison_count":len(scalar_err_fused),
            "scalar_q_coordinates_checked_per_step":len(scalar_map),
            "root":{**root,"window_net_accounting":root_net},
            "scalar_coordinate_window_net_accounting":scalar_net,
            "quaternion":{"ideal_rotation_vector_update_to_preprojection_angle_error_rad":stats(qrot_pre),"ideal_rotation_vector_update_to_accepted_angle_error_rad":stats(qrot_acc),"comparison_normalizes_both_quaternions":True,"native_f32_arithmetic_note":True,"interpretation":"normalized sign-invariant double-precision ideal comparator, not native bit replay"},
            "projection_delta":{"q_root_translation_m_by_axis":{a:stats(qproj_root[i]) for i,a in enumerate("xyz")},
                "q_quaternion_component_unitless":stats(qproj_quat),"q_scalar_coordinates_native_units_mixed":stats(qproj_scalar),
                "v_root_linear_m_s_by_axis":{a:stats(vproj_linear[i]) for i,a in enumerate("xyz")},
                "v_root_angular_rad_s_by_axis":{a:stats(vproj_angular[i]) for i,a in enumerate("xyz")},
                "v_scalar_dof_native_units_mixed_per_s":stats(vproj_scalar),
                "exact_adjacent_accepted_state_mismatches":continuity},
            "five_stage_source_body_momentum_norm_kg_m_s":{name:stats([norm3(x[i]) for x in momentum]) for i,(name,prefix) in enumerate(mfields)},
            "five_stage_momentum_deltas":md,
            "support":{"slip_rows":len(slips),"contacts_per_accepted_step":32,"slip_stage_tangent_speed_m_s":slipstats,
                "contact_impulse_world_basis_reconstruction_error_ns":stats(basiserr),"impulse_segments":segs,
                "friction_disk_utilization":stats([float(x["friction_disk_utilization"]) for x in fr if math.isfinite(float(x["friction_disk_utilization"]))]),
                "friction_disk_violations_gt_1e-5":sum(float(x["friction_disk_utilization"])>1.00001 for x in fr),"impulse_rows_in_window":len(ir),"impulse_sampling":"one last-physical-step impulse at each 8-step observer endpoint, not an interval sum; endpoints are listed in impulse_segments and sample_start_step shows each containing segment"}},
        "inputs_unchanged":False}
    report["failed_terminal_diagnosis"]["source_hypothesis"]={
        "finding":"The bounded Q window forces one-step submissions inside the window, while the borrowed-tendon verifier appears to infer its terminal local step count from the normal submission cap.",
        "reproduction":"With 16 steps, normal cap 8, and bounded window steps 9..16, the window scheduler submits step 16 as one step; ((stepCount-1)%8)+1 would infer 8.",
        "limit":"The run did not log compared borrowed-status counters. This is a concrete scheduling/status-count inconsistency consistent with the terminal error, not proof of which field failed."
    }
    after={str(p):sha(p) for p in paths}; report["input_hashes_after"]=after; report["inputs_unchanged"]=before==after
    if not report["inputs_unchanged"]: report["status"]="input_changed_during_analysis"
    report["analysis_script_sha256"]=sha(pathlib.Path(__file__))
    dest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"report":str(dest),"sha256":sha(dest),"window":[first,last],"rows":len(rows),
      "scalar_fused_fmaf_residual_max_abs":report["results"]["mapped_scalar_q_fused_fmaf_residual"]["max_abs"],
      "scalar_fused_fmaf_exact_matches":report["results"]["mapped_scalar_q_fused_fmaf_exact_match_count"],
      "scalar_nonfused_reference_residual_max_abs":report["results"]["mapped_scalar_q_nonfused_f32_reference_residual"]["max_abs"],
      "root_residual_max_abs_m":max(v["max_abs"] for v in root["compensated_increment_minus_dt_times_preprojection_velocity_m_by_axis"].values()),
      "slip_rows":len(slips),"impulse_rows":len(ir),"friction_violation_count":report["results"]["support"]["friction_disk_violations_gt_1e-5"],
      "inputs_unchanged":report["inputs_unchanged"]},sort_keys=True))
    return 0 if report["status"]=="validated_failed_terminal_diagnostic_not_validated_publication" and report["inputs_unchanged"] else 2
if __name__=="__main__": sys.exit(main())
